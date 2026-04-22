import abc

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import BatchData, PairedBatchData
from fme.downscaling.data.static import StaticInputs
from fme.downscaling.metrics_and_maths import filter_tensor_mapping, interpolate
from fme.downscaling.models import (
    DiffusionModel,
    ModelOutputs,
    _repeat_batch_by_samples,
    _separate_interleaved_samples,
)
from fme.downscaling.samplers import stochastic_sampler as edm_sampler


class PredictorABC(abc.ABC):
    @property
    @abc.abstractmethod
    def coarse_shape(self) -> tuple[int, int]: ...

    @property
    @abc.abstractmethod
    def static_inputs(self) -> StaticInputs | None: ...

    @property
    @abc.abstractmethod
    def full_fine_coords(self) -> LatLonCoordinates: ...

    @abc.abstractmethod
    def get_fine_coords_for_batch(self, batch: BatchData) -> LatLonCoordinates: ...

    @abc.abstractmethod
    def generate(
        self,
        coarse_data: TensorMapping,
        static_inputs: StaticInputs | None,
        n_samples: int = 1,
    ) -> tuple[TensorDict, torch.Tensor, list[torch.Tensor]]: ...

    @abc.abstractmethod
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs: ...

    @abc.abstractmethod
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        n_samples: int = 1,
    ) -> TensorDict: ...


class BasePredictor(PredictorABC):
    def __init__(self, diffusion_model: DiffusionModel):
        self.model = diffusion_model

    @property
    def coarse_shape(self) -> tuple[int, int]:
        return self.model.coarse_shape

    @property
    def static_inputs(self) -> StaticInputs | None:
        return self.model.static_inputs

    @property
    def full_fine_coords(self) -> LatLonCoordinates:
        return self.model.full_fine_coords

    @property
    def modules(self) -> torch.nn.ModuleList:
        return self.model.modules

    @property
    def downscale_factor(self) -> int:
        return self.model.downscale_factor

    @property
    def fine_shape(self) -> tuple[int, int]:
        return self.model.fine_shape

    def get_fine_coords_for_batch(self, batch: BatchData) -> LatLonCoordinates:
        return self.model.get_fine_coords_for_batch(batch)

    def _prepare_generation_inputs(
        self,
        coarse_data: TensorMapping,
        static_inputs: StaticInputs | None,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize coarse input and build random latents for generation.

        Returns:
            inputs: Normalized (and optionally interpolated) coarse input,
                repeated ``n_samples`` times along the batch dimension.
            latents: Random noise tensor shaped for the fine output grid.
        """
        inputs = self.model._get_input_from_coarse(coarse_data, static_inputs)
        inputs = _repeat_batch_by_samples(inputs, n_samples)
        coarse_input_shape = next(iter(coarse_data.values())).shape[-2:]
        outputs_shape = (
            inputs.shape[0],
            len(self.model.out_packer.names),
            *self.model._get_fine_shape(coarse_input_shape),
        )
        latents = torch.randn(outputs_shape).to(device=get_device())
        return inputs, latents

    def _postprocess_generated(
        self,
        generated_norm: torch.Tensor,
        coarse_data: TensorMapping,
        n_samples: int,
    ) -> tuple[TensorDict, torch.Tensor]:
        """Add residual, separate samples, and denormalize sampler output."""
        if self.model.config.predict_residual:
            base_prediction = interpolate(
                self.model.out_packer.pack(
                    self.model.normalizer.coarse.normalize(
                        {k: coarse_data[k] for k in self.model.out_packer.names}
                    ),
                    axis=self.model._channel_axis,
                ),
                self.model.downscale_factor,
            )
            generated_norm = generated_norm + _repeat_batch_by_samples(
                base_prediction, n_samples
            )
        generated_norm_reshaped = _separate_interleaved_samples(
            generated_norm, n_samples
        )
        generated = self.model.normalizer.fine.denormalize(
            self.model.out_packer.unpack(
                generated_norm_reshaped, axis=self.model._channel_axis
            )
        )
        return generated, generated_norm

    @torch.no_grad()
    def generate(
        self,
        coarse_data: TensorMapping,
        static_inputs: StaticInputs | None,
        n_samples: int = 1,
    ) -> tuple[TensorDict, torch.Tensor, list[torch.Tensor]]:
        inputs, latents = self._prepare_generation_inputs(
            coarse_data, static_inputs, n_samples
        )
        generated_norm, latent_steps = edm_sampler(
            self.model.module,
            latents,
            inputs,
            S_churn=self.model.config.churn,
            sigma_min=self.model.config.sigma_min,
            sigma_max=self.model.config.sigma_max,
            num_steps=self.model.config.num_diffusion_generation_steps,
        )
        generated, generated_norm = self._postprocess_generated(
            generated_norm, coarse_data, n_samples
        )
        return generated, generated_norm, latent_steps

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        _static_inputs = self.model._subset_static_if_available(batch.coarse)
        coarse, fine = batch.coarse.data, batch.fine.data
        generated, generated_norm, latent_steps = self.generate(
            coarse, _static_inputs, n_samples
        )

        targets_norm = self.model.out_packer.pack(
            self.model.normalizer.fine.normalize(dict(fine)),
            axis=self.model._channel_axis,
        )
        targets_norm = _repeat_batch_by_samples(targets_norm, n_samples)

        targets = filter_tensor_mapping(
            batch.fine.data, set(self.model.out_packer.names)
        )
        targets = {k: v.unsqueeze(1) for k, v in targets.items()}

        loss = self.model.loss(generated_norm, targets_norm)
        return ModelOutputs(
            prediction=generated,
            target=targets,
            loss=loss,
            latent_steps=latent_steps,
        )

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        n_samples: int = 1,
    ) -> TensorDict:
        _static_inputs = self.model._subset_static_if_available(batch)
        generated, _, _ = self.generate(batch.data, _static_inputs, n_samples)
        return generated
