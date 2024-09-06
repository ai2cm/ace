import dataclasses
import logging
from typing import Any, List, Mapping, Tuple, Union

import dacite
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import NullOptimization, Optimization
from fme.core.packer import Packer
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.metrics_and_maths import filter_tensor_mapping, interpolate
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.samplers import edm_sampler
from fme.downscaling.typing_ import FineResCoarseResPair


@dataclasses.dataclass
class ModelOutputs:
    prediction: TensorDict
    target: TensorMapping
    loss: torch.Tensor
    latent_steps: List[torch.Tensor] = dataclasses.field(default_factory=list)


def _tensor_mapping_to_device(
    tensor_mapping: TensorMapping, device: torch.device
) -> TensorMapping:
    return {k: v.to(device) for k, v in tensor_mapping.items()}


@dataclasses.dataclass
class PairedNormalizationConfig:
    fine: NormalizationConfig
    coarse: NormalizationConfig

    def build(
        self, in_names: List[str], out_names: List[str]
    ) -> FineResCoarseResPair[StandardNormalizer]:
        return FineResCoarseResPair[StandardNormalizer](
            coarse=self.coarse.build(in_names),
            fine=self.fine.build(out_names),
        )


@dataclasses.dataclass
class DownscalingModelConfig:
    module: ModuleRegistrySelector
    loss: LossConfig
    in_names: List[str]
    out_names: List[str]
    normalization: PairedNormalizationConfig
    use_fine_topography: bool

    def build(
        self,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        area_weights: FineResCoarseResPair[torch.Tensor],
        fine_topography: torch.Tensor,
    ) -> "Model":
        normalizer = self.normalization.build(self.in_names, self.out_names)
        loss = self.loss.build(area_weights.fine, reduction="mean")
        module = self.module.build(
            n_in_channels=len(self.in_names),
            n_out_channels=len(self.out_names),
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            fine_topography=fine_topography if self.use_fine_topography else None,
        )
        return Model(
            module,
            normalizer,
            loss,
            self.in_names,
            self.out_names,
            coarse_shape,
            downscale_factor,
            self,
        )

    def get_state(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DownscalingModelConfig":
        return dacite.from_dict(data_class=cls, data=state)

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            names=list(set(self.in_names).union(self.out_names)),
            n_timesteps=1,
            use_fine_topography=self.use_fine_topography,
        )


class Model:
    def __init__(
        self,
        module: torch.nn.Module,
        normalizer: FineResCoarseResPair[StandardNormalizer],
        loss: torch.nn.Module,
        in_names: List[str],
        out_names: List[str],
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        config: DownscalingModelConfig,
    ) -> None:
        self.coarse_shape = coarse_shape
        self.downscale_factor = downscale_factor
        dist = Distributed.get_instance()
        self.module = dist.wrap_module(module.to(get_device()))
        self.normalizer = normalizer
        self.loss = loss
        self.in_packer = Packer(in_names)
        self.out_packer = Packer(out_names)
        self.config = config
        self.null_optimization = NullOptimization()

    @property
    def modules(self) -> torch.nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return torch.nn.ModuleList([self.module])

    def train_on_batch(
        self,
        batch: FineResCoarseResPair[TensorMapping],
        optimization: Union[Optimization, NullOptimization],
    ) -> ModelOutputs:
        result = self._run_on_batch(batch, optimization)
        for k, v in result.prediction.items():
            result.prediction[k] = v.unsqueeze(1)  # insert sample dimension
        return result

    def generate_on_batch(
        self,
        batch: FineResCoarseResPair[TensorMapping],
        n_samples: int = 1,
    ) -> ModelOutputs:
        if n_samples != 1:
            raise ValueError("n_samples must be 1 for deterministic models")
        result = self._run_on_batch(batch, self.null_optimization)
        for k, v in result.prediction.items():
            result.prediction[k] = v.unsqueeze(1)  # insert sample dimension
        return result

    def _run_on_batch(
        self,
        batch: FineResCoarseResPair[TensorMapping],
        optimizer: Union[Optimization, NullOptimization],
    ) -> ModelOutputs:
        channel_axis = -3
        coarse, fine = _tensor_mapping_to_device(
            batch.coarse, get_device()
        ), _tensor_mapping_to_device(batch.fine, get_device())
        inputs_norm = self.in_packer.pack(
            self.normalizer.coarse.normalize(dict(coarse)), axis=channel_axis
        )
        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=channel_axis
        )
        predicted_norm = self.module(inputs_norm)
        loss = self.loss(predicted_norm, targets_norm)
        optimizer.step_weights(loss)
        target = filter_tensor_mapping(batch.fine, set(self.out_packer.names))
        prediction = self.normalizer.fine.denormalize(
            self.out_packer.unpack(predicted_norm, axis=channel_axis)
        )
        return ModelOutputs(
            prediction=prediction, target=target, loss=loss, latent_steps=[]
        )

    def get_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.get_state(),
            "module": self.module.state_dict(),
            "coarse_shape": self.coarse_shape,
            "downscale_factor": self.downscale_factor,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        area_weights: FineResCoarseResPair[torch.Tensor],
        fine_topography: torch.Tensor,
    ) -> "Model":
        config = DownscalingModelConfig.from_state(state["config"])
        model = config.build(
            state["coarse_shape"],
            state["downscale_factor"],
            area_weights,
            fine_topography,
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model


@dataclasses.dataclass
class DiffusionModelConfig:
    """
    This class implements or wraps the algorithms described in `EDM`_.

    .. _EDM: https://arxiv.org/abs/2206.00364

    Attributes:
        module: The module registry selector for the diffusion model.
        loss:  The loss configuration for the diffusion model.
        in_names: The input variable names for the diffusion model.
        out_names: The output variable names for the diffusion model.
        normalization: The normalization configurations for the diffusion model.
        use_fine_topography: Indicates whether to use the fine topography.
        p_mean: The mean of noise distribution used during training.
        p_std: The std of the noise distribution used during training.
        sigma_min: Min noise level for generation.
        sigma_max: Max noise level for generation.
        churn: The amount of stochasticity during generation.
        num_diffusion_generation_steps: Number of diffusion generation steps.
    """

    module: DiffusionModuleRegistrySelector
    loss: LossConfig
    in_names: List[str]
    out_names: List[str]
    normalization: PairedNormalizationConfig
    use_fine_topography: bool
    p_mean: float
    p_std: float
    sigma_min: float
    sigma_max: float
    churn: float
    num_diffusion_generation_steps: int
    predict_residual: bool

    def build(
        self,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        area_weights: FineResCoarseResPair[torch.Tensor],
        fine_topography: torch.Tensor,
    ) -> "DiffusionModel":
        normalizer = self.normalization.build(self.in_names, self.out_names)
        loss = self.loss.build(area_weights.fine, "none")
        # We always use standard score normalization, so sigma_data is
        # always 1.0. See below for standard score normalization:
        # https://en.wikipedia.org/wiki/Standard_score
        sigma_data = 1.0

        module = self.module.build(
            n_in_channels=len(self.in_names),
            n_out_channels=len(self.out_names),
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            fine_topography=fine_topography if self.use_fine_topography else None,
            sigma_data=sigma_data,
        )
        return DiffusionModel(
            config=self,
            module=module,
            normalizer=normalizer,
            loss=loss,
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            sigma_data=sigma_data,
        )

    def get_state(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DiffusionModelConfig":
        return dacite.from_dict(data_class=cls, data=state)

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            names=list(set(self.in_names).union(self.out_names)),
            n_timesteps=1,
            use_fine_topography=self.use_fine_topography,
        )


def _repeat_batch_by_samples(tensor: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Repeat the batch dimension of a tensor n_samples times.  Used to parallelize
    sample generation in the diffusion model, but done so such that the tensor still
    only leads with a single batch dimension.  Added as a separate function for
    round-trip testing.
    """
    return tensor.repeat_interleave(dim=0, repeats=n_samples)


def _separate_interleaved_samples(
    tensor: torch.Tensor, n_batch: int, n_samples: int
) -> torch.Tensor:
    """
    Reshape an interleaved tensor to have a leading [batch, sample, ...] dimension.
    """
    return tensor.reshape(n_batch, n_samples, *tensor.shape[1:])


class DiffusionModel:
    def __init__(
        self,
        config: DiffusionModelConfig,
        module: torch.nn.Module,
        normalizer: FineResCoarseResPair[StandardNormalizer],
        loss: torch.nn.Module,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
    ) -> None:
        """
        Args:
            config: The configuration object for the diffusion model.
            module: The neural network module used for downscaling. Note: this
                should *not* be DistributedDataParallel since it is wrapped by
                it in this init method.
            normalizer: The normalizer object used for data normalization.
            loss: The loss function used for training the model.
            coarse_shape: The height (lat) and width (lon) of the
                coarse-resolution input data.
            downscale_factor: The factor by which the data is downscaled from
                coarse to fine.
            sigma_data: The standard deviation of the data, used for diffusion
                model preconditioning
        """
        self.coarse_shape = coarse_shape
        self.downscale_factor = downscale_factor
        self.sigma_data = sigma_data
        dist = Distributed.get_instance()
        self.module = dist.wrap_module(module.to(get_device()))
        self.normalizer = normalizer
        self.loss = loss
        self.in_packer = Packer(config.in_names)
        self.out_packer = Packer(config.out_names)
        self.config = config

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([self.module])

    def train_on_batch(
        self,
        batch: FineResCoarseResPair[TensorMapping],
        optimizer: Union[Optimization, NullOptimization],
    ) -> ModelOutputs:
        """Performs a denoising training step on a batch of data."""

        channel_axis = -3
        coarse, fine = _tensor_mapping_to_device(
            batch.coarse, get_device()
        ), _tensor_mapping_to_device(batch.fine, get_device())
        coarse_norm = self.in_packer.pack(
            self.normalizer.coarse.normalize(dict(coarse)), axis=channel_axis
        )
        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=channel_axis
        )

        rnd_normal = torch.randn(
            [targets_norm.shape[0], 1, 1, 1], device=targets_norm.device
        )
        # This is taken from EDM's original implementation in EDMLoss:
        # https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L72-L80  # noqa: E501
        sigma = (rnd_normal * self.config.p_std + self.config.p_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(targets_norm) * sigma
        latents = targets_norm + noise

        if self.config.predict_residual:
            base_prediction = interpolate(
                self.in_packer.pack(
                    self.normalizer.coarse.normalize(
                        {k: coarse[k] for k in fine.keys()}
                    ),
                    axis=channel_axis,
                ),
                self.downscale_factor,
            )
            targets_norm = targets_norm - base_prediction
            latents = latents - base_prediction

        denoised_norm = self.module(latents, coarse_norm, sigma)
        loss = torch.mean(weight * self.loss(targets_norm, denoised_norm))
        optimizer.step_weights(loss)

        if self.config.predict_residual:
            denoised_norm = denoised_norm + base_prediction

        target = filter_tensor_mapping(batch.fine, set(self.out_packer.names))
        denoised_norm = denoised_norm.unsqueeze(1)
        denoised = self.normalizer.fine.denormalize(
            self.out_packer.unpack(denoised_norm, axis=channel_axis)
        )
        return ModelOutputs(
            prediction=denoised, target=target, loss=loss, latent_steps=[]
        )

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: FineResCoarseResPair[TensorMapping],
        n_samples: int = 1,
    ) -> ModelOutputs:
        channel_axis = -3
        coarse, fine = _tensor_mapping_to_device(
            batch.coarse, get_device()
        ), _tensor_mapping_to_device(batch.fine, get_device())

        coarse_norm = self.in_packer.pack(
            self.normalizer.coarse.normalize(dict(coarse)), axis=channel_axis
        )
        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=channel_axis
        )

        n_batch = targets_norm.shape[0]
        coarse_norm = _repeat_batch_by_samples(coarse_norm, n_samples)
        targets_norm = _repeat_batch_by_samples(targets_norm, n_samples)
        latents = torch.randn_like(targets_norm)

        logging.info("Running EDM sampler...")
        samples_norm, latent_steps = edm_sampler(
            self.module,
            latents,
            coarse_norm,
            S_churn=self.config.churn,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            num_steps=self.config.num_diffusion_generation_steps,
        )
        logging.info("Done running EDM sampler.")

        if self.config.predict_residual:
            base_prediction = interpolate(
                self.in_packer.pack(
                    self.normalizer.coarse.normalize(
                        {k: coarse[k] for k in fine.keys()}
                    ),
                    axis=channel_axis,
                ),
                self.downscale_factor,
            )
            samples_norm += _repeat_batch_by_samples(base_prediction, n_samples)

        loss = self.loss(targets_norm, samples_norm)

        samples_norm_reshaped = _separate_interleaved_samples(
            samples_norm, n_batch, n_samples
        )
        samples = self.normalizer.fine.denormalize(
            self.out_packer.unpack(samples_norm_reshaped, axis=channel_axis)
        )

        return ModelOutputs(
            prediction=samples, target=batch.fine, loss=loss, latent_steps=latent_steps
        )

    def get_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.get_state(),
            "module": self.module.state_dict(),
            "coarse_shape": self.coarse_shape,
            "downscale_factor": self.downscale_factor,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        area_weights: FineResCoarseResPair[torch.Tensor],
        fine_topography: torch.Tensor,
    ) -> "DiffusionModel":
        config = DiffusionModelConfig.from_state(state["config"])
        model = config.build(
            state["coarse_shape"],
            state["downscale_factor"],
            area_weights,
            fine_topography,
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model
