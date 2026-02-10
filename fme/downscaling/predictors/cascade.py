import dataclasses
import math

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.tensors import unfold_ensemble_dim
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import BatchData, PairedBatchData, scale_tuple
from fme.downscaling.metrics_and_maths import filter_tensor_mapping
from fme.downscaling.models import CheckpointModelConfig, DiffusionModel, ModelOutputs
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.typing_ import FineResCoarseResPair


@dataclasses.dataclass
class CascadePredictorConfig:
    """
    Configuration for cascading multiple diffusion models.

    Args:
        cascade_models: List of model configurations to be used in the cascade,
            in order of input resolution, where coarsest inputs comes first.
    """

    cascade_model_checkpoints: list[CheckpointModelConfig]

    def __post_init__(self):
        self._models = None

    @property
    def models(self):
        if self._models is None:
            self._models = [cfg.build() for cfg in self.cascade_model_checkpoints]
        return self._models

    def build(self):
        for m in range(len(self.models) - 1):
            output_shape = scale_tuple(
                self.models[m].coarse_shape, self.models[m].downscale_factor
            )
            input_shape_next_step = self.models[m + 1].coarse_shape
            if output_shape != input_shape_next_step:
                raise ValueError(
                    "CascadePredictor requires each model in the cascade to operate on "
                    "the same full extent input region with appropriate dimensions. "
                    f"Output shape {output_shape} of model at step {m} does not match "
                    f"input shape {input_shape_next_step} of model at step {m+1}. "
                )

        return CascadePredictor(models=self.models)

    @property
    def data_requirements(self) -> DataRequirements:
        in_names = self.cascade_model_checkpoints[0].in_names
        out_names = self.cascade_model_checkpoints[-1].out_names
        return DataRequirements(
            fine_names=out_names,
            coarse_names=list(set(in_names).union(out_names)),
            n_timesteps=1,
            use_fine_topography=False,
        )


def _restore_batch_and_sample_dims(data: TensorMapping, n_samples: int):
    # Cascaded outputs after the first prediction have sample dim size of 1
    # Restore prediction tensor to shape (batch, generated_samples, ...)
    squeezed = {k: v.squeeze(1) for k, v in data.items()}
    return unfold_ensemble_dim(squeezed, n_samples)


class CascadePredictor:
    def __init__(
        self, models: list[DiffusionModel]
    ):
        self.models = models
        self.out_packer = self.models[-1].out_packer
        self.normalizer = FineResCoarseResPair(
            coarse=self.models[0].normalizer.coarse,
            fine=self.models[-1].normalizer.fine,
        )
        self._channel_axis = -3

    @property
    def coarse_shape(self):
        return self.models[0].coarse_shape

    @property
    def fine_shape(self):
        return self.models[-1].fine_shape

    @property
    def downscale_factor(self):
        return math.prod([model.downscale_factor for model in self.models])

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([model.modules for model in self.models])

    @torch.no_grad()
    def generate(
        self,
        coarse: TensorMapping,
        n_samples: int,
        topographies: list[torch.Tensor | None],
    ):
        """Low-level generate that accepts pre-resolved topography tensors."""
        current_coarse = coarse
        for i, (model, fine_topography) in enumerate(zip(self.models, topographies)):
            sample_data = next(iter(current_coarse.values()))
            batch_size = sample_data.shape[0]
            # n_samples are generated for the first step, and subsequent models
            # generate 1 sample
            n_samples_cascade_step = n_samples if i == 0 else 1

            generated, generated_norm, latent_steps = model.generate(
                current_coarse, fine_topography, n_samples_cascade_step
            )
            generated = {
                k: v.reshape(batch_size * n_samples_cascade_step, *v.shape[-2:])
                for k, v in generated.items()
            }
            current_coarse = generated
        generated = _restore_batch_and_sample_dims(generated, n_samples)
        return generated, generated_norm, latent_steps

    def _resolve_topographies(
        self,
        coarse_coords: LatLonCoordinates,
    ) -> list[torch.Tensor | None]:
        """Resolve topography tensors for each model from internal state."""
        topographies = []
        for model in self.models:
            topo = model._get_static_input_for_batch(coarse_coords)
            if topo is not None:
                topographies.append(topo.data)
                # Update coarse_coords to be the fine coords for next step
                coarse_coords = topo.coords
            else:
                topographies.append(None)
        return topographies

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        n_samples: int = 1,
    ) -> TensorDict:
        topographies = self._resolve_topographies(batch.latlon_coordinates[0])
        generated, _, _ = self.generate(batch.data, n_samples, topographies)
        return generated

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        topographies = self._resolve_topographies(batch.coarse.latlon_coordinates[0])
        generated, _, latent_steps = self.generate(
            batch.coarse.data, n_samples, topographies
        )
        targets = filter_tensor_mapping(batch.fine.data, set(self.out_packer.names))
        targets = {k: v.unsqueeze(1) for k, v in targets.items()}

        return ModelOutputs(
            prediction=generated,
            target=targets,
            loss=torch.tensor(float("inf"), device=get_device()),
            latent_steps=latent_steps,
        )
