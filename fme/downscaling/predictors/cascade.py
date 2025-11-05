import dataclasses
import math
from collections.abc import Sequence

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.tensors import unfold_ensemble_dim
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import (
    BatchData,
    ClosedInterval,
    PairedBatchData,
    Topography,
    adjust_fine_coord_range,
    scale_tuple,
)
from fme.downscaling.metrics_and_maths import filter_tensor_mapping
from fme.downscaling.models import CheckpointModelConfig, DiffusionModel, ModelOutputs
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.typing_ import FineResCoarseResPair


def _closed_interval_from_coord(coord: torch.Tensor) -> ClosedInterval:
    return ClosedInterval(start=coord.min().item(), stop=coord.max().item())


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

    def get_topographies(self) -> list[Topography | None]:
        topographies = []
        for ckpt in self.cascade_model_checkpoints:
            topographies.append(ckpt.get_topography())
        return topographies

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

        return CascadePredictor(
            models=self.models, topographies=self.get_topographies()
        )

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
        self, models: list[DiffusionModel], topographies: list[Topography | None]
    ):
        self.models = models
        self._topographies = topographies
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
        topographies=list[Topography | None],
    ):
        current_coarse = coarse
        for i, (model, fine_topography) in enumerate(zip(self.models, topographies)):
            sample_data = next(iter(current_coarse.values()))
            batch_size = sample_data.shape[0]
            # n_samples are generated for the first step, and subsequent models
            # generate 1 sample
            n_samples_cascade_step = n_samples if i == 0 else 1
            _fine_topography = fine_topography.data

            generated, generated_norm, latent_steps = model.generate(
                current_coarse, _fine_topography, n_samples_cascade_step
            )
            generated = {
                k: v.reshape(batch_size * n_samples_cascade_step, *v.shape[-2:])
                for k, v in generated.items()
            }
            current_coarse = generated
        generated = _restore_batch_and_sample_dims(generated, n_samples)
        return generated, generated_norm, latent_steps

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        topography: Topography | None,
        n_samples: int = 1,
    ) -> TensorDict:
        topographies = self._get_subset_topographies(
            coarse_coords=batch.latlon_coordinates[0]
        )
        generated, _, _ = self.generate(batch.data, n_samples, topographies)
        return generated

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        topography: Topography | None,
        n_samples: int = 1,
    ) -> ModelOutputs:
        topographies = self._get_subset_topographies(
            coarse_coords=batch.coarse.latlon_coordinates[0]
        )
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

    def _get_subset_topographies(
        self,
        coarse_coords: LatLonCoordinates,
    ) -> Sequence[Topography | None]:
        # Intermediate topographies are loaded as full range and need to be subset
        # to the matching lat/lon range for each batch.
        # TODO: Will eventually move subsetting into checkpoint model.
        subset_topographies = []
        _coarse_coords = coarse_coords
        lat_range = _closed_interval_from_coord(_coarse_coords.lat)
        lon_range = _closed_interval_from_coord(_coarse_coords.lon)

        for i, full_intermediate_topography in enumerate(self._topographies):
            if full_intermediate_topography is not None:
                _adjusted_lat_range = adjust_fine_coord_range(
                    lat_range,
                    _coarse_coords.lat,
                    full_intermediate_topography.coords.lat,
                    downscale_factor=self.models[i].downscale_factor,
                )
                _adjusted_lon_range = adjust_fine_coord_range(
                    lon_range,
                    _coarse_coords.lon,
                    full_intermediate_topography.coords.lon,
                    downscale_factor=self.models[i].downscale_factor,
                )
                subset_interm_topo = full_intermediate_topography.subset_latlon(
                    lat_interval=_adjusted_lat_range, lon_interval=_adjusted_lon_range
                )
                _coarse_coords = subset_interm_topo.coords
                lat_range = _closed_interval_from_coord(_coarse_coords.lat)
                lon_range = _closed_interval_from_coord(_coarse_coords.lon)
            else:
                subset_interm_topo = None
            subset_topographies.append(subset_interm_topo)
        return subset_topographies
