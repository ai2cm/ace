import dataclasses
from collections.abc import Mapping

import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Image

from ..plotting import plot_paneled_data
from .build_context import OneStepBuildContext, OneStepMetricBuildResult


class SkillMapAggregator:
    """
    Records per-gridcell skill maps computed over the sample dimension at the
    first forecast step.

    Two statistics are computed, for each gridcell, with sums over the sample
    dimension:

    - Coefficient of determination (R²)::

          R² = 1 - MSE / Var_target

      where ``MSE = mean_samples((prediction - target)**2)`` and
      ``Var_target = E_samples[target**2] - E_samples[target]**2``. R² is
      dimensionless and invariant to the affine per-channel normalization.

    - Root-mean-squared error (RMSE)::

          RMSE = sqrt(mean_samples((prediction - target)**2))

      the non-normalized, physical-units sibling of R².

    Both are computed on the denormalized data at the first forecast step,
    matching the convention of the other one-step map aggregators (see
    ``MapAggregator``).
    """

    _captions = {
        "r2": (
            "{name} coefficient of determination (R²) over samples at the first "
            "forecast step; 1 is perfect, 0 matches the sample mean [unitless]"
        ),
        "rmse": (
            "{name} root-mean-squared error over samples at the first forecast "
            "step [{units}]"
        ),
    }
    # R² has natural bounds (1 = perfect, 0 = no better than the sample mean),
    # so its map uses a fixed diverging scale rather than data-derived limits.
    # This keeps the colorbar comparable across epochs and stops extreme
    # negative outliers (from gridcells with tiny target variance) from washing
    # out the informative range. Out-of-range values saturate at the ends.
    _R2_VMIN = -1.0
    _R2_VMAX = 1.0

    def __init__(
        self,
        dims: list[str],
        metadata: Mapping[str, VariableMetadata] | None = None,
        include_r2: bool = True,
        include_rmse: bool = True,
    ):
        """
        Args:
            dims: Names of the horizontal (spatial) dimensions.
            metadata: Mapping of variable names to their metadata, used in
                generating logged image captions.
            include_r2: Whether to compute and log the R² map.
            include_rmse: Whether to compute and log the RMSE map.
        """
        self._dims = dims
        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata
        self._include_r2 = include_r2
        self._include_rmse = include_rmse
        self._n_batches = 0
        self._sq_err_sum: TensorDict = {}
        self._target_sum: TensorDict = {}
        self._target_sq_sum: TensorDict = {}

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
    ):
        time_dim = 1
        # only the first forecast step is scored, matching the other one-step
        # map aggregators; see https://github.com/ai2cm/full-model/issues/1005
        target_time = 1
        for name in gen_data:
            target = target_data[name].select(dim=time_dim, index=target_time).double()
            gen = gen_data[name].select(dim=time_dim, index=target_time).double()
            # accumulate in double precision: the variance is computed as
            # E[X^2] - E[X]^2, which suffers catastrophic cancellation in
            # single precision for fields with a large mean relative to their
            # variance (e.g. temperature).
            sq_err = ((gen - target) ** 2).mean(dim=0)
            target_mean = target.mean(dim=0)
            target_sq_mean = (target**2).mean(dim=0)
            if name in self._sq_err_sum:
                self._sq_err_sum[name] += sq_err
                self._target_sum[name] += target_mean
                self._target_sq_sum[name] += target_sq_mean
            else:
                self._sq_err_sum[name] = sq_err
                self._target_sum[name] = target_mean
                self._target_sq_sum[name] = target_sq_mean
        self._n_batches += 1

    def _get_maps(self) -> Mapping[str, TensorDict]:
        """Return the requested skill maps keyed by statistic then variable."""
        dist = Distributed.get_instance()
        r2: TensorDict = {}
        rmse: TensorDict = {}
        # reduce_mean's all_reduce mutates its argument in place, so reduce a clone
        # to keep this getter idempotent -- it is called once for the netCDF dataset
        # and once for the wandb logs, and re-reducing the stored accumulators would
        # inflate every moment by total_ranks (driving the variance negative -> NaN).
        for name in sorted(self._sq_err_sum.keys()):
            mse = dist.reduce_mean(self._sq_err_sum[name].clone()) / self._n_batches
            if self._include_rmse:
                rmse[name] = torch.sqrt(mse)
            if self._include_r2:
                target_mean = (
                    dist.reduce_mean(self._target_sum[name].clone()) / self._n_batches
                )
                target_sq_mean = (
                    dist.reduce_mean(self._target_sq_sum[name].clone())
                    / self._n_batches
                )
                target_var = target_sq_mean - target_mean**2
                r2[name] = torch.where(
                    target_var > 0,
                    1 - mse / target_var,
                    torch.full_like(target_var, float("nan")),
                )
        maps: dict[str, TensorDict] = {}
        if self._include_r2:
            maps["r2"] = r2
        if self._include_rmse:
            maps["rmse"] = rmse
        return maps

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, Image]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        maps = self._get_maps()
        image_logs = {}
        for statistic, data in maps.items():
            for name in data:
                panel = [[data[name].cpu().numpy()]]
                caption = self._get_caption(statistic, name)
                if statistic == "r2":
                    # fixed diverging [-1, 1] scale (see _R2_VMIN/_R2_VMAX)
                    image = plot_paneled_data(
                        panel,
                        diverging=True,
                        caption=caption,
                        vmin=self._R2_VMIN,
                        vmax=self._R2_VMAX,
                    )
                else:
                    image = plot_paneled_data(panel, diverging=False, caption=caption)
                image_logs[f"{statistic}/{name}"] = image
        return {f"{label}/{key}": image_logs[key] for key in image_logs}

    def _get_caption(self, statistic: str, name: str) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name].display_long_name(name)
            units = self._metadata[name].display_units()
        else:
            caption_name, units = name, "unknown_units"
        return self._captions[statistic].format(name=caption_name, units=units)

    def get_dataset(self) -> xr.Dataset:
        maps = self._get_maps()
        long_names = {
            "r2": "coefficient of determination",
            "rmse": "root-mean-squared error",
        }
        ds = xr.Dataset()
        for statistic, data in maps.items():
            for name in data:
                if name in self._metadata:
                    long_name = self._metadata[name].display_long_name(name)
                else:
                    long_name = name
                attrs = {"long_name": f"{long_names[statistic]} of {long_name}"}
                if statistic == "rmse" and name in self._metadata:
                    attrs["units"] = self._metadata[name].display_units()
                ds[f"{statistic}-{name}"] = xr.DataArray(
                    data=data[name].cpu().numpy(), dims=self._dims, attrs=attrs
                )
        return ds


@dataclasses.dataclass
class OneStepSkillMapMetricConfig:
    """Per-gridcell skill maps (R², RMSE) over the sample dimension."""

    name: str = "skill_map"
    enabled: bool = True
    strict: bool = False
    include_r2: bool = True
    include_rmse: bool = True

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: OneStepBuildContext) -> OneStepMetricBuildResult:
        agg = SkillMapAggregator(
            ctx.horizontal_coordinates.dims,
            ctx.variable_metadata,
            include_r2=self.include_r2,
            include_rmse=self.include_rmse,
        )
        return OneStepMetricBuildResult(deterministic=agg)
