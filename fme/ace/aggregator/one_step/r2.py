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


class R2Aggregator:
    """
    Records the per-gridcell coefficient of determination (R²) of the
    prediction, computed over the sample dimension at the first forecast step.

    For each gridcell, with sums of squares taken over the sample dimension,

        R² = 1 - SS_res / SS_tot = 1 - MSE / Var_target

    where ``MSE = mean_samples((prediction - target)**2)`` and
    ``Var_target = E_samples[target**2] - E_samples[target]**2``. R² is
    invariant to the affine per-channel normalization, so it is identical in
    normalized and denormalized space; the denormalized data is used here.

    Only the first forecast step is scored, matching the convention of the
    other one-step map aggregators (see ``MapAggregator``).
    """

    _caption = (
        "{name} coefficient of determination (R²) over samples at the first "
        "forecast step; 1 is perfect, 0 matches the sample mean [unitless]"
    )

    def __init__(
        self, dims: list[str], metadata: Mapping[str, VariableMetadata] | None = None
    ):
        """
        Args:
            dims: Names of the horizontal (spatial) dimensions.
            metadata: Mapping of variable names to their metadata, used in
                generating logged image captions.
        """
        self._dims = dims
        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata
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

    def _get_r2(self) -> Mapping[str, torch.Tensor]:
        dist = Distributed.get_instance()
        r2 = {}
        for name in sorted(self._sq_err_sum.keys()):
            mse = dist.reduce_mean(self._sq_err_sum[name]) / self._n_batches
            target_mean = dist.reduce_mean(self._target_sum[name]) / self._n_batches
            target_sq_mean = (
                dist.reduce_mean(self._target_sq_sum[name]) / self._n_batches
            )
            target_var = target_sq_mean - target_mean**2
            r2[name] = torch.where(
                target_var > 0,
                1 - mse / target_var,
                torch.full_like(target_var, float("nan")),
            )
        return r2

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, Image]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        r2 = self._get_r2()
        image_logs = {}
        for name in r2:
            image_logs[name] = plot_paneled_data(
                [[r2[name].cpu().numpy()]],
                diverging=False,
                caption=self._get_caption(name),
            )
        return {f"{label}/{key}": image_logs[key] for key in image_logs}

    def _get_caption(self, name: str) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name].display_long_name(name)
        else:
            caption_name = name
        return self._caption.format(name=caption_name)

    def get_dataset(self) -> xr.Dataset:
        r2 = self._get_r2()
        ds = xr.Dataset()
        for name in r2:
            if name in self._metadata:
                long_name = self._metadata[name].display_long_name(name)
            else:
                long_name = name
            attrs = {"long_name": f"coefficient of determination of {long_name}"}
            ds[f"r2-{name}"] = xr.DataArray(
                data=r2[name].cpu().numpy(), dims=self._dims, attrs=attrs
            )
        return ds


@dataclasses.dataclass
class OneStepR2MetricConfig:
    """Per-gridcell coefficient of determination (R²) map metric."""

    name: str = "r2"
    enabled: bool = True
    strict: bool = False

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: OneStepBuildContext) -> OneStepMetricBuildResult:
        agg = R2Aggregator(
            ctx.horizontal_coordinates.dims,
            ctx.variable_metadata,
        )
        return OneStepMetricBuildResult(deterministic=agg)
