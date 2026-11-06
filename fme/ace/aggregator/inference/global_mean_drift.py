import dataclasses
import datetime
from collections.abc import Mapping

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations

from .build_context import MetricBuildContext, MetricNotSupportedError, maybe_filter
from .data import InferenceBatchData, MetricBuildResult
from .trend import _years_since_reference

# Same ~2 year minimum as the annual metric: shorter spans cannot separate a
# slow drift from interannual variability.
MIN_DRIFT_DURATION = datetime.timedelta(days=730)


class GlobalMeanDriftAggregator:
    """Per-sample least-squares slope vs. time (per year) of the area-weighted
    global-mean bias (prediction - target) of each variable.

    Slopes are computed from running sums, so memory does not grow with the
    rollout length. They are NaN if the recorded data covers less than
    ``MIN_DRIFT_DURATION``.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._ops = gridded_operations
        self._timestep = timestep
        self._variable_metadata = variable_metadata or {}
        self._n_time = 0
        # Per-sample time of the first recorded step, so the regressor stays
        # small and the float64 sums well conditioned.
        self._t0: torch.Tensor | None = None
        zero = torch.zeros((), dtype=torch.float64, device=get_device())
        # [n_sample] running sums; y and t*y are per variable.
        self._sum_t = zero
        self._sum_tt = zero
        self._sum_y: dict[str, torch.Tensor] = {}
        self._sum_ty: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData) -> None:
        # As in TrendEvaluatorAggregator, skip the initial condition.
        time_slice = slice(1, None) if data.i_time_start == 0 else slice(None)
        years = torch.tensor(
            _years_since_reference(data.time)[:, time_slice],
            dtype=torch.float64,
            device=get_device(),
        )
        if years.shape[1] == 0:
            return
        if self._t0 is None:
            self._t0 = years[:, 0].clone()
        t = years - self._t0[:, None]
        self._n_time += t.shape[1]
        self._sum_t = self._sum_t + t.sum(dim=1)
        self._sum_tt = self._sum_tt + (t * t).sum(dim=1)
        for name, prediction in data.prediction.items():
            bias = self._ops.area_weighted_mean_bias(
                truth=data.target[name][:, time_slice],
                predicted=prediction[:, time_slice],
                name=name,
            ).to(torch.float64)  # [n_sample, n_time]
            self._sum_y[name] = self._sum_y.get(name, 0.0) + bias.sum(dim=1)
            self._sum_ty[name] = self._sum_ty.get(name, 0.0) + (t * bias).sum(dim=1)

    def _get_slopes(self) -> tuple[list[str], torch.Tensor] | None:
        """Sorted variable names and ``[n_sample, n_variable]`` slopes over all
        data-parallel ranks on the root rank, ``None`` on other ranks.
        """
        if self._n_time == 0:
            raise ValueError("No data has been recorded yet.")
        names = sorted(self._sum_y)
        local = torch.stack(
            [
                self._sum_t,
                self._sum_tt,
                *(self._sum_y[name] for name in names),
                *(self._sum_ty[name] for name in names),
            ],
            dim=1,
        )
        # Samples are split across data-parallel ranks; spatial co-ranks hold
        # identical global means.
        dist = Distributed.get_instance()
        gathered = dist.data_parallel_gather(local)
        if not dist.is_root() or gathered is None:
            return None
        n_var = len(names)
        sum_t, sum_tt, sum_y, sum_ty = torch.cat(gathered).split(
            [1, 1, n_var, n_var], dim=1
        )
        n = self._n_time
        slopes = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t)
        if n * self._timestep < MIN_DRIFT_DURATION:
            slopes = torch.full_like(slopes, torch.nan)
        return names, slopes

    def get_per_sample_slopes(self) -> dict[str, torch.Tensor]:
        """Collective; must be called on every rank.

        Returns:
            Variable name to ``[n_sample]`` slopes (variable units per year)
            over all data-parallel ranks. Empty on non-root ranks.
        """
        result = self._get_slopes()
        if result is None:
            return {}
        names, slopes = result
        return {name: slopes[:, i] for i, name in enumerate(names)}

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, float]:
        per_sample = self.get_per_sample_slopes()
        if len(per_sample) == 0:
            return {}
        prefix = f"{label}/" if label else ""
        logs: dict[str, float] = {}
        abs_t = []
        for name, slopes in per_sample.items():
            slopes_np = slopes.cpu().numpy()
            logs[f"{prefix}slope/{name}"] = float(np.mean(slopes_np))
            t = _t_statistic(slopes_np)
            logs[f"{prefix}t/{name}"] = t
            if not np.isnan(t):
                abs_t.append(abs(t))
        logs[f"{prefix}max_abs_t"] = max(abs_t) if abs_t else np.nan
        return logs

    def get_dataset(self) -> xr.Dataset:
        result = self._get_slopes()
        if result is None:
            return xr.Dataset()
        names, slopes = result
        units = [
            self._variable_metadata[name].display_units("unknown units")
            if name in self._variable_metadata
            else "unknown units"
            for name in names
        ]
        return xr.Dataset(
            {
                "slope": (
                    ["sample", "variable"],
                    slopes.cpu().numpy(),
                    {"long_name": "slope vs. time of global-mean bias"},
                ),
                "slope_units": (["variable"], [f"{u} / year" for u in units]),
            },
            coords={"variable": names},
        )


def _t_statistic(slopes: np.ndarray) -> float:
    """t-statistic of the mean slope against zero; NaN for < 2 samples."""
    n_sample = slopes.shape[0]
    if n_sample < 2:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(slopes) / (np.std(slopes, ddof=1) / np.sqrt(n_sample)))


@dataclasses.dataclass
class GlobalMeanDriftConfig:
    """Per-sample drift of the global-mean bias.

    Logs, per variable, the sample-mean slope (units per year) of the
    global-mean bias and its t-statistic across samples, plus the maximum
    absolute t-statistic over variables. Per-sample slopes are saved in the
    diagnostics dataset. Skipped for rollouts of ~2 years or less.

    Attributes:
        variables: Variables to compute drift for. If ``None``, all available
            variables are included.
        name: Name used to label the metric's logs and diagnostics.
        enabled: Whether the metric is computed.
        strict: If True, raise rather than skip when the metric is not
            supported for the current configuration.
    """

    variables: list[str] | None = None
    name: str = "global_mean_drift"
    enabled: bool = True
    strict: bool = False

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        total_duration = ctx.n_timesteps * ctx.timestep
        if total_duration <= MIN_DRIFT_DURATION:
            raise MetricNotSupportedError(
                "global_mean_drift metric requires > ~2 years of data, "
                f"got {total_duration.days} days"
            )
        agg = GlobalMeanDriftAggregator(
            ctx.ops, timestep=ctx.timestep, variable_metadata=ctx.variable_metadata
        )
        return MetricBuildResult(
            aggregator=maybe_filter(agg, self.variables), per_sample_slopes=agg
        )
