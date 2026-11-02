import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations

from .build_context import MetricBuildContext, MetricNotSupportedError, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator
from .trend import SECONDS_PER_YEAR, _years_since_reference

# Minimum rollout duration for drift, matching the ~2 year minimum used by the
# annual metric: a slow drift cannot be separated from interannual variability
# over a shorter span. The metric is not built for configured rollouts at or
# below this duration, and slopes are reported as NaN for samples whose
# recorded data covers less than it.
MIN_DRIFT_DURATION = datetime.timedelta(days=730)

# Index of each per-sample sufficient statistic of the time regressor.
_N, _SUM_T, _SUM_TT, _T_MIN, _T_MAX = range(5)


class GlobalMeanDriftAggregator:
    """Per-sample linear drift of the global-mean bias over a rollout.

    For each variable and each sample (initial condition or ensemble member),
    the drift is the ordinary least-squares slope, against time in years, of
    the area-weighted global mean of ``prediction - target``. As in
    :class:`TrendEvaluatorAggregator`, it is computed from running sums of
    ``t``, ``t**2``, ``y``, ``t*y`` and ``n`` accumulated over every
    ``record_batch`` call, here kept separately for each sample, so memory is
    ``O(n_samples * n_variables)`` regardless of the rollout length. Time is
    measured from each sample's first recorded timestep.

    Logged scalars are the sample-mean slope per variable, its t-statistic
    ``mean / (std / sqrt(n_samples))`` across samples (NaN with fewer than two
    samples), and the maximum absolute t-statistic over variables. Slopes are
    NaN for samples whose rollout covers less than ``MIN_DRIFT_DURATION``.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        """
        Args:
            gridded_operations: Computes the area-weighted global means.
            timestep: Timestep of the data, used to compute the duration
                covered by the recorded timesteps.
            variable_metadata: Mapping of variable names to metadata used in
                dataset attributes.
        """
        self._ops = gridded_operations
        self._timestep_years = timestep.total_seconds() / SECONDS_PER_YEAR
        self._variable_metadata = variable_metadata or {}
        # per-sample start time [years since reference], set on first record
        self._t0: torch.Tensor | None = None
        # [n_sample, 5] time-regressor statistics, indexed by _N, _SUM_T, ...
        self._time_stats: torch.Tensor | None = None
        # per-variable [n_sample] running sums of y and t*y
        self._sum_y: dict[str, torch.Tensor] = {}
        self._sum_ty: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData) -> None:
        # Mirror TrendEvaluatorAggregator: when the initial condition is the
        # first timestep of the first batch, skip it so it is not counted.
        time_slice = slice(1, None) if data.i_time_start == 0 else slice(None)
        device = get_device()
        years = torch.tensor(
            _years_since_reference(data.time)[:, time_slice],
            dtype=torch.float64,
            device=device,
        )
        if years.shape[1] == 0:
            return
        if self._t0 is None:
            self._t0 = years[:, 0].clone()
            self._time_stats = torch.zeros(
                years.shape[0], 5, dtype=torch.float64, device=device
            )
            self._time_stats[:, _T_MIN] = torch.inf
            self._time_stats[:, _T_MAX] = -torch.inf
        assert self._time_stats is not None
        t = years - self._t0[:, None]
        stats = self._time_stats.clone()
        stats[:, _N] += t.shape[1]
        stats[:, _SUM_T] += t.sum(dim=1)
        stats[:, _SUM_TT] += (t * t).sum(dim=1)
        stats[:, _T_MIN] = torch.minimum(stats[:, _T_MIN], t.min(dim=1).values)
        stats[:, _T_MAX] = torch.maximum(stats[:, _T_MAX], t.max(dim=1).values)
        self._time_stats = stats
        for name, prediction in data.prediction.items():
            bias = self._ops.area_weighted_mean_bias(
                truth=data.target[name][:, time_slice],
                predicted=prediction[:, time_slice],
                name=name,
            ).to(torch.float64)  # [n_sample, n_time]
            contrib_y = bias.sum(dim=1)
            contrib_ty = (t * bias).sum(dim=1)
            if name in self._sum_y:
                self._sum_y[name] = self._sum_y[name] + contrib_y
                self._sum_ty[name] = self._sum_ty[name] + contrib_ty
            else:
                self._sum_y[name] = contrib_y
                self._sum_ty[name] = contrib_ty

    def _get_slopes(self) -> tuple[list[str], torch.Tensor] | None:
        """Gather the per-sample statistics across data-parallel ranks and
        finalize the slopes.

        Returns:
            The sorted variable names and a ``[n_sample, n_variable]`` tensor
            of slopes on the root rank, or ``None`` on other ranks.
        """
        if self._time_stats is None:
            raise ValueError("No data has been recorded yet.")
        names = sorted(self._sum_y)
        # [n_sample, 5 + 2 * n_variable], one collective for all statistics
        local = torch.cat(
            [
                self._time_stats,
                torch.stack([self._sum_y[name] for name in names], dim=1),
                torch.stack([self._sum_ty[name] for name in names], dim=1),
            ],
            dim=1,
        )
        # Samples are split across data-parallel ranks; spatial co-ranks hold
        # identical global means, so gather within the data-parallel group.
        dist = Distributed.get_instance()
        gathered = dist.data_parallel_gather(local)
        if not dist.is_root() or gathered is None:
            return None
        stats = torch.cat(gathered, dim=0)
        n_var = len(names)
        n = stats[:, _N, None]
        sum_t = stats[:, _SUM_T, None]
        sum_tt = stats[:, _SUM_TT, None]
        sum_y = stats[:, 5 : 5 + n_var]
        sum_ty = stats[:, 5 + n_var :]
        slopes = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t)
        duration = stats[:, _T_MAX] - stats[:, _T_MIN] + self._timestep_years
        min_duration = MIN_DRIFT_DURATION.total_seconds() / SECONDS_PER_YEAR
        slopes[duration < min_duration] = torch.nan
        return names, slopes

    def get_per_sample_slopes(self) -> dict[str, torch.Tensor]:
        """Per-sample drift slopes of the global-mean bias.

        This is a collective operation and must be called on every rank.

        Returns:
            Mapping from variable name to a ``[n_sample]`` float64 tensor of
            slopes in units of the variable per year, with samples from all
            data-parallel ranks. Empty on non-root ranks.
        """
        result = self._get_slopes()
        if result is None:
            return {}
        names, slopes = result
        return {name: slopes[:, i] for i, name in enumerate(names)}

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, Any]:
        per_sample = self.get_per_sample_slopes()
        if len(per_sample) == 0:
            return {}  # only the root process returns logs
        if len(label) > 0:
            label = label + "/"
        logs: dict[str, float] = {}
        abs_t_values = []
        for name, slopes in per_sample.items():
            slopes_np = slopes.cpu().numpy()
            logs[f"{label}slope/{name}"] = float(np.mean(slopes_np))
            t = _t_statistic(slopes_np)
            logs[f"{label}t/{name}"] = t
            if not np.isnan(t):
                abs_t_values.append(abs(t))
        logs[f"{label}max_abs_t"] = max(abs_t_values) if abs_t_values else np.nan
        return logs

    def _units(self, name: str) -> str:
        if name in self._variable_metadata:
            units = self._variable_metadata[name].display_units("unknown units")
        else:
            units = "unknown units"
        return f"{units} / year"

    def get_dataset(self) -> xr.Dataset:
        result = self._get_slopes()
        if result is None:
            return xr.Dataset()
        names, slopes = result
        slope = xr.DataArray(
            slopes.cpu().numpy(),
            dims=["sample", "variable"],
            coords={"variable": names},
            attrs={
                "long_name": (
                    "least-squares slope vs. time of the global-mean bias "
                    "(prediction - target)"
                ),
            },
        )
        units = xr.DataArray(
            [self._units(name) for name in names],
            dims=["variable"],
            coords={"variable": names},
        )
        return xr.Dataset({"slope": slope, "slope_units": units})


def _t_statistic(slopes: np.ndarray) -> float:
    """One-sample t-statistic of the slopes against zero drift, using the
    spread across samples; NaN for fewer than two samples.
    """
    n_sample = slopes.shape[0]
    if n_sample < 2:
        return np.nan
    std = np.std(slopes, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(slopes) / (std / np.sqrt(n_sample)))


@dataclasses.dataclass
class GlobalMeanDriftConfig:
    """Configuration for the per-sample global-mean drift inference metric.

    Logs the sample-mean least-squares slope (per year) of the global-mean
    bias of each variable, its t-statistic across samples, and the maximum
    absolute t-statistic over variables. Per-sample slopes are saved in the
    diagnostics dataset. Like the annual metric, it is skipped for rollouts
    of ~2 years or less, and slopes are NaN for samples whose recorded data
    covers less than ~2 years.

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
            ctx.ops,
            timestep=ctx.timestep,
            variable_metadata=ctx.variable_metadata,
        )
        filtered: SubAggregator = maybe_filter(agg, self.variables)
        return MetricBuildResult(aggregator=filtered, per_sample_slopes=agg)
