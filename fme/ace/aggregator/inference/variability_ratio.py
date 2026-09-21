import dataclasses
import logging
from collections.abc import Sequence

import numpy as np
import torch
import xarray as xr

from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations

from .build_context import MetricBuildContext, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator


@dataclasses.dataclass
class VariabilityRatioMetricConfig:
    """Ratio of predicted to target temporal variance, per variable, over the rollout.

    Time-mean metrics score where a rollout settles; this scores how much it
    moves once there. For each variable the per-cell variance of the
    deseasonalized monthly-scale anomaly is computed over every sample and
    timestep of the rollout, for the prediction and for the target, each is
    area-averaged, and the ratio prediction / target is logged. A healthy
    emulator sits near 1; a value well below 1 is a field whose variability
    the emulator damps, which a time-mean RMSE cannot see. The case that
    motivated it: an ocean emulator whose 130-200 m tropical-Pacific
    temperature variance was 0.55 of the reference's while its channel-mean
    time-mean error was indistinguishable from a healthy model's, and whose
    seasonal-forecast skill was correspondingly poor.

    Deseasonalizing removes the variance of the calendar-month climatology
    (mean by month of year over the rollout) so that the ratio measures
    interannual and higher-frequency variability rather than the seasonal
    cycle, which every emulator reproduces. It needs the rollout to span at
    least a full year to mean anything; on shorter rollouts set
    ``deseasonalize`` to False.

    Memory: one float32 field per variable per source for the shift
    reference, the sum and the sum of squares, plus twelve monthly sums when
    deseasonalizing: about 60 bytes per grid cell per variable per source.
    Restrict ``variables`` on large grids if that matters.

    Parameters:
        variables: variables to compute the ratio for; None means every
            variable with a target.
        deseasonalize: subtract the calendar-month climatology's variance.
        name: log prefix and wandb key prefix.
        log_channel_mean: also log the mean ratio over variables as
            ``<name>/channel_mean``. Off where two components' logs are merged
            under one namespace (the coupled evaluator), since the key would
            collide.
        enabled: master toggle for the metric.
        strict: raise if the metric can't be built.
    """

    variables: list[str] | None = None
    deseasonalize: bool = True
    name: str = "variability_ratio"
    log_channel_mean: bool = True
    enabled: bool = True
    strict: bool = False

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        agg: SubAggregator = VariabilityRatioAggregator(
            gridded_operations=ctx.ops,
            deseasonalize=self.deseasonalize,
            horizontal_dims=ctx.horizontal_coordinates.dims,
            log_channel_mean=self.log_channel_mean,
        )
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))


class _VarianceAccumulator:
    """Per-cell running sums for a deseasonalized temporal variance.

    Values are accumulated as deviations from the first field seen, so the
    sums stay small and float32 is accurate; variance is invariant to the
    shift. With ``deseasonalize`` the calendar-month means are accumulated
    too, and their variance about the overall mean is removed.
    """

    def __init__(self, deseasonalize: bool):
        self._deseasonalize = deseasonalize
        self._ref: torch.Tensor | None = None
        self._sum: torch.Tensor | None = None
        self._sumsq: torch.Tensor | None = None
        self._n = 0
        self._month_sum: torch.Tensor | None = None  # [12, ...]
        self._month_n: torch.Tensor | None = None  # [12]

    def add(self, tensor: torch.Tensor, months: torch.Tensor | None):
        """tensor: [sample, time, *horizontal]; months: [sample, time] in 0..11."""
        flat = tensor.reshape(-1, *tensor.shape[2:]).to(torch.float32)
        if self._ref is None:
            self._ref = torch.nan_to_num(flat[0].clone())
            self._sum = torch.zeros_like(self._ref)
            self._sumsq = torch.zeros_like(self._ref)
            if self._deseasonalize:
                self._month_sum = torch.zeros(
                    (12, *self._ref.shape), dtype=torch.float32, device=flat.device
                )
                self._month_n = torch.zeros(12, dtype=torch.float64, device=flat.device)
        dev = torch.nan_to_num(flat - self._ref)
        assert self._sum is not None and self._sumsq is not None
        self._sum += dev.sum(0)
        self._sumsq += (dev * dev).sum(0)
        self._n += dev.shape[0]
        if self._deseasonalize:
            assert (
                months is not None
                and self._month_sum is not None
                and self._month_n is not None
            )
            m = months.reshape(-1).to(flat.device)
            for k in range(12):
                sel = m == k
                if bool(sel.any()):
                    self._month_sum[k] += dev[sel].sum(0)
                    self._month_n[k] += int(sel.sum())

    def variance(self, dist: Distributed) -> torch.Tensor | None:
        if self._sum is None or self._sumsq is None:
            return None
        n = dist.reduce_sum(torch.tensor(float(self._n), device=self._sum.device))
        s = dist.reduce_sum(self._sum.clone())
        ss = dist.reduce_sum(self._sumsq.clone())
        if n is None or s is None or ss is None:
            return None
        mean = s / n
        var = ss / n - mean * mean
        if self._deseasonalize:
            assert self._month_sum is not None and self._month_n is not None
            ms = dist.reduce_sum(self._month_sum.clone())
            mn = dist.reduce_sum(self._month_n.clone())
            if ms is None or mn is None:
                return None
            present = mn > 0
            month_mean = torch.where(
                present[:, None, None]
                if ms.ndim == 3
                else present.reshape(-1, *([1] * (ms.ndim - 1))),
                ms / mn.clamp(min=1).reshape(-1, *([1] * (ms.ndim - 1))).to(ms.dtype),
                torch.zeros_like(ms),
            )
            weights = (mn / n).to(ms.dtype).reshape(-1, *([1] * (ms.ndim - 1)))
            seasonal = (weights * (month_mean - mean) ** 2).sum(0)
            var = var - seasonal
        return var.clamp(min=0.0)


class VariabilityRatioAggregator:
    """Area-averaged deseasonalized temporal variance of the prediction divided
    by that of the target, per variable, over the whole rollout.

    The initial-condition timestep of the first batch is excluded, as in the
    time-mean aggregator, since it is prescribed rather than predicted. Logs
    ``<label>/<variable>`` for every variable seen with a target and
    ``<label>/channel_mean`` for their mean. Cells that are NaN in a source
    contribute zero deviation; the per-variable area weighting (which carries
    the mask for masked grids) handles their exclusion.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        deseasonalize: bool = True,
        horizontal_dims: Sequence[str] | None = None,
        log_channel_mean: bool = True,
    ):
        self._ops = gridded_operations
        self._deseasonalize = deseasonalize
        self._log_channel_mean = log_channel_mean
        self._horizontal_dims = (
            list(horizontal_dims) if horizontal_dims else ["lat", "lon"]
        )
        self._dist = Distributed.get_instance()
        self._gen: dict[str, _VarianceAccumulator] = {}
        self._target: dict[str, _VarianceAccumulator] = {}

    @staticmethod
    def _months(time: xr.DataArray, time_slice: slice) -> torch.Tensor:
        """Calendar month index (0-11) of each (sample, time) entry.

        The time array is (sample, time) and may hold cftime objects, which the
        datetime accessor only handles one-dimensionally, so it is read one
        sample at a time (as the annual aggregator does).
        """
        rows = []
        for i in range(time.sizes[time.dims[0]]):
            row = time.isel({time.dims[0]: i})
            try:
                rows.append(np.asarray(row.dt.month.values))
            except (AttributeError, TypeError):
                rows.append(np.array([t.month for t in row.values]))
        months = np.stack(rows)[:, time_slice] - 1
        return torch.as_tensor(months, dtype=torch.int64)

    def _months_or_disable(
        self, time: xr.DataArray, time_slice: slice
    ) -> torch.Tensor | None:
        """Months for this batch, or None after switching deseasonalizing off when
        the time axis carries no calendar (e.g. plain step indices).
        """
        if not self._deseasonalize:
            return None
        try:
            return self._months(time, time_slice)
        except (AttributeError, TypeError, ValueError):
            if self._gen:
                raise ValueError(
                    "variability_ratio: the time axis lost its calendar part-way "
                    "through the rollout; cannot deseasonalize consistently."
                )
            logging.warning(
                "variability_ratio: time axis has no calendar; reporting the raw "
                "(not deseasonalized) variance ratio."
            )
            self._deseasonalize = False
            return None

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData):
        if not data.has_target:
            return
        time_slice = slice(1, None) if data.i_time_start == 0 else slice(None)
        months = self._months_or_disable(data.time, time_slice)
        for name, gen in data.prediction.items():
            if name not in data.target:
                continue
            self._gen.setdefault(name, _VarianceAccumulator(self._deseasonalize)).add(
                gen[:, time_slice], months
            )
            self._target.setdefault(
                name, _VarianceAccumulator(self._deseasonalize)
            ).add(data.target[name][:, time_slice], months)

    def _ratios(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in sorted(self._gen):  # sorted for rank-consistent collectives
            vg = self._gen[name].variance(self._dist)
            vt = self._target[name].variance(self._dist)
            if vg is None or vt is None:
                continue
            g = self._ops.area_weighted_mean(vg, name=name)
            t = self._ops.area_weighted_mean(vt, name=name)
            t_val = float(t.cpu())
            out[name] = float(g.cpu()) / t_val if t_val > 0 else float("nan")
        return out

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, float]:
        ratios = self._ratios()
        prefix = f"{label}/" if label else ""
        logs = {f"{prefix}{name}": value for name, value in ratios.items()}
        finite = [v for v in ratios.values() if np.isfinite(v)]
        if finite and self._log_channel_mean:
            logs[f"{prefix}channel_mean"] = float(np.mean(finite))
        return logs

    def get_dataset(self) -> xr.Dataset:
        ratios = self._ratios()
        return xr.Dataset(
            {
                name: xr.DataArray(
                    value,
                    attrs={
                        "long_name": f"{name} variability ratio (prediction / target)"
                    },
                )
                for name, value in ratios.items()
            }
        )
