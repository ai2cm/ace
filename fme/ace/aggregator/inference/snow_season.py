import dataclasses
import datetime
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import cftime
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.wandb import WandB

from .build_context import MetricBuildContext, MetricNotSupportedError
from .data import InferenceBatchData, MetricBuildResult, SubAggregator
from .utils import LatLonBoxConfig

SOURCES = ("prediction", "target")
QUANTITIES = ("peak", "midwinter_mean", "meltout_day", "summer_floor")
MELTOUT_FRACTION = 0.10
MIDWINTER_MONTHS = (3, 4, 5)
SUMMER_MONTHS = (10, 11, 12)
MINIMUM_DURATION = datetime.timedelta(days=730)
_DAY_UNITS = "days since 0001-01-01 00:00:00"
_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MONTH_LETTERS = "JFMAMJJASOND"
_TARGET_COLOR = "0.35"
_PREDICTION_COLOR = "tab:blue"


def days_since_epoch(time: xr.DataArray) -> torch.Tensor:
    """``(sample, time)`` float64 days of a cftime array, gatherable across ranks."""
    days = cftime.date2num(
        time.values.ravel(), units=_DAY_UNITS, calendar=time.dt.calendar
    ).reshape(time.shape)
    return torch.tensor(np.asarray(days, dtype=np.float64))


def dates_from_days(days: np.ndarray, calendar: str) -> list[cftime.datetime]:
    return list(cftime.num2date(days, units=_DAY_UNITS, calendar=calendar))


def water_year_month(month: int, start_month: int) -> int:
    """Month index within the water year, 1 for the start month through 12."""
    return (month - start_month) % 12 + 1


def water_year_slices(
    dates: Sequence[cftime.datetime], start_month: int
) -> list[slice]:
    """Index ranges of the complete water years in an ordered date sequence.

    A water year is complete when the record contains both its first day and
    the first day of the following water year.
    """
    starts = [
        i
        for i, d in enumerate(dates)
        if d.month == start_month
        and d.day == 1
        and (i == 0 or (dates[i - 1].month, dates[i - 1].day) != (start_month, 1))
    ]
    return [slice(a, b) for a, b in zip(starts[:-1], starts[1:])]


def phenology(
    values: np.ndarray, days: np.ndarray, months: np.ndarray
) -> dict[str, float]:
    """Season statistics of one complete water-year trace.

    Args:
        values: Region-mean snow amount at each timestep.
        days: Days since the start of the water year at each timestep.
        months: Month within the water year (1-12) at each timestep.

    Returns ``peak`` (maximum), ``midwinter_mean`` (mean over water-year months
    3-5), ``meltout_day`` (days from the water-year start to the first timestep
    after the peak at or below 10% of the peak; the last timestep if never
    reached) and ``summer_floor`` (mean over water-year months 10-12). All NaN
    when the trace never rises above zero.
    """
    finite = np.isfinite(values)
    if not finite.any() or np.nanmax(values) <= 0:
        return dict.fromkeys(QUANTITIES, np.nan)
    i_peak = int(np.nanargmax(values))
    peak = float(values[i_peak])
    below = np.flatnonzero(values[i_peak:] <= MELTOUT_FRACTION * peak)
    i_meltout = i_peak + int(below[0]) if below.size else len(values) - 1
    return {
        "peak": peak,
        "midwinter_mean": float(np.nanmean(values[np.isin(months, MIDWINTER_MONTHS)])),
        "meltout_day": float(days[i_meltout]),
        "summer_floor": float(np.nanmean(values[np.isin(months, SUMMER_MONTHS)])),
    }


def _mean_of_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


@dataclasses.dataclass
class _RegionTraces:
    """Complete water-year traces of one region for the gathered samples."""

    values: np.ndarray  # (sample, water_year, step), NaN-padded
    days: np.ndarray  # (sample, water_year, step), days since the water-year start
    start_year: np.ndarray  # (sample, water_year), calendar year of the start
    stats: dict[str, np.ndarray]  # each (sample, water_year)


class SnowSeasonAggregator:
    """Region-mean snow amount through each water year, with season statistics.

    Retains one area-weighted region-mean series per region for the prediction
    and the target (the box enters as a 0/1 mask; the grid operations supply
    the area weights), kilobytes of state, and at the end splits each sample's
    series into complete water years: twelve months from the first day of a
    configured start month (October north of the equator, April south by
    default), so that one snow season falls within one year. Every complete
    year is drawn against the target's, and from each the peak, midwinter mean,
    melt-out day and summer floor are read off, then averaged over years and
    samples for logging. Non-finite values of the variable count as zero, which
    is right for a snow channel that is NaN outside a snow mask.

    Only the leading timestep of the first window (the initial condition) is
    dropped; windows must arrive in time order.
    """

    def __init__(
        self,
        ops: GriddedOperations,
        lat: torch.Tensor,
        lon: torch.Tensor,
        variable: str,
        regions: Sequence[LatLonBoxConfig],
        start_month_northern: int,
        start_month_southern: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._ops = ops
        self._variable = variable
        self._regions = {region.name: region for region in regions}
        self._weights = {
            name: (region.build(lat, lon).regional_weights > 0).to(torch.float32)
            for name, region in self._regions.items()
        }
        self._start_month = {
            name: (
                start_month_northern
                if (region.lat[0] + region.lat[1]) / 2 >= 0
                else start_month_southern
            )
            for name, region in self._regions.items()
        }
        self._variable_metadata = variable_metadata or {}
        self._series: dict[str, dict[str, torch.Tensor]] = {s: {} for s in SOURCES}
        self._days: torch.Tensor | None = None
        self._calendar: str | None = None
        self._logged_missing = False

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData) -> None:
        if self._variable not in data.prediction or self._variable not in data.target:
            if not self._logged_missing:
                logging.info(
                    f"Variable {self._variable} not found in data; "
                    "snow_season metric records nothing."
                )
                self._logged_missing = True
            return
        time_slice = slice(1, None) if data.i_time_start == 0 else slice(None)
        time = data.time.isel(time=time_slice)
        if time.sizes["time"] == 0:
            return
        if self._calendar is None:
            self._calendar = time.dt.calendar
        days = days_since_epoch(time)
        self._days = (
            days if self._days is None else torch.cat([self._days, days], dim=1)
        )
        fields = {"prediction": data.prediction, "target": data.target}
        for source in SOURCES:
            field = fields[source][self._variable][:, time_slice]
            field = torch.nan_to_num(field, nan=0.0)
            for name, weights in self._weights.items():
                mean = self._ops.regional_area_weighted_mean(
                    field, weights, name=self._variable
                ).to(torch.float64)
                series = self._series[source]
                series[name] = (
                    mean
                    if name not in series
                    else torch.cat([series[name], mean], dim=1)
                )

    def _gather(self) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray] | None:
        """Gather series and times from all ranks; ``None`` off the root rank.

        Collectives are issued in the same order on every rank before the
        ``is_root`` return.
        """
        dist = Distributed.get_instance()
        assert self._days is not None

        def gathered(tensor: torch.Tensor) -> list[torch.Tensor] | None:
            if dist.world_size > 1:
                return dist.gather_irregular(tensor)
            return [tensor]

        pieces = {
            source: {
                name: gathered(self._series[source][name])
                for name in sorted(self._weights)
            }
            for source in SOURCES
        }
        days = gathered(self._days)
        if not dist.is_root() or days is None:
            return None
        series = {
            source: {
                name: torch.cat(parts, dim=0).cpu().numpy()  # type: ignore[arg-type]
                for name, parts in by_name.items()
            }
            for source, by_name in pieces.items()
        }
        return series, torch.cat(days, dim=0).cpu().numpy()

    def _traces(
        self, series: np.ndarray, days: np.ndarray, start_month: int
    ) -> _RegionTraces:
        assert self._calendar is not None
        per_sample = []
        for s in range(series.shape[0]):
            dates = dates_from_days(days[s], self._calendar)
            years = []
            for sl in water_year_slices(dates, start_month):
                values = series[s, sl]
                day = days[s, sl] - days[s, sl.start]
                months = np.array(
                    [water_year_month(d.month, start_month) for d in dates[sl]]
                )
                years.append(
                    (values, day, dates[sl.start].year, phenology(values, day, months))
                )
            per_sample.append(years)
        n_years = max(len(y) for y in per_sample)
        n_steps = max((len(v) for y in per_sample for v, *_ in y), default=0)
        shape = (series.shape[0], n_years, n_steps)
        values = np.full(shape, np.nan)
        day = np.full(shape, np.nan)
        start_year = np.full(shape[:2], -1, dtype=int)
        stats = {q: np.full(shape[:2], np.nan) for q in QUANTITIES}
        for s, years in enumerate(per_sample):
            for w, (v, d, year, st) in enumerate(years):
                values[s, w, : len(v)] = v
                day[s, w, : len(d)] = d
                start_year[s, w] = year
                for q in QUANTITIES:
                    stats[q][s, w] = st[q]
        return _RegionTraces(values, day, start_year, stats)

    def _get_traces(self) -> dict[str, dict[str, _RegionTraces]] | None:
        if not self._series["prediction"]:
            return None
        gathered = self._gather()
        if gathered is None:
            return None
        series, days = gathered
        return {
            source: {
                name: self._traces(series[source][name], days, self._start_month[name])
                for name in sorted(self._weights)
            }
            for source in SOURCES
        }

    def _display_name(self) -> str:
        if self._variable in self._variable_metadata:
            return self._variable_metadata[self._variable].display_long_name(
                self._variable
            )
        return self._variable

    def _units(self) -> str:
        if self._variable in self._variable_metadata:
            return self._variable_metadata[self._variable].display_units(
                "unknown units"
            )
        return "unknown units"

    def _draw(self, traces: dict[str, dict[str, _RegionTraces]]):
        names = sorted(self._weights)
        n_cols = 2 if len(names) > 1 else 1
        n_rows = -(-len(names) // n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6.5 * n_cols, 2.6 * n_rows), squeeze=False
        )
        for ax, name in zip(axes.ravel(), names):
            target = traces["target"][name]
            prediction = traces["prediction"][name]
            for source, color in (
                (target, _TARGET_COLOR),
                (prediction, _PREDICTION_COLOR),
            ):
                flat_v = source.values.reshape(-1, source.values.shape[-1])
                flat_d = source.days.reshape(-1, source.days.shape[-1])
                for v, d in zip(flat_v, flat_d):
                    ax.plot(d, v, color=color, lw=0.6, alpha=0.4)
                ax.plot(
                    np.nanmean(flat_d, axis=0),
                    np.nanmean(flat_v, axis=0),
                    color=color,
                    lw=2,
                )
            top = np.nanmax(target.values) if np.isfinite(target.values).any() else None
            ax.set_ylim(bottom=0, top=1.4 * top if top else None)
            start = self._start_month[name]
            order = [(start - 1 + k) % 12 for k in range(12)]
            ticks = np.concatenate(
                [[0], np.cumsum([_MONTH_LENGTHS[m] for m in order[:-1]])]
            )
            ax.set_xticks(ticks)
            ax.set_xticklabels([_MONTH_LETTERS[m] for m in order])
            ax.set_xlim(0, 365)
            ax.set_title(name, fontsize=9, loc="left")
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(names) :]:
            ax.set_visible(False)
        fig.suptitle(
            f"{self._display_name()} [{self._units()}], region mean by water year: "
            "target gray, prediction blue; thin lines single years, bold their mean",
            fontsize=9,
        )
        fig.tight_layout()
        image = WandB.get_instance().Image(fig)
        plt.close(fig)
        return image

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, Any]:
        traces = self._get_traces()
        if traces is None:
            return {}
        logs: dict[str, Any] = {}
        prefix = f"{label}/" if label else ""
        for name in sorted(self._weights):
            for quantity in QUANTITIES:
                values = {
                    source: _mean_of_finite(traces[source][name].stats[quantity])
                    for source in SOURCES
                }
                for source in SOURCES:
                    logs[f"{prefix}{source}/{name}/{quantity}"] = values[source]
                logs[f"{prefix}gap/{name}/{quantity}"] = (
                    values["prediction"] - values["target"]
                )
        logs[f"{prefix}traces"] = self._draw(traces)
        return logs

    def get_dataset(self) -> xr.Dataset:
        traces = self._get_traces()
        if traces is None:
            return xr.Dataset()
        names = sorted(self._weights)
        n_sample = max(t.values.shape[0] for t in traces["target"].values())
        n_years = max(t.values.shape[1] for s in SOURCES for t in traces[s].values())
        n_steps = max(t.values.shape[2] for s in SOURCES for t in traces[s].values())

        def padded(array: np.ndarray, shape: tuple[int, ...], fill) -> np.ndarray:
            out = np.full(shape, fill, dtype=array.dtype)
            out[tuple(slice(0, n) for n in array.shape)] = array
            return out

        trace = np.stack(
            [
                np.stack(
                    [
                        padded(
                            traces[s][n].values, (n_sample, n_years, n_steps), np.nan
                        )
                        for n in names
                    ]
                )
                for s in SOURCES
            ]
        )
        days = np.stack(
            [
                padded(traces["target"][n].days, (n_sample, n_years, n_steps), np.nan)
                for n in names
            ]
        )
        start_year = np.stack(
            [
                padded(traces["target"][n].start_year, (n_sample, n_years), -1)
                for n in names
            ]
        )
        units = self._units()
        data_vars: dict[str, tuple] = {
            "trace": (
                ["source", "region", "sample", "water_year", "step"],
                trace,
                {"long_name": f"{self._display_name()} region mean", "units": units},
            ),
            "days_since_start": (
                ["region", "sample", "water_year", "step"],
                days,
                {
                    "long_name": "days since the start of the water year",
                    "units": "days",
                },
            ),
            "start_year": (
                ["region", "sample", "water_year"],
                start_year,
                {"long_name": "calendar year in which the water year starts"},
            ),
        }
        stat_units = {
            "peak": units,
            "midwinter_mean": units,
            "meltout_day": "days",
            "summer_floor": units,
        }
        for quantity in QUANTITIES:
            data_vars[quantity] = (
                ["source", "region", "sample", "water_year"],
                np.stack(
                    [
                        np.stack(
                            [
                                padded(
                                    traces[s][n].stats[quantity],
                                    (n_sample, n_years),
                                    np.nan,
                                )
                                for n in names
                            ]
                        )
                        for s in SOURCES
                    ]
                ),
                {"units": stat_units[quantity]},
            )
        return xr.Dataset(data_vars, coords={"source": list(SOURCES), "region": names})


@dataclasses.dataclass
class SnowSeasonMetricConfig:
    """Region-mean snow amount through each water year, with season statistics.

    For one snow-amount variable, retains the area-weighted mean over each
    configured box for the prediction and the target, splits it into complete
    water years (twelve months from the first day of the start month, October
    north of the equator and April south of it by default, so that one snow
    season falls within one year) and logs, per region, the peak, the midwinter
    mean (water-year months 3-5), the melt-out day (days from the water-year
    start to the first day after the peak at or below 10% of it) and the summer
    floor (mean over water-year months 10-12), each averaged over years and
    samples, as ``{prediction,target,gap}/<region>/<quantity>``, plus one
    figure of every water year's trace against the target's. The full traces
    and per-year statistics go to the dataset. Non-finite values count as
    zero, which suits a snow channel stored as NaN outside a snow mask.
    Requires a lat-lon grid and more than two years of rollout. Disabled by
    default.

    Parameters:
        variable: Name of the snow-amount variable.
        regions: Latitude-longitude boxes to average over; a box whose
            latitude midpoint is south of the equator uses the southern
            start month.
        start_month_northern: Calendar month (1-12) that starts the water year
            for northern boxes.
        start_month_southern: The same for southern boxes.
        name: Name used to label the metric's logs and diagnostics.
        enabled: Whether the metric is computed. Disabled by default.
        strict: If True, raise rather than skip when the metric is not
            supported for the current configuration.
    """

    variable: str = "surface_snow_amount_masked"
    regions: list[LatLonBoxConfig] = dataclasses.field(default_factory=list)
    start_month_northern: int = 10
    start_month_southern: int = 4
    name: str = "snow_season"
    enabled: bool = False
    strict: bool = False

    def __post_init__(self):
        for field_name in ("start_month_northern", "start_month_southern"):
            month = getattr(self, field_name)
            if month not in range(1, 13):
                raise ValueError(
                    f"snow_season {field_name} must be in 1-12, got {month}"
                )
        names = [region.name for region in self.regions]
        if len(set(names)) != len(names):
            raise ValueError("snow_season region names must be unique")
        if self.enabled and not self.regions:
            raise ValueError("snow_season requires at least one region when enabled")

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        if not isinstance(ctx.horizontal_coordinates, LatLonCoordinates):
            raise MetricNotSupportedError(
                "snow_season metric requires LatLonCoordinates."
            )
        duration = ctx.n_timesteps * ctx.timestep
        if duration <= MINIMUM_DURATION:
            raise MetricNotSupportedError(
                "snow_season metric requires more than two years of rollout to hold a "
                f"complete water year, got {duration.days} days"
            )
        agg: SubAggregator = SnowSeasonAggregator(
            ops=ctx.ops,
            lat=ctx.horizontal_coordinates.lat,
            lon=ctx.horizontal_coordinates.lon,
            variable=self.variable,
            regions=self.regions,
            start_month_northern=self.start_month_northern,
            start_month_southern=self.start_month_southern,
            variable_metadata=ctx.variable_metadata,
        )
        return MetricBuildResult(aggregator=agg)
