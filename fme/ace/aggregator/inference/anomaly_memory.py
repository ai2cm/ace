import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping
from fme.core.wandb import Image

from ..plotting import plot_paneled_data
from .build_context import MetricBuildContext, MetricNotSupportedError, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator
from .utils import LatLonBoxConfig

ALL_MONTHS = list(range(1, 13))
N_HARMONICS = 3
RELATIVE_VARIANCE_FLOOR = 1e-10


def annual_harmonic_basis(time: xr.DataArray, n_harmonics: int) -> torch.Tensor:
    """Evaluate the climatology basis at each time.

    The basis is a constant plus ``n_harmonics`` annual harmonics of the
    fractional position of each date within its year, so leap and 360-day
    calendars are handled by the calendar of ``time`` itself.

    Args:
        time: ``(sample, time)`` array of cftime datetimes.
        n_harmonics: Number of annual harmonics.

    Returns:
        ``(sample, time, 1 + 2 * n_harmonics)`` float64 tensor.
    """
    day_of_year = np.asarray(time.dt.dayofyear.values, dtype=np.float64)
    days_in_year = np.asarray(time.dt.days_in_year.values, dtype=np.float64)
    angle = 2.0 * np.pi * (day_of_year - 0.5) / days_in_year
    columns = [np.ones_like(angle)]
    for k in range(1, n_harmonics + 1):
        columns.append(np.cos(k * angle))
        columns.append(np.sin(k * angle))
    basis = np.stack(columns, axis=-1)
    return torch.tensor(basis, dtype=torch.float64, device=get_device())


def month_mask(time: xr.DataArray, months: Sequence[int]) -> torch.Tensor:
    """Boolean ``(sample, time)`` tensor marking dates whose month is in ``months``."""
    in_season = np.isin(time.dt.month.values, list(months))
    return torch.tensor(in_season, dtype=torch.bool, device=get_device())


class LaggedAnomalyMoments:
    """Streaming sufficient statistics for lagged anomaly covariance.

    Anomalies are defined relative to a smooth seasonal climatology, a
    least-squares fit of each cell's series onto the annual-harmonic basis
    over the whole record. The lagged anomaly covariance expands as

        sum_t x'(t) x'(t+L)
            = sum_t x(t) x(t+L)
            - sum_k a_k sum_t x(t) phi_k(t+L)
            - sum_k a_k sum_t phi_k(t) x(t+L)
            + sum_kl a_k a_l sum_t phi_k(t) phi_l(t+L)

    where ``phi_k`` are the basis functions and ``a_k`` the per-cell
    coefficients. Every sum over ``t`` accumulates as windows stream through,
    and the coefficients are solved from the normal equations at the end, so
    anomalies use the full-record climatology in a single pass and no time
    series is retained. The same expansion gives the anomaly variance of the
    leading and of the lagged endpoints over exactly the pairs that enter
    each lag, so the covariance can be normalized into a Pearson correlation
    bounded by one. The state per cell is ``O(n_lags * n_basis)``.

    Pairs are counted with the leading time restricted to a per-time mask
    (the season of interest); the lagged endpoint may fall in any month.
    Non-finite values are treated as zero while accumulating and tracked in a
    per-cell count, so cells that are ever non-finite can be masked at the end.

    Windows must arrive in time order, one trajectory per sample, and the
    tail of each sample's series is retained so pairs that straddle a window
    boundary are counted exactly once.
    """

    def __init__(
        self, lags: Sequence[int], n_basis: int, spatial_shape: tuple[int, ...]
    ):
        device = get_device()
        self.lags = list(lags)
        self._max_lag = max(self.lags)
        self._n_basis = n_basis
        self._shape = tuple(spatial_shape)
        n_lags = len(self.lags)
        shape = self._shape
        f64 = dict(dtype=torch.float64, device=device)
        self.raw = torch.zeros((n_lags, *shape), **f64)
        self.x_phi = torch.zeros((n_lags, n_basis, *shape), **f64)
        self.phi_x = torch.zeros((n_lags, n_basis, *shape), **f64)
        self.phi_phi = torch.zeros((n_lags, n_basis, n_basis), **f64)
        self.lead_sq = torch.zeros((n_lags, *shape), **f64)
        self.lag_sq = torch.zeros((n_lags, *shape), **f64)
        self.lead_phi = torch.zeros((n_lags, n_basis, *shape), **f64)
        self.lag_phi = torch.zeros((n_lags, n_basis, *shape), **f64)
        self.phi_lead_lead = torch.zeros((n_lags, n_basis, n_basis), **f64)
        self.phi_lag_lag = torch.zeros((n_lags, n_basis, n_basis), **f64)
        self.counts = torch.zeros(n_lags, **f64)
        self.gram = torch.zeros((n_basis, n_basis), **f64)
        self.basis_x = torch.zeros((n_basis, *shape), **f64)
        self.n_finite = torch.zeros(shape, **f64)
        self.n_total = torch.zeros((), **f64)
        self._tail_x: list[torch.Tensor] = []
        self._tail_basis: list[torch.Tensor] = []
        self._tail_season: list[torch.Tensor] = []

    def _reset_tails(self, n_samples: int) -> None:
        device = get_device()
        self._tail_x = [
            torch.zeros((0, *self._shape), dtype=torch.float32, device=device)
            for _ in range(n_samples)
        ]
        self._tail_basis = [
            torch.zeros((0, self._n_basis), dtype=torch.float64, device=device)
            for _ in range(n_samples)
        ]
        self._tail_season = [
            torch.zeros(0, dtype=torch.bool, device=device) for _ in range(n_samples)
        ]

    def update(
        self,
        x: torch.Tensor,
        basis: torch.Tensor,
        season: torch.Tensor,
        new_trajectories: bool,
    ) -> None:
        """Record one window.

        Args:
            x: ``(sample, time, *spatial)`` values.
            basis: ``(sample, time, n_basis)`` climatology basis at each time.
            season: ``(sample, time)`` mask of leading times to count.
            new_trajectories: True when this window starts new trajectories,
                so no pairs are formed with the previous window's tail.
        """
        if new_trajectories or len(self._tail_x) != x.shape[0]:
            self._reset_tails(x.shape[0])
        finite = torch.isfinite(x)
        filled = torch.where(finite, x, torch.zeros_like(x)).to(torch.float32)
        self.n_finite += finite.to(torch.float64).sum(dim=(0, 1))
        self.n_total += x.shape[0] * x.shape[1]
        self.gram += torch.einsum("stk,stl->kl", basis, basis)
        self.basis_x += torch.einsum("stk,st...->k...", basis, filled.to(torch.float64))
        for i_sample in range(x.shape[0]):
            self._update_sample(
                i_sample, filled[i_sample], basis[i_sample], season[i_sample]
            )

    def _update_sample(
        self,
        i_sample: int,
        x: torch.Tensor,
        basis: torch.Tensor,
        season: torch.Tensor,
    ) -> None:
        xs = torch.cat([self._tail_x[i_sample], x])
        bs = torch.cat([self._tail_basis[i_sample], basis])
        ss = torch.cat([self._tail_season[i_sample], season])
        n_tail = self._tail_x[i_sample].shape[0]
        for i, lag in enumerate(self.lags):
            lo = max(0, n_tail - lag)
            hi = xs.shape[0] - lag
            if hi <= lo:
                continue
            lead = torch.arange(lo, hi, device=xs.device)
            lead = lead[ss[lead]]
            if lead.numel() == 0:
                continue
            a = xs[lead].to(torch.float64)
            b = xs[lead + lag].to(torch.float64)
            self.raw[i] += (a * b).sum(dim=0)
            self.x_phi[i] += torch.einsum("tk,t...->k...", bs[lead + lag], a)
            self.phi_x[i] += torch.einsum("tk,t...->k...", bs[lead], b)
            self.phi_phi[i] += bs[lead].T @ bs[lead + lag]
            self.lead_sq[i] += (a * a).sum(dim=0)
            self.lag_sq[i] += (b * b).sum(dim=0)
            self.lead_phi[i] += torch.einsum("tk,t...->k...", bs[lead], a)
            self.lag_phi[i] += torch.einsum("tk,t...->k...", bs[lead + lag], b)
            self.phi_lead_lead[i] += bs[lead].T @ bs[lead]
            self.phi_lag_lag[i] += bs[lead + lag].T @ bs[lead + lag]
            self.counts[i] += lead.numel()
        keep = min(self._max_lag, xs.shape[0])
        self._tail_x[i_sample] = xs[xs.shape[0] - keep :]
        self._tail_basis[i_sample] = bs[bs.shape[0] - keep :]
        self._tail_season[i_sample] = ss[ss.shape[0] - keep :]

    def reduced_state(self, dist: Distributed) -> dict[str, torch.Tensor]:
        """Sum the accumulated statistics across ranks.

        Clones before reducing, since ``reduce_sum`` mutates in place and
        finalization may be requested more than once.
        """
        names = (
            "raw",
            "x_phi",
            "phi_x",
            "phi_phi",
            "lead_sq",
            "lag_sq",
            "lead_phi",
            "lag_phi",
            "phi_lead_lead",
            "phi_lag_lag",
            "counts",
            "gram",
            "basis_x",
            "n_finite",
            "n_total",
        )
        return {name: dist.reduce_sum(getattr(self, name).clone()) for name in names}

    @staticmethod
    def finalize(state: Mapping[str, torch.Tensor]) -> "LaggedAnomalyStats":
        """Lagged anomaly moments, each ``(n_lags, *spatial)``, from reduced state.

        Cells with any non-finite value, and lags with no pairs, are NaN. So
        are cells whose anomaly variance is below ``RELATIVE_VARIANCE_FLOOR``
        times their mean square: a constant series leaves only cancellation
        roundoff in the variance, whose sign would otherwise decide at random
        whether the cell reports a meaningless correlation or NaN.
        """
        gram = state["gram"]
        basis_x = state["basis_x"]
        spatial = basis_x.shape[1:]
        coeffs = torch.linalg.pinv(gram) @ basis_x.reshape(gram.shape[0], -1)
        coeffs = coeffs.reshape(basis_x.shape)
        counts = state["counts"].reshape(-1, *([1] * len(spatial)))

        def anomaly_moment(raw, phi_a, phi_b, phi_phi):
            cross_a = torch.einsum("k...,lk...->l...", coeffs, phi_a)
            cross_b = torch.einsum("k...,lk...->l...", coeffs, phi_b)
            clim = torch.einsum("k...,lkm,m...->l...", coeffs, phi_phi, coeffs)
            return (raw - cross_a - cross_b + clim) / counts

        cov = anomaly_moment(
            state["raw"], state["x_phi"], state["phi_x"], state["phi_phi"]
        )
        var_lead = anomaly_moment(
            state["lead_sq"],
            state["lead_phi"],
            state["lead_phi"],
            state["phi_lead_lead"],
        )
        var_lag = anomaly_moment(
            state["lag_sq"], state["lag_phi"], state["lag_phi"], state["phi_lag_lag"]
        )
        all_finite = state["n_finite"] == state["n_total"]
        mean_square = state["lead_sq"][0] / state["counts"][0]
        resolved = var_lead[0] > RELATIVE_VARIANCE_FLOOR * mean_square
        keep = (counts > 0) & all_finite & resolved
        nan = torch.full_like(cov, float("nan"))
        return LaggedAnomalyStats(
            cov=torch.where(keep, cov, nan),
            var_lead=torch.where(keep, var_lead, nan),
            var_lag=torch.where(keep, var_lag, nan),
        )


@dataclasses.dataclass
class LaggedAnomalyStats:
    """Anomaly covariance and endpoint variances per lag, ``(n_lags, *spatial)``."""

    cov: torch.Tensor
    var_lead: torch.Tensor
    var_lag: torch.Tensor

    @property
    def correlation(self) -> torch.Tensor:
        """Pearson correlation per lag in [-1, 1]; NaN where a variance is zero."""
        scale = torch.sqrt(self.var_lead * self.var_lag)
        nan = torch.full_like(self.cov, float("nan"))
        return torch.where(scale > 0, (self.cov / scale).clamp(-1.0, 1.0), nan)

    @property
    def variance(self) -> torch.Tensor:
        """Anomaly variance of the scored (leading) times, ``(*spatial)``."""
        return self.var_lead[0]


class AnomalyMemoryAggregator:
    """Lagged autocorrelation of deseasonalized anomalies, per grid cell.

    Measures how long a field remembers its own anomalies: the correlation
    between a cell's anomaly today and its anomaly ``lag`` days later, pooled
    over the record and over samples. For snow and the surface fluxes it
    controls, this is the reservoir memory that a model without prognostic
    snow lacks. The metric is normalized by the anomaly variance, so it is
    blind to drift and to amplitude errors; the ratio of anomaly standard
    deviations is reported alongside as the amplitude complement.

    Statistics are accumulated per cell in a single streaming pass (see
    ``LaggedAnomalyMoments``) for the prediction and the target, then
    reduced to correlation curves. Region-mean scalars average the per-cell
    correlation over finite cells in each box, so transient snow regions are
    not swamped by spatial averaging of the series themselves. Two
    accumulators per variable and source split the grid by hemisphere so
    each hemisphere is scored in its own season.

    Device memory scales with ``n_variables * n_lags * n_basis * n_cells``
    for the accumulators and ``n_samples * max(lags) * n_cells`` for the
    retained tails.
    """

    def __init__(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        lags: Sequence[int],
        report_lags: Sequence[int],
        map_lags: Sequence[int],
        n_harmonics: int,
        months_northern: Sequence[int],
        months_southern: Sequence[int],
        regions: Sequence[LatLonBoxConfig],
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._lat = lat.to(get_device())
        self._lon = lon.to(get_device())
        self._lags = list(lags)
        self._report_lags = list(report_lags)
        self._map_lags = list(map_lags)
        self._n_harmonics = n_harmonics
        self._n_basis = 1 + 2 * n_harmonics
        self._months = {"northern": months_northern, "southern": months_southern}
        self._hemisphere_rows = {
            "northern": self._lat >= 0,
            "southern": self._lat < 0,
        }
        self._regions = list(regions)
        self._region_weights = {
            region.name: region.build(self._lat, self._lon).regional_weights
            for region in self._regions
        }
        self._variable_metadata = variable_metadata or {}
        self._moments: dict[str, dict[str, dict[str, LaggedAnomalyMoments]]] = {
            "target": {},
            "prediction": {},
        }

    def _get_moments(
        self, source: str, name: str, hemisphere: str
    ) -> LaggedAnomalyMoments:
        by_hemisphere = self._moments[source].setdefault(name, {})
        if hemisphere not in by_hemisphere:
            rows = int(self._hemisphere_rows[hemisphere].sum())
            by_hemisphere[hemisphere] = LaggedAnomalyMoments(
                self._lags, self._n_basis, (rows, len(self._lon))
            )
        return by_hemisphere[hemisphere]

    def _record_source(
        self,
        source: str,
        data: TensorMapping,
        basis: torch.Tensor,
        seasons: Mapping[str, torch.Tensor],
        new_trajectories: bool,
    ) -> None:
        for name in sorted(data):
            for hemisphere, rows in self._hemisphere_rows.items():
                if not rows.any():
                    continue
                self._get_moments(source, name, hemisphere).update(
                    data[name][:, :, rows, :],
                    basis,
                    seasons[hemisphere],
                    new_trajectories,
                )

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData) -> None:
        new_trajectories = data.i_time_start == 0
        time_slice = slice(1, None) if new_trajectories else slice(None)
        time = data.time.isel(time=time_slice)
        if time.sizes["time"] == 0:
            return
        basis = annual_harmonic_basis(time, self._n_harmonics)
        seasons = {
            hemisphere: month_mask(time, months)
            for hemisphere, months in self._months.items()
        }
        prediction = {k: v[:, time_slice] for k, v in data.prediction.items()}
        self._record_source("prediction", prediction, basis, seasons, new_trajectories)
        target = {k: v[:, time_slice] for k, v in data.target.items()}
        self._record_source("target", target, basis, seasons, new_trajectories)

    def _get_stats(self) -> dict[str, dict[str, LaggedAnomalyStats]] | None:
        """Reduce across ranks and finalize the ``(n_lags, lat, lon)`` anomaly
        moments per source and variable. Returns ``None`` on non-root ranks.

        Collectives are issued in the same deterministic order on every rank
        before the ``is_root`` return.
        """
        if not self._moments["prediction"]:
            raise ValueError("No data has been recorded yet.")
        dist = Distributed.get_instance()
        reduced: dict[str, dict[str, dict[str, dict[str, torch.Tensor]]]] = {}
        for source in sorted(self._moments):
            reduced[source] = {}
            for name in sorted(self._moments[source]):
                reduced[source][name] = {
                    hemisphere: moments.reduced_state(dist)
                    for hemisphere, moments in sorted(
                        self._moments[source][name].items()
                    )
                }
        if not dist.is_root():
            return None
        out: dict[str, dict[str, LaggedAnomalyStats]] = {}
        for source, by_name in reduced.items():
            out[source] = {}
            for name, by_hemisphere in by_name.items():
                full = {
                    field: torch.full(
                        (len(self._lags), len(self._lat), len(self._lon)),
                        float("nan"),
                        dtype=torch.float64,
                        device=get_device(),
                    )
                    for field in ("cov", "var_lead", "var_lag")
                }
                for hemisphere, state in by_hemisphere.items():
                    rows = self._hemisphere_rows[hemisphere]
                    stats = LaggedAnomalyMoments.finalize(state)
                    for field in full:
                        full[field][:, rows, :] = getattr(stats, field)
                out[source][name] = LaggedAnomalyStats(**full)
        return out

    def _region_mean(self, field: torch.Tensor, region: str) -> float:
        """Area-weighted mean of a ``(lat, lon)`` field over finite cells."""
        weights = self._region_weights[region].to(field.device)
        finite = torch.isfinite(field)
        weights = torch.where(finite, weights, torch.zeros_like(weights))
        total = weights.sum()
        if total <= 0:
            return float("nan")
        values = torch.where(finite, field, torch.zeros_like(field))
        return float((values * weights).sum() / total)

    _image_captions = {
        "maps": (
            "{name} anomaly autocorrelation at lag {lag}; "
            "(left) target and (right) generated"
        ),
        "difference_map": (
            "{name} anomaly autocorrelation at lag {lag}, generated minus target"
        ),
    }

    def _caption(self, key: str, name: str, lag: int) -> str:
        return self._image_captions[key].format(name=self._long_name(name), lag=lag)

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, Any]:
        stats = self._get_stats()
        if stats is None:
            return {}
        metrics: dict[str, float] = {}
        images: dict[str, Image] = {}
        for name in sorted(stats["prediction"]):
            gen_corr = stats["prediction"][name].correlation
            target_corr = stats["target"][name].correlation
            for lag in self._report_lags:
                i = self._lags.index(lag)
                for region in self._regions:
                    key = f"lag{lag}/{region.name}/{name}"
                    gen_value = self._region_mean(gen_corr[i], region.name)
                    target_value = self._region_mean(target_corr[i], region.name)
                    metrics[f"prediction/{key}"] = gen_value
                    metrics[f"target/{key}"] = target_value
                    metrics[f"gap/{key}"] = gen_value - target_value
            for region in self._regions:
                metrics[f"anomaly_std_ratio/{region.name}/{name}"] = float(
                    np.sqrt(
                        self._region_mean(
                            stats["prediction"][name].variance, region.name
                        )
                        / self._region_mean(stats["target"][name].variance, region.name)
                    )
                )
            for lag in self._map_lags:
                i = self._lags.index(lag)
                target_map = target_corr[i].cpu().numpy()
                gen_map = gen_corr[i].cpu().numpy()
                images[f"maps/lag{lag}/{name}"] = plot_paneled_data(
                    [[target_map, gen_map]],
                    diverging=True,
                    caption=self._caption("maps", name, lag),
                    vmin=-1.0,
                    vmax=1.0,
                )
                images[f"difference_map/lag{lag}/{name}"] = plot_paneled_data(
                    [[gen_map - target_map]],
                    diverging=True,
                    caption=self._caption("difference_map", name, lag),
                )
        logs: dict[str, Any] = {}
        if len(label) > 0:
            label = label + "/"
        logs.update({f"{label}{key}": image for key, image in images.items()})
        logs.update({f"{label}{key}": value for key, value in metrics.items()})
        return logs

    def _long_name(self, name: str) -> str:
        if name in self._variable_metadata:
            return self._variable_metadata[name].display_long_name(name)
        return name

    def get_dataset(self) -> xr.Dataset:
        stats = self._get_stats()
        if stats is None:
            return xr.Dataset()
        sources = ["target", "prediction"]
        region_names = [region.name for region in self._regions]
        data_vars: dict[str, tuple] = {}
        for name in sorted(stats["prediction"]):
            corrs = [stats[source][name].correlation for source in sources]
            data_vars[f"corr-{name}"] = (
                ["source", "lag", "lat", "lon"],
                torch.stack(corrs).cpu().numpy(),
                {"long_name": f"{self._long_name(name)} anomaly autocorrelation"},
            )
            data_vars[f"variance-{name}"] = (
                ["source", "lat", "lon"],
                torch.stack([stats[source][name].variance for source in sources])
                .cpu()
                .numpy(),
                {"long_name": f"{self._long_name(name)} anomaly variance"},
            )
            data_vars[f"region_corr-{name}"] = (
                ["source", "region", "lag"],
                np.array(
                    [
                        [
                            [
                                self._region_mean(corr[i], region)
                                for i in range(len(self._lags))
                            ]
                            for region in region_names
                        ]
                        for corr in corrs
                    ]
                ).reshape(len(sources), len(region_names), len(self._lags)),
                {
                    "long_name": (
                        f"{self._long_name(name)} anomaly autocorrelation, "
                        "region mean of per-cell values"
                    )
                },
            )
        return xr.Dataset(
            data_vars,
            coords={
                "source": sources,
                "lag": np.array(self._lags),
                "region": region_names,
            },
        )


@dataclasses.dataclass
class AnomalyMemoryMetricConfig:
    """Lagged autocorrelation of deseasonalized anomalies, per grid cell.

    For each variable, the correlation between a cell's anomaly from its
    seasonal climatology and the anomaly ``lag`` steps later, for the
    prediction and the target, accumulated in a single pass. Region-mean
    scalars are logged as ``{prediction,target,gap}/lag<L>/<region>/<var>``,
    the ratio of predicted to target anomaly standard deviation (each relative
    to its own climatology, so it measures anomaly amplitude independent of
    bias) as ``anomaly_std_ratio/<region>/<var>``, maps as ``maps/lag<L>/<var>``
    (target beside prediction) and ``difference_map/lag<L>/<var>``, and the
    full curves go to the dataset.
    The seasonal climatology is a constant plus three annual harmonics fit per
    cell over the whole record. Intended for fields with a slow reservoir such
    as snow and the surface fluxes it controls. Disabled by default.

    Parameters:
        variables: Variables to compute memory for. If ``None``, all
            available variables are included.
        lags: Lags in timesteps at which to accumulate. Must include 0,
            which normalizes the covariances to correlations.
        report_lags: Lags at which region-mean scalars are logged. Must be a
            subset of ``lags``.
        map_lags: Lags at which target/prediction maps are logged. Must be a
            subset of ``lags``; empty disables maps.
        months_northern: Calendar months (1-12) whose anomalies count as the
            leading time for cells at or north of the equator.
        months_southern: The same for cells south of the equator.
        regions: Latitude-longitude boxes over which per-cell correlations
            are averaged for the logged scalars.
        name: Name used to label the metric's logs and diagnostics.
        enabled: Whether the metric is computed. Disabled by default.
        strict: If True, raise rather than skip when the metric is not
            supported for the current configuration.
    """

    variables: list[str] | None = None
    lags: list[int] = dataclasses.field(default_factory=lambda: [0, 1, 3, 7, 14, 30])
    report_lags: list[int] = dataclasses.field(default_factory=lambda: [7])
    map_lags: list[int] = dataclasses.field(default_factory=lambda: [7])
    months_northern: list[int] = dataclasses.field(default_factory=lambda: ALL_MONTHS)
    months_southern: list[int] = dataclasses.field(default_factory=lambda: ALL_MONTHS)
    regions: list[LatLonBoxConfig] = dataclasses.field(default_factory=list)
    name: str = "anomaly_memory"
    enabled: bool = False
    strict: bool = False

    def __post_init__(self):
        if 0 not in self.lags:
            raise ValueError("anomaly_memory lags must include 0")
        if any(lag < 0 for lag in self.lags):
            raise ValueError("anomaly_memory lags must be non-negative")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("anomaly_memory lags must be unique")
        for field_name in ("report_lags", "map_lags"):
            missing = set(getattr(self, field_name)) - set(self.lags)
            if missing:
                raise ValueError(
                    f"anomaly_memory {field_name} {sorted(missing)} not in lags"
                )
        for field_name in ("months_northern", "months_southern"):
            months = getattr(self, field_name)
            if not months or any(m not in ALL_MONTHS for m in months):
                raise ValueError(
                    f"anomaly_memory {field_name} must be a non-empty subset of 1-12"
                )
        names = [region.name for region in self.regions]
        if len(set(names)) != len(names):
            raise ValueError("anomaly_memory region names must be unique")

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        if not isinstance(ctx.horizontal_coordinates, LatLonCoordinates):
            raise MetricNotSupportedError(
                "anomaly_memory metric requires LatLonCoordinates."
            )
        if ctx.n_forward_steps <= max(self.lags):
            raise MetricNotSupportedError(
                f"anomaly_memory metric requires more than {max(self.lags)} "
                f"forward steps, got {ctx.n_forward_steps}"
            )
        agg: SubAggregator = AnomalyMemoryAggregator(
            lat=ctx.horizontal_coordinates.lat,
            lon=ctx.horizontal_coordinates.lon,
            lags=self.lags,
            report_lags=self.report_lags,
            map_lags=self.map_lags,
            n_harmonics=N_HARMONICS,
            months_northern=self.months_northern,
            months_southern=self.months_southern,
            regions=self.regions,
            variable_metadata=ctx.variable_metadata,
        )
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))
