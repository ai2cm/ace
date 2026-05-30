"""Compute per-dataset summary statistics over the pilot's processed
zarrs.

For each ``ok`` dataset in ``<output_directory>/index.csv``:

* Open the zarr.
* For each data variable, compute area-weighted scalar statistics
  (mean, std, p01/p50/p99, skewness, kurtosis, lag-1 autocorr,
  finite_fraction) and analogues for the one-step finite difference
  in time (d1_mean, d1_std). 3D variables produce a value per plev.
* Write a per-dataset ``stats.nc`` next to ``data.zarr``.
* Append rows to a tidy aggregate (one row per
  ``(dataset, variable, plev)``).

The aggregate is written as ``<output_directory>/stats.csv`` and
(when pyarrow is installed) ``stats.parquet``.

Statistics are area-weighted with **Gauss-Legendre quadrature
weights**, which is the natural cell-area weighting on the F22.5
grid (latitudes are GL roots, not equispaced; cos(lat) weighting is
the wrong normalization here).

Usage:
    python compute_stats.py --config configs/pilot.yaml
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import dask
import fsspec
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from config import ProcessConfig  # noqa: E402
from grid import GAUSSIAN_GRID_N  # noqa: E402

# ---------------------------------------------------------------------------
# Area weights from the Gauss-Legendre target grid
# ---------------------------------------------------------------------------


def gauss_legendre_weights(n: float) -> np.ndarray:
    """Sorted south-to-north Gauss-Legendre quadrature weights for grid
    F<N> (nlat = 2N).
    """
    from numpy.polynomial.legendre import leggauss

    nlat = round(2 * n)
    x, w = leggauss(nlat)
    idx = np.argsort(x)  # south to north
    return w[idx]


def area_weights_2d(grid_name: str, n_lon: int) -> np.ndarray:
    """Normalized 2D ``(lat, lon)`` area weights. Sums to 1."""
    n = GAUSSIAN_GRID_N[grid_name]
    w_lat = gauss_legendre_weights(n)
    w2d = np.tile(w_lat[:, None], (1, n_lon))
    w2d /= w2d.sum()
    return w2d


# ---------------------------------------------------------------------------
# Scalar statistics
# ---------------------------------------------------------------------------

# Variables we don't need stats for (mask channels — uninformative).
# Includes both pre-flatten (below_surface_mask) and post-flatten
# pressure-named forms (below_surface_mask1000, etc.).
_SKIP_VARS = frozenset(
    ("below_surface_mask", "siconc_mask")
    + tuple(f"below_surface_mask{p}" for p in (1000, 850, 700, 500, 250, 100, 50, 10))
)

_SCALAR_NAMES = (
    "mean",
    "std",
    # Variance decomposition: ``clim_std`` is the area-weighted spatial
    # std of the time-mean climatology; ``anom_std`` is the std of the
    # remaining (X - time_mean) anomaly. Identity (orthogonal):
    # ``std**2 = clim_std**2 + anom_std**2`` exactly under uniform
    # time weighting + area-summed-to-1 spatial weights. ``clim_var_frac``
    # = ``clim_std**2 / std**2`` is the share of total variance carried
    # by the static spatial pattern; high values (e.g. ``ts`` ~ 0.99)
    # mean total ``std`` mostly reflects "warm tropics vs cold poles"
    # and ``d1_std / std`` understates day-to-day predictability — use
    # ``d1_std / anom_std`` instead.
    "clim_std",
    "anom_std",
    "clim_var_frac",
    "d1_mean",
    "d1_std",
    "p01",
    "p50",
    "p99",
    "skewness",
    "kurtosis",
    "autocorr_lag1",
    "finite_fraction",
)


def _nan_dict() -> dict[str, float]:
    return {name: float("nan") for name in _SCALAR_NAMES}


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    w = weights / weights.sum()
    m = float(np.sum(values * w))
    var = float(np.sum((values - m) ** 2 * w))
    return m, float(np.sqrt(max(var, 0.0)))


def _moment_stats(values: np.ndarray, mean: float, std: float) -> tuple[float, float]:
    """Unweighted skewness and excess kurtosis. Weighted versions are
    tedious and the unweighted estimates are fine for ML diagnostics.
    """
    if std <= 0 or values.size < 3:
        return float("nan"), float("nan")
    centered = values - mean
    skew = float(np.mean(centered**3) / std**3)
    excess_kurt = float(np.mean(centered**4) / std**4 - 3.0)
    return skew, excess_kurt


def _scalar_stats_from_maps(
    time_mean_map: np.ndarray,
    time_var_map: np.ndarray,
    n_valid_map: np.ndarray,
    w2d: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Derive area- and time-weighted scalar (mean, std, clim_std,
    anom_std, clim_var_frac) from per-cell maps via the law of total
    variance.

    For a field ``x(t, xy)`` with area weights ``w(xy)`` and per-cell
    finite-count ``n(xy)``, the (n_valid-weighted, area-weighted)
    population mean and variance decompose as::

        N(xy)   = w(xy) * n(xy)
        N       = Σ_xy N(xy)
        μ       = Σ_xy N(xy) * time_mean(xy) / N
        σ²      = Σ_xy N(xy) * [time_var(xy) + (time_mean(xy) - μ)²] / N

    (Total variance = mean of within-cell variance + variance of
    cell means, area- and count-weighted.) Climatology variance is
    the area-weighted spatial variance of ``time_mean``; anomaly
    variance is ``σ² - clim_var`` by construction. All inputs are
    small ``(lat, lon)`` arrays, so this runs in microseconds.
    """
    valid = np.isfinite(time_mean_map) & (n_valid_map > 0)
    if not valid.any():
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    w_eff = (w2d * n_valid_map).astype(np.float64)
    w_eff = np.where(valid, w_eff, 0.0)
    total_w = float(w_eff.sum())
    if total_w <= 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    tm = np.where(valid, time_mean_map.astype(np.float64), 0.0)
    tv = np.where(valid, time_var_map.astype(np.float64), 0.0)
    mu = float((w_eff * tm).sum() / total_w)
    var = float((w_eff * (tv + (tm - mu) ** 2)).sum() / total_w)
    std = float(np.sqrt(max(var, 0.0)))

    # Climatology contribution: area-weighted spatial variance of the
    # time-mean field. Uses pure area weights ``w2d`` (no n-weighting,
    # since this is a per-cell scalar field).
    w_area = np.where(valid, w2d.astype(np.float64), 0.0)
    tw = float(w_area.sum())
    if tw <= 0 or std == 0:
        return mu, std, float("nan"), float("nan"), float("nan")
    clim_mean = float((w_area * tm).sum() / tw)
    clim_var = float((w_area * (tm - clim_mean) ** 2).sum() / tw)
    clim_std = float(np.sqrt(max(clim_var, 0.0)))
    if var > 0:
        anom_var = max(var - clim_var, 0.0)
        anom_std = float(np.sqrt(anom_var))
        clim_frac = float(min(max(clim_var / var, 0.0), 1.0))
    else:
        anom_std = 0.0
        clim_frac = float("nan")
    return mu, std, clim_std, anom_std, clim_frac


def _stats_for_time_field(
    da: xr.DataArray, w2d: np.ndarray
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Stats for a ``(time, lat, lon)`` DataArray.

    Reductions are expressed via xarray operations so a dask-backed
    ``da`` is reduced chunk-by-chunk without ever materialising the
    full ``(time, lat, lon)`` array. Peak memory is dominated by a
    single chunk's worth of data per dask worker (≈ tens of MB at
    the zarr's native chunk size) rather than the whole variable
    (≈ 500 MB for an 86-year ssp variable, ×3 intermediates per
    ``np.diff`` / ``np.nanvar`` call). That's the difference between
    a per-dataset pod staying under 4 GiB RSS and OOM-killing at
    32 GiB.

    Returns ``(scalar_stats, maps)``. ``maps`` always has keys
    ``time_mean_map``, ``time_var_map``, ``n_valid_map``, and
    ``d1_var_map`` (all ``(lat, lon)``, float32 except ``n_valid_map``
    which is int32). Persisting per-cell maps lets
    ``make_normalization`` aggregate pooled per-cell mean / std /
    tendency-std across datasets via the law of total variance,
    without re-opening every zarr.

    Scalar global ``mean`` / ``std`` are derived from the per-cell
    maps via :func:`_scalar_stats_from_maps` (same identity used by
    ``make_normalization`` for cross-dataset pooling), which is
    exactly equivalent to the previous direct weighted reduction over
    all valid ``(t, xy)`` cells — no second pass through raw data.

    Percentiles, skewness, and kurtosis use a strided time-subsample
    (~600 timesteps) since exact percentiles need a sort and the
    subsample is statistically sufficient for diagnostics. The
    sample is small enough to fit in memory regardless of variable
    size.
    """
    out = _nan_dict()
    lat_lon_shape = (int(da.sizes["lat"]), int(da.sizes["lon"]))
    maps: dict[str, np.ndarray] = {
        "time_mean_map": np.full(lat_lon_shape, np.nan, dtype=np.float32),
        "time_var_map": np.full(lat_lon_shape, np.nan, dtype=np.float32),
        "n_valid_map": np.zeros(lat_lon_shape, dtype=np.int32),
        "d1_var_map": np.full(lat_lon_shape, np.nan, dtype=np.float32),
    }

    # --- Pass 1: per-cell maps + spatial-mean time series ---
    # All reductions on (time, lat, lon) → (lat, lon) or (time,).
    # ``warnings.catch_warnings`` suppresses "Mean of empty slice" /
    # "Degrees of freedom <= 0" — expected for all-NaN cells (e.g.
    # land in an ocean variable).
    finite = da.notnull()
    n_valid_lazy = finite.sum("time").astype(np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        time_mean_lazy = da.mean("time", skipna=True)
        time_var_lazy = da.var("time", skipna=True)
        spatial_mean_t_lazy = (da * w2d).sum(("lat", "lon"), skipna=True)

        n_valid_map, time_mean_map, time_var_map, spatial_mean_t = dask.compute(
            n_valid_lazy,
            time_mean_lazy,
            time_var_lazy,
            spatial_mean_t_lazy,
        )

    maps["n_valid_map"] = n_valid_map.values.astype(np.int32)
    maps["time_mean_map"] = time_mean_map.values.astype(np.float32)
    maps["time_var_map"] = time_var_map.values.astype(np.float32)

    n_valid_total = int(maps["n_valid_map"].sum())
    n_total = int(np.prod(da.shape))
    out["finite_fraction"] = float(n_valid_total / n_total) if n_total else 0.0
    if n_valid_total == 0:
        return out, maps

    # Scalar mean/std/clim_std/anom_std via law-of-total-variance.
    # Mathematically identical to the prior direct weighted reduction
    # over all (t, xy) finite cells, but uses only the maps we just
    # produced.
    mu, sigma, clim_std, anom_std, clim_frac = _scalar_stats_from_maps(
        maps["time_mean_map"], maps["time_var_map"], maps["n_valid_map"], w2d
    )
    out["mean"] = mu
    out["std"] = sigma
    if np.isfinite(clim_std):
        out["clim_std"] = clim_std
    if np.isfinite(anom_std):
        out["anom_std"] = anom_std
    if np.isfinite(clim_frac):
        out["clim_var_frac"] = clim_frac

    # --- Pass 2: strided time-subsample for percentiles + moments ---
    # ``da.isel(time=slice(None, None, stride)).values`` materialises
    # only the subsampled slab — ~600 timesteps × lat × lon, which
    # is ~10 MB regardless of how long the original time axis is.
    # Percentiles + skew + kurt are diagnostics; the subsample
    # estimator is unbiased and the error is dominated by far less
    # than the cross-source spread we care about.
    n_time = int(da.sizes["time"])
    stride = max(1, n_time // 600)
    sampled = da.isel(time=slice(None, None, stride)).values
    sample_finite = sampled[np.isfinite(sampled)]
    if sample_finite.size > 0:
        out["p01"] = float(np.percentile(sample_finite, 1))
        out["p50"] = float(np.percentile(sample_finite, 50))
        out["p99"] = float(np.percentile(sample_finite, 99))
        if out["std"] > 0 and sample_finite.size > 2:
            centered = sample_finite - out["mean"]
            out["skewness"] = float(np.mean(centered**3) / out["std"] ** 3)
            out["kurtosis"] = float(np.mean(centered**4) / out["std"] ** 4 - 3.0)

    # --- d1 stats via per-cell map decomposition ---
    if n_time > 1:
        d1 = da.diff("time")
        d1_finite = d1.notnull()
        d1_n_valid_lazy = d1_finite.sum("time").astype(np.int32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            d1_time_mean_lazy = d1.mean("time", skipna=True)
            d1_time_var_lazy = d1.var("time", skipna=True)
            d1_n_valid_map, d1_time_mean_map, d1_time_var_map = dask.compute(
                d1_n_valid_lazy, d1_time_mean_lazy, d1_time_var_lazy
            )
        maps["d1_var_map"] = d1_time_var_map.values.astype(np.float32)
        d1_mu, d1_sigma, _, _, _ = _scalar_stats_from_maps(
            d1_time_mean_map.values.astype(np.float32),
            d1_time_var_map.values.astype(np.float32),
            d1_n_valid_map.values.astype(np.int32),
            w2d,
        )
        if np.isfinite(d1_mu):
            out["d1_mean"] = d1_mu
        if np.isfinite(d1_sigma):
            out["d1_std"] = d1_sigma

    # --- Lag-1 autocorrelation of the area-weighted spatial mean ---
    # ``spatial_mean_t`` was already reduced to (time,) in pass 1; the
    # series is tiny (≤ 32k floats for the longest ssp). The sum of
    # NaN-weighted contributions can dip slightly negative below
    # rounding; treat values that are zero-after-finite as the
    # legacy ``np.nansum`` behaviour did.
    sm_full = spatial_mean_t.values
    sm = sm_full[np.isfinite(sm_full)]
    if sm.size > 2:
        c = sm - sm.mean()
        denom = float((c**2).sum())
        if denom > 0:
            out["autocorr_lag1"] = float((c[:-1] * c[1:]).sum() / denom)
    return out, maps


def _stats_for_static_field(
    da: xr.DataArray, w2d: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    """Stats for a ``(lat, lon)`` static field (no time, no d1).

    Static fields are (lat, lon)-shaped and already small (≈ 16 KB
    on F22.5), so we materialise once and compute everything with
    plain numpy. Returns ``(scalar_stats, static_map)`` where
    ``static_map`` is the cast-to-float32 input. Persisted once per
    variable (not per period) so ``make_normalization`` can read
    static fields out of ``stats.nc`` without re-opening the zarr.
    """
    arr = np.asarray(da.values)
    out = _nan_dict()
    valid = np.isfinite(arr)
    out["finite_fraction"] = float(valid.mean())
    static_map = arr.astype(np.float32)
    if not valid.any():
        return out, static_map

    val_v = arr[valid]
    val_w = w2d[valid]

    out["mean"], out["std"] = _weighted_mean_std(val_v, val_w)
    out["p01"] = float(np.percentile(val_v, 1))
    out["p50"] = float(np.percentile(val_v, 50))
    out["p99"] = float(np.percentile(val_v, 99))
    out["skewness"], out["kurtosis"] = _moment_stats(val_v, out["mean"], out["std"])
    # d1 and autocorr left as NaN.
    return out, static_map


# ---------------------------------------------------------------------------
# Per-dataset stats
# ---------------------------------------------------------------------------


def _slice_ds_to_period(ds: xr.Dataset, period) -> Optional[xr.Dataset]:
    """Restrict ``ds`` to a ``StatsPeriod``'s time window. Returns the
    sliced dataset, or ``None`` when no timesteps fall in the period.
    ``ds`` without a ``time`` dim is returned unchanged (statics).
    """
    if "time" not in ds.dims:
        return ds
    if period.start is None and period.end is None:
        return ds
    times = ds["time"].values
    if not len(times):
        return None
    if len(times) and hasattr(times[0], "calendar"):
        # Use the calendar-aware date clipper from ``processing.py`` so
        # 360-day calendars (where Dec 31 doesn't exist) don't crash on
        # the end-of-day construction below.
        from processing import clip_date_for_calendar

        date_type = type(times[0])
        calendar = str(times[0].calendar)

        def to_dt(s, *, end: bool):
            clipped = clip_date_for_calendar(s, calendar)
            y, m, d = (int(x) for x in clipped.split("-"))
            return date_type(y, m, d, 23, 59, 59) if end else date_type(y, m, d)
    else:

        def to_dt(s, *, end: bool):
            return (
                np.datetime64(s) + np.timedelta64(1, "D") - np.timedelta64(1, "s")
                if end
                else np.datetime64(s)
            )

    mask = np.ones(len(times), dtype=bool)
    if period.start is not None:
        mask &= times >= to_dt(period.start, end=False)
    if period.end is not None:
        mask &= times <= to_dt(period.end, end=True)
    if not mask.any():
        return None
    return ds.isel(time=np.where(mask)[0])


def _nan_stats() -> dict[str, float]:
    return _nan_dict()


# Names of the per-cell maps emitted by ``_stats_for_time_field``.
# Order is the persistence order in stats.nc; ``n_valid_map`` is int32,
# the rest are float32.
_MAP_NAMES = ("time_mean_map", "time_var_map", "n_valid_map", "d1_var_map")
_INT_MAP_NAMES = frozenset({"n_valid_map"})


def _empty_period_maps(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """All-NaN / zero-count map dict for a period with no data."""
    return {
        name: (
            np.zeros(shape, dtype=np.int32)
            if name in _INT_MAP_NAMES
            else np.full(shape, np.nan, dtype=np.float32)
        )
        for name in _MAP_NAMES
    }


def _stack_plev_maps(
    maps_per_plev: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stack a list of ``(lat, lon)`` map dicts along a new leading
    ``plev`` axis, keyed by map name."""
    return {
        name: np.stack([m[name] for m in maps_per_plev], axis=0) for name in _MAP_NAMES
    }


def compute_dataset_stats(
    ds: xr.Dataset,
    w2d: np.ndarray,
    periods=None,
) -> tuple[xr.Dataset, list[dict]]:
    """Compute stats for every data variable in ``ds`` over one or more
    named time periods.

    ``periods`` is a sequence of ``StatsPeriod`` objects (defaults to
    ``DEFAULT_STATS_PERIODS``). Each period produces a row in a new
    ``period`` dim of the returned dataset; periods with no overlap on
    a given variable get NaN stats. Static (no-``time``) variables get
    the same values across all periods.

    Returns ``(stats_ds, rows)``:
    * ``stats_ds``: ``{var}__{stat}`` variables, each with shape
      ``(period,)``, ``(period, plev)`` for 3D-source variables. Coord
      ``period`` is a string label; ``plev`` carried through when
      present.
      For time-varying variables, also includes a ``{var}__time_mean_map``
      variable with shape ``(period, lat, lon)`` (2D) or
      ``(period, plev, lat, lon)`` (3D). This lets
      ``make_normalization`` aggregate pooled time-mean maps cheaply
      across datasets instead of re-scanning every zarr.
    * ``rows``: tidy aggregate, one dict per ``(variable, plev, period)``.
    """
    from config import DEFAULT_STATS_PERIODS

    if periods is None:
        periods = DEFAULT_STATS_PERIODS
    periods = list(periods)
    period_names = [p.name for p in periods]

    vars_to_process = [v for v in ds.data_vars if v not in _SKIP_VARS]
    plev_size = int(ds.sizes.get("plev", 0))
    lat_size = int(ds.sizes.get("lat", 0))
    lon_size = int(ds.sizes.get("lon", 0))

    nc_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    rows: list[dict] = []

    # Per-variable stats accumulator: stats_by_var[v][period_index]
    # = either a list of per-plev dicts (3D) or a single dict.
    # Pre-allocate so empty periods (no overlap) emit NaN cleanly.
    for v in vars_to_process:
        da = ds[v]
        is_3d = "time" in da.dims and "plev" in da.dims
        is_static = "time" not in da.dims

        # Accumulators for scalar stats (dicts) and the per-period
        # per-cell maps. Maps are stored separately because they're
        # only persisted for time-varying variables and don't enter the
        # tidy ``rows`` aggregate. ``per_period_maps[pi]`` is a dict
        # of per-cell arrays for that period (keys match the names
        # returned by ``_stats_for_time_field``).
        per_period_stats: list = []
        per_period_maps: list[dict[str, np.ndarray]] = []
        static_map: Optional[np.ndarray] = None
        for p in periods:
            if is_static:
                # Static: identical across periods; only compute once.
                if per_period_stats:
                    per_period_stats.append(per_period_stats[0])
                else:
                    stats, static_map = _stats_for_static_field(da, w2d)
                    per_period_stats.append(stats)
                continue
            sub = _slice_ds_to_period(ds, p)
            if sub is None:
                # No data in this period — emit NaN stats.
                if is_3d:
                    per_period_stats.append([_nan_stats() for _ in range(plev_size)])
                    per_period_maps.append(
                        _empty_period_maps((plev_size, lat_size, lon_size))
                    )
                else:
                    per_period_stats.append(_nan_stats())
                    per_period_maps.append(_empty_period_maps((lat_size, lon_size)))
                continue
            sub_da = sub[v]
            if is_3d:
                stats_per_plev: list = []
                maps_per_plev: list[dict[str, np.ndarray]] = []
                for k in range(sub_da.sizes["plev"]):
                    s, m = _stats_for_time_field(sub_da.isel(plev=k), w2d)
                    stats_per_plev.append(s)
                    maps_per_plev.append(m)
                per_period_stats.append(stats_per_plev)
                per_period_maps.append(_stack_plev_maps(maps_per_plev))
            else:
                s, m = _stats_for_time_field(sub_da, w2d)
                per_period_stats.append(s)
                per_period_maps.append(m)

        # Materialize per-stat arrays. 3D shape: (period, plev). 2D / static
        # shape: (period,).
        for stat in _SCALAR_NAMES:
            if is_3d:
                arr = np.array(
                    [
                        [d.get(stat, float("nan")) for d in per_p]
                        for per_p in per_period_stats
                    ],
                    dtype=float,
                )
                nc_vars[f"{v}__{stat}"] = (("period", "plev"), arr)
            else:
                arr = np.array(
                    [d.get(stat, float("nan")) for d in per_period_stats],
                    dtype=float,
                )
                nc_vars[f"{v}__{stat}"] = (("period",), arr)

        # Persist per-period per-cell maps for time-varying variables.
        # Static vars get a single ``{v}__static_map`` with shape
        # ``(lat, lon)`` (or ``(plev, lat, lon)`` for 3D statics —
        # unused today, but cheap to support) since they're identical
        # across periods.
        if is_static:
            if static_map is not None and lat_size > 0 and lon_size > 0:
                if static_map.ndim == 3:
                    nc_vars[f"{v}__static_map"] = (
                        ("plev", "lat", "lon"),
                        static_map.astype(np.float32),
                    )
                else:
                    nc_vars[f"{v}__static_map"] = (
                        ("lat", "lon"),
                        static_map.astype(np.float32),
                    )
        elif per_period_maps and lat_size > 0 and lon_size > 0:
            dims: tuple[str, ...] = (
                ("period", "plev", "lat", "lon") if is_3d else ("period", "lat", "lon")
            )
            for map_name in _MAP_NAMES:
                stacked = np.stack([pm[map_name] for pm in per_period_maps], axis=0)
                nc_vars[f"{v}__{map_name}"] = (dims, stacked)

        # Tidy rows for the aggregate, one per (variable, plev, period).
        for pi, period_name in enumerate(period_names):
            if is_3d:
                for k in range(plev_size):
                    row = {
                        "variable": v,
                        "plev_index": float(k),
                        "plev_pa": float(ds["plev"].values[k]),
                        "period": period_name,
                    }
                    row.update(per_period_stats[pi][k])
                    rows.append(row)
            else:
                row = {
                    "variable": v,
                    "plev_index": float("nan"),
                    "plev_pa": float("nan"),
                    "period": period_name,
                }
                row.update(per_period_stats[pi])
                rows.append(row)

    coords: dict[str, xr.DataArray] = {
        "period": xr.DataArray(np.array(period_names, dtype=object), dims=("period",)),
    }
    if "plev" in ds.coords:
        coords["plev"] = ds.plev
    # Carry lat/lon for the ``{var}__time_mean_map`` arrays so consumers
    # can align them to a target grid without round-tripping through the
    # source zarr.
    if "lat" in ds.coords:
        coords["lat"] = ds.lat
    if "lon" in ds.coords:
        coords["lon"] = ds.lon
    data = {
        name: xr.DataArray(np.asarray(payload), dims=dims)
        for name, (dims, payload) in nc_vars.items()
    }
    stats_ds = xr.Dataset(data, coords=coords)
    # Persist the period start/end for downstream consumers — useful
    # when reading stats without having the config handy.
    stats_ds["period_start"] = xr.DataArray(
        np.array([p.start or "" for p in periods], dtype=object), dims=("period",)
    )
    stats_ds["period_end"] = xr.DataArray(
        np.array([p.end or "" for p in periods], dtype=object), dims=("period",)
    )
    return stats_ds, rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _process_one(
    zarr_path: str,
    stats_path: str,
    identity: dict,
    grid_name: str,
    force: bool,
    periods: Optional[tuple] = None,
) -> tuple[str, list[dict], str]:
    """Compute or reuse stats for a single dataset.

    Runs in a worker process; logs via the root logger (configured in
    each subprocess on import). Returns ``(status, rows, message)`` so
    the parent can aggregate without sharing state.
    """
    fs, rel = fsspec.core.url_to_fs(stats_path)
    if fs.exists(rel) and not force:
        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            with fs.open(rel, "rb") as fobj:
                tmp.write(fobj.read())
                tmp.flush()
            stats_ds = xr.open_dataset(tmp.name)
            rows = _rows_from_stats_ds(stats_ds, identity)
            stats_ds.close()
        return "reused", rows, stats_path

    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception as e:  # noqa: BLE001
        return "error", [], f"could not open {zarr_path}: {e}"

    try:
        dataset_rows = compute_and_write_stats(
            ds, stats_path, identity, grid_name, periods=periods
        )
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        return ("error", [], f"compute failed for {zarr_path}: {e}\n{tb}")
    finally:
        ds.close()

    return "computed", dataset_rows, zarr_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Process YAML")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute stats.nc even where it already exists.",
    )
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=None,
        help="Optional subset of source_ids to compute stats for.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel worker processes. Defaults to the pod's "
            "CPU count (``os.cpu_count()``). Use 1 to disable pooling "
            "(easier debugging)."
        ),
    )
    parser.add_argument(
        "--zarr-path",
        default=None,
        help=(
            "Compute stats for a single dataset at this zarr URL "
            "instead of iterating ``index.csv``. The dataset's "
            "``metadata.json`` sidecar is read for identity; the "
            "stats.nc is written next to the zarr. Mirrors the "
            "``migrate.py --zarr-path`` mode and is used by the "
            "per-dataset Argo ``compute-stats-dataset`` template so "
            "stats regen parallelises naturally across pods rather "
            "than via an in-pod worker pool (which OOMs at scale on "
            "the larger 86-year ssp variables)."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    out_dir = cfg.output_directory.rstrip("/")
    grid_name = cfg.defaults.target_grid.name
    periods = tuple(cfg.defaults.stats_periods)
    logging.info(
        "Using %d stats period(s): %s",
        len(periods),
        ", ".join(p.name for p in periods),
    )

    jobs: list[tuple] = []
    if args.zarr_path is not None:
        # Single-dataset mode (per-pod operation; identity comes from
        # the zarr's metadata.json sidecar so we don't depend on
        # index.csv being current).
        sidecar_url = args.zarr_path.rstrip("/") + "/metadata.json"
        with fsspec.open(sidecar_url, "r") as f:
            sidecar = json.load(f)
        stats_path = args.zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
        identity = {
            "source_id": sidecar.get("source_id", ""),
            "experiment": sidecar.get("experiment", ""),
            "variant_label": sidecar.get("variant_label", ""),
            "label": sidecar.get("label", ""),
        }
        jobs.append(
            (args.zarr_path, stats_path, identity, grid_name, args.force, periods)
        )
        logging.info(
            "Single-dataset mode: %s/%s/%s",
            identity["source_id"],
            identity["experiment"],
            identity["variant_label"],
        )
    else:
        # Index-scan mode (legacy / local-driver path).
        index_path = f"{out_dir}/index.csv"
        fs, rel = fsspec.core.url_to_fs(index_path)
        if not fs.exists(rel):
            raise FileNotFoundError(f"{index_path} not found; run process.py first")
        idx = pd.read_csv(index_path)

        ok = idx[idx.status == "ok"].reset_index(drop=True)
        if args.source_ids is not None:
            ok = ok[ok.source_id.isin(args.source_ids)].reset_index(drop=True)
        logging.info("Found %d ok datasets in the index.", len(ok))

        for _, r in ok.iterrows():
            zarr_path = r.output_zarr
            stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
            identity = {
                "source_id": r.source_id,
                "experiment": r.experiment,
                "variant_label": r.variant_label,
                "label": r.get("label", ""),
            }
            jobs.append(
                (zarr_path, stats_path, identity, grid_name, args.force, periods)
            )

    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(jobs))) if jobs else 1
    logging.info("Computing stats with %d worker(s).", workers)

    rows: list[dict] = []
    n_done = 0
    n_reused = 0
    n_error = 0

    if workers == 1:
        # Single-process path for easier local debugging.
        for i, job in enumerate(jobs):
            status, job_rows, msg = _process_one(*job)
            if status == "reused":
                logging.info("[%d/%d] reuse %s", i + 1, len(jobs), msg)
                n_reused += 1
            elif status == "computed":
                logging.info("[%d/%d] computed %s", i + 1, len(jobs), msg)
                n_done += 1
            else:
                logging.warning("[%d/%d] %s", i + 1, len(jobs), msg)
                n_error += 1
            rows.extend(job_rows)
    else:
        # ``spawn`` (not the Linux default ``fork``) — gcsfs holds gRPC
        # threads that abort with "Check failed: next_worker->state ==
        # KICKED" when copied across fork().
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(_process_one, *job): job for job in jobs}
            for i, fut in enumerate(as_completed(futures)):
                status, job_rows, msg = fut.result()
                if status == "reused":
                    logging.info("[%d/%d] reuse %s", i + 1, len(jobs), msg)
                    n_reused += 1
                elif status == "computed":
                    logging.info("[%d/%d] computed %s", i + 1, len(jobs), msg)
                    n_done += 1
                else:
                    logging.warning("[%d/%d] %s", i + 1, len(jobs), msg)
                    n_error += 1
                rows.extend(job_rows)

    if not rows:
        logging.warning("no stats produced")
        return

    # Skip the cross-dataset aggregate write in single-dataset mode —
    # we'd otherwise clobber the bucket-wide ``stats.csv`` with one
    # dataset's row.
    if args.zarr_path is not None:
        logging.info(
            "Single-dataset done. New stats.nc: %d. Reused: %d. Errors: %d.",
            n_done,
            n_reused,
            n_error,
        )
        if n_error:
            raise RuntimeError(f"{n_error} dataset(s) failed; see warnings above")
        return

    df = pd.DataFrame(rows)
    id_cols = [
        "source_id",
        "experiment",
        "variant_label",
        "label",
        "variable",
        "plev_index",
        "plev_pa",
        "period",
    ]
    stat_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + stat_cols]
    df = df.sort_values(
        ["source_id", "experiment", "variant_label", "variable", "plev_index", "period"]
    )

    csv_path = f"{out_dir}/stats.csv"
    parquet_path = f"{out_dir}/stats.parquet"
    df.to_csv(csv_path, index=False)
    logging.info("Wrote %s (%d rows)", csv_path, len(df))
    try:
        df.to_parquet(parquet_path, index=False)
        logging.info("Wrote %s", parquet_path)
    except ImportError:
        logging.warning("parquet engine missing; skipped %s", parquet_path)

    logging.info(
        "Done. New stats.nc: %d. Reused existing: %d. Errors: %d. "
        "Total rows in aggregate: %d.",
        n_done,
        n_reused,
        n_error,
        len(df),
    )


def _rows_from_stats_ds(stats_ds: xr.Dataset, identity: dict) -> list[dict]:
    """Reconstruct tidy aggregate rows from an existing stats.nc.

    Expects the new schema with a ``period`` coord; returns one row
    per ``(variable, plev, period)`` triple.
    """
    if "period" not in stats_ds.coords:
        # Legacy single-period stats.nc — read with implicit period=full.
        period_labels = ["full"]
    else:
        period_labels = [str(p) for p in stats_ds["period"].values]

    var_set: dict[str, list[str]] = {}
    for name in stats_ds.data_vars:
        if "__" not in name:
            continue
        var, stat = name.rsplit("__", 1)
        if stat not in _SCALAR_NAMES:
            continue
        var_set.setdefault(var, []).append(stat)

    rows: list[dict] = []
    for var in var_set:
        sample = stats_ds[f"{var}__mean"] if f"{var}__mean" in stats_ds else None
        if sample is None:
            continue
        has_plev = "plev" in sample.dims
        has_period = "period" in sample.dims
        n_plev = int(sample.sizes["plev"]) if has_plev else 0
        plev_pa = (
            stats_ds["plev"].values
            if has_plev and "plev" in stats_ds.coords
            else np.full(max(n_plev, 1), np.nan)
        )
        for pi, period_name in enumerate(period_labels):
            iter_plev = range(n_plev) if has_plev else (None,)
            for k in iter_plev:
                row: dict = {
                    "variable": var,
                    "plev_index": float(k) if k is not None else float("nan"),
                    "plev_pa": float(plev_pa[k]) if k is not None else float("nan"),
                    "period": period_name,
                }
                for stat in _SCALAR_NAMES:
                    key = f"{var}__{stat}"
                    if key not in stats_ds:
                        row[stat] = float("nan")
                        continue
                    da = stats_ds[key]
                    sel = {}
                    if has_period:
                        sel["period"] = pi
                    if has_plev and k is not None:
                        sel["plev"] = k
                    row[stat] = (
                        float(da.isel(**sel).values) if sel else float(da.values)
                    )
                row.update(identity)
                rows.append(row)
    return rows


def compute_and_write_stats(
    ds: xr.Dataset,
    stats_path: str,
    identity: dict,
    grid_name: str,
    periods=None,
) -> list[dict]:
    """Compute multi-period stats for ``ds`` and write a netCDF at
    ``stats_path``. Returns the tidy aggregate rows so callers can
    append to a per-pod log / index without re-reading.

    Used by ``process.py`` / ``process_esgf.py`` to compute stats
    inline as part of dataset processing (one stats.nc per pod, fully
    parallel via the orchestrator); the standalone ``compute_stats.py``
    main aggregates these into the cross-dataset ``stats.csv``.
    """
    n_lon = int(ds.sizes.get("lon", 90))
    w2d = area_weights_2d(grid_name, n_lon)
    stats_ds, rows = compute_dataset_stats(ds, w2d, periods=periods)
    for k, v in identity.items():
        stats_ds.attrs[k] = v
    fs, rel = fsspec.core.url_to_fs(stats_path)
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        stats_ds.to_netcdf(tmp.name)
        with fs.open(rel, "wb") as fobj:
            fobj.write(open(tmp.name, "rb").read())
    for row in rows:
        row.update(identity)
    return rows


if __name__ == "__main__":
    main()
