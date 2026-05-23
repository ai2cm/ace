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
import logging
import os
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

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


def _stats_for_time_field(arr: np.ndarray, w2d: np.ndarray) -> dict[str, float]:
    """Stats for a ``(time, lat, lon)`` array."""
    out = _nan_dict()
    valid_mask = np.isfinite(arr)
    out["finite_fraction"] = float(valid_mask.mean())
    if not valid_mask.any():
        return out

    val_v = arr[valid_mask]
    w3 = np.broadcast_to(w2d, arr.shape)
    val_w = w3[valid_mask]

    out["mean"], out["std"] = _weighted_mean_std(val_v, val_w)
    out["p01"] = float(np.percentile(val_v, 1))
    out["p50"] = float(np.percentile(val_v, 50))
    out["p99"] = float(np.percentile(val_v, 99))
    out["skewness"], out["kurtosis"] = _moment_stats(val_v, out["mean"], out["std"])

    # Variance decomposition: total = climatology + anomaly. The
    # climatology contribution is the spatial std of the time-mean
    # field; the anomaly is whatever's left.
    time_mean = np.nanmean(arr, axis=0)  # (lat, lon)
    finite_clim = np.isfinite(time_mean)
    if finite_clim.any() and out["std"] > 0:
        clim_w = w2d[finite_clim]
        clim_w = clim_w / clim_w.sum()
        clim_v = time_mean[finite_clim]
        clim_mean = float(np.sum(clim_v * clim_w))
        clim_var = float(np.sum((clim_v - clim_mean) ** 2 * clim_w))
        clim_std = float(np.sqrt(max(clim_var, 0.0)))
        out["clim_std"] = clim_std
        total_var = out["std"] ** 2
        # ``clim_var > total_var`` shouldn't happen analytically but
        # can by O(eps) due to weighting/precision; clamp.
        anom_var = max(total_var - clim_var, 0.0)
        out["anom_std"] = float(np.sqrt(anom_var))
        out["clim_var_frac"] = float(min(max(clim_var / total_var, 0.0), 1.0))

    # One-step finite difference in time.
    if arr.shape[0] > 1:
        d1 = np.diff(arr, axis=0)
        d1_mask = np.isfinite(d1)
        if d1_mask.any():
            d1_w = np.broadcast_to(w2d, d1.shape)[d1_mask]
            d1_v = d1[d1_mask]
            out["d1_mean"], out["d1_std"] = _weighted_mean_std(d1_v, d1_w)

    # Lag-1 autocorrelation of the area-weighted spatial mean time series.
    spatial_mean_t = np.nansum(arr * w2d, axis=(1, 2))
    sm = spatial_mean_t[np.isfinite(spatial_mean_t)]
    if sm.size > 2:
        c = sm - sm.mean()
        denom = float((c**2).sum())
        if denom > 0:
            out["autocorr_lag1"] = float((c[:-1] * c[1:]).sum() / denom)
    return out


def _stats_for_static_field(arr: np.ndarray, w2d: np.ndarray) -> dict[str, float]:
    """Stats for a ``(lat, lon)`` static field (no time, no d1)."""
    out = _nan_dict()
    valid = np.isfinite(arr)
    out["finite_fraction"] = float(valid.mean())
    if not valid.any():
        return out

    val_v = arr[valid]
    val_w = w2d[valid]

    out["mean"], out["std"] = _weighted_mean_std(val_v, val_w)
    out["p01"] = float(np.percentile(val_v, 1))
    out["p50"] = float(np.percentile(val_v, 50))
    out["p99"] = float(np.percentile(val_v, 99))
    out["skewness"], out["kurtosis"] = _moment_stats(val_v, out["mean"], out["std"])
    # d1 and autocorr left as NaN.
    return out


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
        date_type = type(times[0])

        def to_dt(s, *, end: bool):
            y, m, d = (int(x) for x in s.split("-"))
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
    * ``rows``: tidy aggregate, one dict per ``(variable, plev, period)``.
    """
    from config import DEFAULT_STATS_PERIODS

    if periods is None:
        periods = DEFAULT_STATS_PERIODS
    periods = list(periods)
    period_names = [p.name for p in periods]

    vars_to_process = [v for v in ds.data_vars if v not in _SKIP_VARS]
    plev_size = int(ds.sizes.get("plev", 0))

    nc_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    rows: list[dict] = []

    # Per-variable stats accumulator: stats_by_var[v][period_index]
    # = either a list of per-plev dicts (3D) or a single dict.
    # Pre-allocate so empty periods (no overlap) emit NaN cleanly.
    for v in vars_to_process:
        da = ds[v]
        is_3d = "time" in da.dims and "plev" in da.dims
        is_static = "time" not in da.dims

        # Build a (n_periods, [plev]) array per stat.
        per_period: list = []
        for p in periods:
            if is_static:
                # Static: identical across periods; only compute once.
                if per_period:
                    per_period.append(per_period[0])
                else:
                    per_period.append(_stats_for_static_field(da.values, w2d))
                continue
            sub = _slice_ds_to_period(ds, p)
            if sub is None:
                # No data in this period — emit NaN stats.
                if is_3d:
                    per_period.append([_nan_stats() for _ in range(plev_size)])
                else:
                    per_period.append(_nan_stats())
                continue
            sub_da = sub[v]
            if is_3d:
                per_period.append(
                    [
                        _stats_for_time_field(sub_da.isel(plev=k).values, w2d)
                        for k in range(sub_da.sizes["plev"])
                    ]
                )
            else:
                per_period.append(_stats_for_time_field(sub_da.values, w2d))

        # Materialize per-stat arrays. 3D shape: (period, plev). 2D / static
        # shape: (period,).
        for stat in _SCALAR_NAMES:
            if is_3d:
                arr = np.array(
                    [
                        [d.get(stat, float("nan")) for d in per_p]
                        for per_p in per_period
                    ],
                    dtype=float,
                )
                nc_vars[f"{v}__{stat}"] = (("period", "plev"), arr)
            else:
                arr = np.array(
                    [d.get(stat, float("nan")) for d in per_period], dtype=float
                )
                nc_vars[f"{v}__{stat}"] = (("period",), arr)

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
                    row.update(per_period[pi][k])
                    rows.append(row)
            else:
                row = {
                    "variable": v,
                    "plev_index": float("nan"),
                    "plev_pa": float("nan"),
                    "period": period_name,
                }
                row.update(per_period[pi])
                rows.append(row)

    coords: dict[str, xr.DataArray] = {
        "period": xr.DataArray(np.array(period_names, dtype=object), dims=("period",)),
    }
    if "plev" in ds.coords:
        coords["plev"] = ds.plev
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    out_dir = cfg.output_directory.rstrip("/")
    index_path = f"{out_dir}/index.csv"
    fs, rel = fsspec.core.url_to_fs(index_path)
    if not fs.exists(rel):
        raise FileNotFoundError(f"{index_path} not found; run process.py first")
    idx = pd.read_csv(index_path)

    ok = idx[idx.status == "ok"].reset_index(drop=True)
    if args.source_ids is not None:
        ok = ok[ok.source_id.isin(args.source_ids)].reset_index(drop=True)
    logging.info("Found %d ok datasets in the index.", len(ok))

    grid_name = cfg.defaults.target_grid.name
    periods = tuple(cfg.defaults.stats_periods)
    logging.info(
        "Using %d stats period(s): %s",
        len(periods),
        ", ".join(p.name for p in periods),
    )
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(ok))) if len(ok) else 1
    logging.info("Computing stats with %d worker(s).", workers)

    jobs = []
    for _, r in ok.iterrows():
        zarr_path = r.output_zarr
        stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
        identity = {
            "source_id": r.source_id,
            "experiment": r.experiment,
            "variant_label": r.variant_label,
            "label": r.get("label", ""),
        }
        jobs.append((zarr_path, stats_path, identity, grid_name, args.force, periods))

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
