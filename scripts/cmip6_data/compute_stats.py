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
import sys
from pathlib import Path

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


def compute_dataset_stats(
    ds: xr.Dataset,
    w2d: np.ndarray,
) -> tuple[xr.Dataset, list[dict[str, float]]]:
    """Compute stats for every data variable in ``ds``.

    Returns:

    * ``stats_ds`` — an ``xarray.Dataset`` with one variable per
      ``<varname>__<stat>`` pair. 3D variables get a ``plev`` dim.
      Suitable for writing to netCDF as the per-dataset stats file.
    * ``rows`` — a list of dicts (one per ``(variable, plev)``) for
      the cross-dataset aggregate.
    """
    nc_vars: dict[str, tuple[tuple[str, ...], np.ndarray | float]] = {}
    rows: list[dict[str, float]] = []
    for v in ds.data_vars:
        if v in _SKIP_VARS:
            continue
        da = ds[v]
        if "time" in da.dims and "plev" in da.dims:
            n_plev = da.sizes["plev"]
            per_plev = []
            for k in range(n_plev):
                arr = da.isel(plev=k).values
                per_plev.append(_stats_for_time_field(arr, w2d))
            for stat in _SCALAR_NAMES:
                vals = np.array(
                    [s.get(stat, float("nan")) for s in per_plev], dtype=float
                )
                nc_vars[f"{v}__{stat}"] = (("plev",), vals)
            for k, s in enumerate(per_plev):
                row = {
                    "variable": v,
                    "plev_index": float(k),
                    "plev_pa": float(da.plev.values[k]),
                }
                row.update(s)
                rows.append(row)
        elif "time" in da.dims:
            arr = da.values
            stats = _stats_for_time_field(arr, w2d)
            for stat in _SCALAR_NAMES:
                nc_vars[f"{v}__{stat}"] = ((), float(stats[stat]))
            row = {"variable": v, "plev_index": float("nan"), "plev_pa": float("nan")}
            row.update(stats)
            rows.append(row)
        else:
            arr = da.values
            stats = _stats_for_static_field(arr, w2d)
            for stat in _SCALAR_NAMES:
                nc_vars[f"{v}__{stat}"] = ((), float(stats[stat]))
            row = {"variable": v, "plev_index": float("nan"), "plev_pa": float("nan")}
            row.update(stats)
            rows.append(row)

    coords: dict[str, xr.DataArray] = {}
    if "plev" in ds.coords:
        coords["plev"] = ds.plev
    data = {
        name: xr.DataArray(np.asarray(payload), dims=dims)
        for name, (dims, payload) in nc_vars.items()
    }
    stats_ds = xr.Dataset(data, coords=coords)
    return stats_ds, rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    out_dir = cfg.output_directory.rstrip("/")
    index_path = f"{out_dir}/index.csv"
    if not Path(index_path).exists():
        raise FileNotFoundError(f"{index_path} not found; run process.py first")
    idx = pd.read_csv(index_path)

    ok = idx[idx.status == "ok"].reset_index(drop=True)
    if args.source_ids is not None:
        ok = ok[ok.source_id.isin(args.source_ids)].reset_index(drop=True)
    logging.info("Found %d ok datasets in the index.", len(ok))

    grid_name = cfg.defaults.target_grid.name
    rows: list[dict] = []
    n_done = 0
    n_reused = 0

    for i, r in ok.iterrows():
        zarr_path = r.output_zarr
        stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
        identity = {
            "source_id": r.source_id,
            "experiment": r.experiment,
            "variant_label": r.variant_label,
            "label": r.get("label", ""),
        }

        fs, rel = fsspec.core.url_to_fs(stats_path)
        if fs.exists(rel) and not args.force:
            logging.info("[%d/%d] reuse %s", i + 1, len(ok), stats_path)
            with fs.open(rel, "rb") as fobj:
                stats_ds = xr.open_dataset(fobj)
                rows.extend(_rows_from_stats_ds(stats_ds, identity))
                stats_ds.close()
            n_reused += 1
            continue

        try:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        except Exception as e:  # noqa: BLE001
            logging.warning("could not open %s: %s", zarr_path, e)
            continue
        n_lon = ds.sizes.get("lon", 90)
        w2d = area_weights_2d(grid_name, n_lon)

        logging.info("[%d/%d] computing %s", i + 1, len(ok), zarr_path)
        stats_ds, dataset_rows = compute_dataset_stats(ds, w2d)
        for k, val in identity.items():
            stats_ds.attrs[k] = val
        with fs.open(rel, "wb") as fobj:
            stats_ds.to_netcdf(fobj)
        ds.close()

        for row in dataset_rows:
            row.update(identity)
        rows.extend(dataset_rows)
        n_done += 1

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
    ]
    stat_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + stat_cols]
    df = df.sort_values(
        ["source_id", "experiment", "variant_label", "variable", "plev_index"]
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
        "Done. New stats.nc: %d. Reused existing: %d. Total rows in aggregate: %d.",
        n_done,
        n_reused,
        len(df),
    )


def _rows_from_stats_ds(
    stats_ds: xr.Dataset, identity: dict[str, str]
) -> list[dict[str, float]]:
    """Reconstruct tidy aggregate rows from an existing stats.nc."""
    var_set: dict[str, list[str]] = {}
    for name in stats_ds.data_vars:
        if "__" not in name:
            continue
        var, stat = name.rsplit("__", 1)
        var_set.setdefault(var, []).append(stat)
    rows: list[dict[str, float]] = []
    for var, stats in var_set.items():
        sample = stats_ds[f"{var}__{stats[0]}"]
        if sample.dims == ():
            row = {"variable": var, "plev_index": float("nan"), "plev_pa": float("nan")}
            for stat in _SCALAR_NAMES:
                key = f"{var}__{stat}"
                row[stat] = (
                    float(stats_ds[key].values) if key in stats_ds else float("nan")
                )
            row.update(identity)
            rows.append(row)
        else:
            n_plev = sample.sizes["plev"]
            plev_pa = (
                stats_ds["plev"].values
                if "plev" in stats_ds.coords
                else np.full(n_plev, np.nan)
            )
            for k in range(n_plev):
                row = {
                    "variable": var,
                    "plev_index": float(k),
                    "plev_pa": float(plev_pa[k]),
                }
                for stat in _SCALAR_NAMES:
                    key = f"{var}__{stat}"
                    if key in stats_ds:
                        row[stat] = float(stats_ds[key].isel(plev=k).values)
                    else:
                        row[stat] = float("nan")
                row.update(identity)
                rows.append(row)
    return rows


if __name__ == "__main__":
    main()
