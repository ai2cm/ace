"""Produce pooled normalization files for the CMIP6 daily pilot.

Reads the processed zarrs and computes area-weighted global statistics
pooled across source models.  For each source_id the **first ensemble
member** (lowest ``variant_r``) is selected and both experiments
(historical + ssp585, when available) are concatenated in time before
computing per-model statistics.  Models are then averaged with equal
weight.

Outputs (into ``<output_directory>``):

- ``centering.nc`` / ``scaling.nc`` — full-field global mean and std
- ``residual_centering.nc`` / ``residual_scaling.nc`` — one-step-
  difference global mean and std
- ``time_mean_map.nc`` — per-variable time-mean spatial field
  ``(lat, lon)``, averaged across models with equal weight

All files carry global attributes describing provenance (source
models, experiments, member selection, etc.).

Usage:
    python make_normalization.py --config configs/pilot.yaml
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from compute_stats import area_weights_2d  # noqa: E402
from config import ProcessConfig  # noqa: E402

_SKIP_VARS = frozenset(
    ("below_surface_mask", "siconc_mask")
    + tuple(f"below_surface_mask{p}" for p in (1000, 850, 700, 500, 250, 100, 50, 10))
)


def _select_first_members(
    index: pd.DataFrame,
) -> dict[str, list[tuple[str, str]]]:
    """For each source_id, pick the first ensemble member and return
    its available (experiment, output_zarr) pairs.

    "First" = lowest ``variant_r``, breaking ties by ``variant_i``,
    ``variant_p``, ``variant_f``.
    """
    ok = index[index.status == "ok"].copy()
    out: dict[str, list[tuple[str, str]]] = {}
    for source_id, group in ok.groupby("source_id"):
        sorted_g = group.sort_values(
            ["variant_r", "variant_i", "variant_p", "variant_f"]
        )
        first_variant = sorted_g.iloc[0].variant_label
        member_rows = group[group.variant_label == first_variant]
        out[str(source_id)] = [
            (r.experiment, r.output_zarr) for _, r in member_rows.iterrows()
        ]
    return out


def _data_variables(ds: xr.Dataset) -> list[str]:
    return sorted(v for v in ds.data_vars if v not in _SKIP_VARS)


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    w = weights / weights.sum()
    mean = float(np.sum(values * w))
    var = float(np.sum((values - mean) ** 2 * w))
    return mean, float(np.sqrt(max(var, 0.0)))


def _stats_for_var(
    datasets: list[xr.Dataset], var: str, w2d: np.ndarray
) -> dict | None:
    """Compute stats for one variable across experiment datasets.

    Returns dict with keys: mean, std, d1_mean, d1_std, time_mean_map.
    Returns None if the variable has no valid data.
    """
    is_static = all(
        "time" not in ds[var].dims for ds in datasets if var in ds.data_vars
    )

    if is_static:
        arr = next(ds[var].values for ds in datasets if var in ds.data_vars)
        valid = np.isfinite(arr)
        if not valid.any():
            return None
        mean, std = _weighted_mean_std(arr[valid], w2d[valid])
        return {
            "mean": mean,
            "std": std,
            "d1_mean": 0.0,
            "d1_std": 0.0,
            "time_mean_map": arr.copy(),
        }

    time_arrays = []
    for ds in datasets:
        if var in ds.data_vars and "time" in ds[var].dims:
            time_arrays.append(ds[var].values)
    if not time_arrays:
        return None

    combined = np.concatenate(time_arrays, axis=0)
    valid = np.isfinite(combined)
    w3 = np.broadcast_to(w2d, combined.shape)
    if not valid.any():
        return None
    mean, std = _weighted_mean_std(combined[valid], w3[valid])

    # One-step difference — computed per-experiment to avoid a spurious
    # jump at the historical/ssp585 boundary.
    d1_parts = [np.diff(a, axis=0) for a in time_arrays]
    d1 = np.concatenate(d1_parts, axis=0)
    d1_valid = np.isfinite(d1)
    w3_d1 = np.broadcast_to(w2d, d1.shape)
    if d1_valid.any():
        d1_mean, d1_std = _weighted_mean_std(d1[d1_valid], w3_d1[d1_valid])
    else:
        d1_mean, d1_std = 0.0, 0.0

    time_mean_map = np.nanmean(combined, axis=0)
    return {
        "mean": mean,
        "std": std,
        "d1_mean": d1_mean,
        "d1_std": d1_std,
        "time_mean_map": time_mean_map,
    }


def _compute_model_stats(zarr_paths: list[str], w2d: np.ndarray) -> dict[str, dict]:
    """Load zarrs for one model and compute per-variable stats."""
    datasets = [xr.open_zarr(p, consolidated=True) for p in zarr_paths]
    all_vars: set[str] = set()
    for ds in datasets:
        all_vars |= set(_data_variables(ds))

    out: dict[str, dict] = {}
    for var in sorted(all_vars):
        result = _stats_for_var(datasets, var, w2d)
        if result is not None:
            out[var] = result
    for ds in datasets:
        ds.close()
    return out


def _pool_across_models(
    per_model: dict[str, dict[str, dict]],
) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Average per-variable stats across models with equal weight.

    Returns:
        pooled: {var: {mean, std, d1_mean, d1_std, time_mean_map}}
        contributors: {var: [source_id, ...]}
    """
    all_vars: set[str] = set()
    for stats in per_model.values():
        all_vars |= set(stats.keys())

    pooled: dict[str, dict] = {}
    contributors: dict[str, list[str]] = {}
    for var in sorted(all_vars):
        models_with_var = [
            (src, stats[var]) for src, stats in per_model.items() if var in stats
        ]
        if not models_with_var:
            continue
        n = len(models_with_var)
        contributors[var] = [src for src, _ in models_with_var]
        pooled[var] = {
            "mean": sum(s["mean"] for _, s in models_with_var) / n,
            "std": sum(s["std"] for _, s in models_with_var) / n,
            "d1_mean": sum(s["d1_mean"] for _, s in models_with_var) / n,
            "d1_std": sum(s["d1_std"] for _, s in models_with_var) / n,
            "time_mean_map": sum(s["time_mean_map"] for _, s in models_with_var) / n,
        }
    return pooled, contributors


def _global_attrs(
    source_ids: list[str],
    members: dict[str, list[tuple[str, str]]],
) -> dict[str, str]:
    return {
        "description": (
            "Pooled normalization statistics for the CMIP6 daily pilot. "
            "Per-model stats computed on the first ensemble member across "
            "all available experiments, then averaged across models with "
            "equal weight."
        ),
        "source_models": ", ".join(sorted(source_ids)),
        "n_source_models": str(len(source_ids)),
        "member_selection": "; ".join(
            f"{src}: {', '.join(exp for exp, _ in exps)}"
            for src, exps in sorted(members.items())
        ),
        "script": "make_normalization.py",
    }


def _write_scalar_nc(
    path: str,
    pooled: dict[str, dict],
    key: str,
    contributors: dict[str, list[str]],
    global_attrs: dict[str, str],
) -> None:
    data_vars = {}
    for var, stats in sorted(pooled.items()):
        da = xr.DataArray(
            stats[key],
            attrs={"contributors": ", ".join(contributors[var])},
        )
        data_vars[var] = da
    ds = xr.Dataset(data_vars, attrs=global_attrs)
    # Write to a local tempfile first: scipy's netCDF backend calls
    # seek() on flush, which fsspec's GCS write streams don't support.
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        ds.to_netcdf(tmp.name)
        with fsspec.open(path, "wb") as f:
            f.write(open(tmp.name, "rb").read())
    logging.info("Wrote %s (%d variables)", path, len(data_vars))


def _write_map_nc(
    path: str,
    pooled: dict[str, dict],
    contributors: dict[str, list[str]],
    lat: np.ndarray,
    lon: np.ndarray,
    global_attrs: dict[str, str],
) -> None:
    data_vars = {}
    for var, stats in sorted(pooled.items()):
        tmap = stats["time_mean_map"]
        if tmap.ndim != 2:
            continue
        da = xr.DataArray(
            tmap,
            dims=["lat", "lon"],
            attrs={"contributors": ", ".join(contributors[var])},
        )
        data_vars[var] = da
    coords = {"lat": lat, "lon": lon}
    ds = xr.Dataset(data_vars, coords=coords, attrs=global_attrs)
    # See _write_scalar_nc for why we use a tempfile.
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        ds.to_netcdf(tmp.name)
        with fsspec.open(path, "wb") as f:
            f.write(open(tmp.name, "rb").read())
    logging.info("Wrote %s (%d variables)", path, len(data_vars))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to the process YAML (pilot.yaml)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    out_dir = cfg.output_directory.rstrip("/")
    index_path = f"{out_dir}/index.csv"
    fs, rel = fsspec.core.url_to_fs(index_path)
    if not fs.exists(rel):
        raise FileNotFoundError(f"{index_path} not found; run process.py first.")
    idx = pd.read_csv(index_path)
    members = _select_first_members(idx)

    grid_name = cfg.defaults.target_grid.name
    sample_zarr = members[next(iter(members))][0][1]
    sample_ds = xr.open_zarr(sample_zarr, consolidated=True)
    w2d = area_weights_2d(grid_name, sample_ds.sizes["lon"])
    lat = sample_ds["lat"].values
    lon = sample_ds["lon"].values
    sample_ds.close()

    logging.info(
        "Computing normalization from %d source models: %s",
        len(members),
        ", ".join(sorted(members)),
    )

    per_model: dict[str, dict[str, dict]] = {}
    for source_id, exps in sorted(members.items()):
        zarr_paths = [p for _, p in exps]
        exp_names = [e for e, _ in exps]
        logging.info(
            "  %s (member %s, experiments %s)",
            source_id,
            idx[(idx.source_id == source_id) & (idx.output_zarr == zarr_paths[0])]
            .iloc[0]
            .variant_label,
            ", ".join(exp_names),
        )
        per_model[source_id] = _compute_model_stats(zarr_paths, w2d)

    pooled, contributors = _pool_across_models(per_model)
    attrs = _global_attrs(list(members.keys()), members)

    _write_scalar_nc(
        f"{out_dir}/centering.nc",
        pooled,
        "mean",
        contributors,
        attrs,
    )
    _write_scalar_nc(
        f"{out_dir}/scaling.nc",
        pooled,
        "std",
        contributors,
        attrs,
    )
    _write_scalar_nc(
        f"{out_dir}/residual_centering.nc",
        pooled,
        "d1_mean",
        contributors,
        attrs,
    )
    _write_scalar_nc(
        f"{out_dir}/residual_scaling.nc",
        pooled,
        "d1_std",
        contributors,
        attrs,
    )
    _write_map_nc(
        f"{out_dir}/time_mean_map.nc",
        pooled,
        contributors,
        lat,
        lon,
        attrs,
    )


if __name__ == "__main__":
    main()
