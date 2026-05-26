"""Produce pooled + per-source normalization files for the CMIP6 daily pilot.

Reads the processed zarrs and computes area-weighted global statistics
both pooled across source models (cross-source) **and** per source.
For each source_id the **first ensemble member** (lowest ``variant_r``)
is selected and its available experiments are concatenated in time
before computing per-model statistics. Models are then averaged with
equal weight to form the cross-source pooled stats.

The ``--period`` argument selects one of the configured
``stats_periods`` (default ``full``). The chosen period's time window
is applied to every dataset's time axis before computing stats — so
``--period 1940-2014`` produces stats over the historical era only,
and pure-SSP datasets contribute nothing (no overlap → skipped).

Outputs (into ``<output_directory>``):

- ``normalization_{period}/centering.nc`` / ``scaling.nc`` —
  cross-source pooled global mean / std.
- ``normalization_{period}/residual_centering.nc`` /
  ``residual_scaling.nc`` — cross-source pooled one-step-difference
  mean / std.
- ``normalization_{period}/time_mean_map.nc`` — cross-source pooled
  time-mean spatial field per variable.
- ``per_source_normalization_{period}/<source_id>/centering.nc`` /
  ``scaling.nc`` / ``residual_centering.nc`` / ``residual_scaling.nc``
  — per-source variants, one set per source. Consumed by
  ``fme.core.per_source_normalizer.PerSourceNormalizationConfig`` at
  training time via the matching ``subdirectory:`` setting.

All files carry global attributes describing provenance (source
models, experiments, member selection, period window).

Usage:
    python make_normalization.py --config configs/pilot.yaml \\
        [--period full | 1940-2014 | 1979-2014]
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
from config import SURFACE_AND_OCEAN_VARIABLES, ProcessConfig, StatsPeriod  # noqa: E402
from processing import clip_date_for_calendar  # noqa: E402

# Mask variables produced by the pipeline — excluded from normalization
# stats, since their semantics (0/1 valid-cell) shouldn't be standardised.
# Includes the per-plev below-surface masks plus one ``{name}_mask`` per
# ocean/sea-ice surface-and-ocean variable (atmos_surface variables emit
# no mask channel — see ``finalize_surface_and_ocean_variable``).
_SKIP_VARS = frozenset(
    ("below_surface_mask",)
    + tuple(f"below_surface_mask{p}" for p in (1000, 850, 700, 500, 250, 100, 50, 10))
    + tuple(
        f"{h.output_name}_mask"
        for h in SURFACE_AND_OCEAN_VARIABLES
        if h.kind in ("ocean_surface", "seaice_surface")
    )
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


def _slice_to_period(ds: xr.Dataset, period: StatsPeriod) -> xr.Dataset:
    """Return ``ds`` restricted to ``period``'s ``[start, end]`` window.

    Unbounded ends (``None``) leave that side open. Calendar-aware: a
    360-day calendar's invalid ``YYYY-MM-31`` end-of-month strings are
    clipped to day 30. Returns an empty (zero-time) Dataset when there
    is no overlap.
    """
    if "time" not in ds.dims:
        return ds
    if period.start is None and period.end is None:
        return ds
    try:
        calendar = str(ds["time"].dt.calendar)
    except (AttributeError, TypeError):
        calendar = "standard"
    start = clip_date_for_calendar(period.start, calendar) if period.start else None
    end = clip_date_for_calendar(period.end, calendar) if period.end else None
    return ds.sel(time=slice(start, end))


def _compute_model_stats(
    zarr_paths: list[str],
    w2d: np.ndarray,
    period: StatsPeriod,
) -> dict[str, dict]:
    """Load zarrs for one model, slice to ``period``, and compute stats.

    Datasets with no overlap on ``period`` (e.g. an ssp585 zarr when
    ``period.start, period.end = 1940-01-01, 2014-12-31``) get an empty
    time axis and are silently dropped from the contributing set.
    """
    datasets: list[xr.Dataset] = []
    for p in zarr_paths:
        ds = xr.open_zarr(p, consolidated=True)
        sliced = _slice_to_period(ds, period)
        if "time" in sliced.dims and sliced.sizes.get("time", 0) == 0:
            ds.close()
            continue
        datasets.append(sliced)

    if not datasets:
        return {}

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
    period: StatsPeriod,
    scope: str,
) -> dict[str, str]:
    if scope == "pooled":
        desc = (
            "Cross-source pooled normalization statistics for the CMIP6 "
            "daily pilot. Per-model stats computed on the first ensemble "
            "member across available experiments, then averaged across "
            "models with equal weight."
        )
    elif scope == "per_source":
        desc = (
            "Per-source normalization statistics for the CMIP6 daily pilot. "
            "First ensemble member, all available experiments concatenated "
            "in time, restricted to the configured period window."
        )
    else:
        raise ValueError(f"unexpected scope {scope!r}")
    return {
        "description": desc,
        "scope": scope,
        "period": period.name,
        "period_start": period.start or "unbounded",
        "period_end": period.end or "unbounded",
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


def _resolve_period(cfg: ProcessConfig, name: str) -> StatsPeriod:
    """Look up a configured ``StatsPeriod`` by name."""
    for p in cfg.defaults.stats_periods:
        if p.name == name:
            return p
    available = [p.name for p in cfg.defaults.stats_periods]
    raise ValueError(
        f"unknown period {name!r}; available periods in config: {available}"
    )


def _write_per_source(
    base_dir: str,
    source_id: str,
    stats: dict[str, dict],
    attrs: dict[str, str],
) -> None:
    """Write a single source's stats as the same four scalar nc files
    used for the cross-source pooled output. The per-source files live
    at ``<base_dir>/<source_id>/{centering,scaling,residual_*}.nc`` —
    the directory layout the training-side
    ``PerSourceNormalizationConfig`` expects.

    No ``time_mean_map.nc`` per source — pooled is the only useful
    map (per-source map is just that source's time-mean, which
    consumers can load from the zarr directly).
    """
    contributors_one = {var: [source_id] for var in stats}
    _write_scalar_nc(
        f"{base_dir}/{source_id}/centering.nc",
        stats,
        "mean",
        contributors_one,
        attrs,
    )
    _write_scalar_nc(
        f"{base_dir}/{source_id}/scaling.nc",
        stats,
        "std",
        contributors_one,
        attrs,
    )
    _write_scalar_nc(
        f"{base_dir}/{source_id}/residual_centering.nc",
        stats,
        "d1_mean",
        contributors_one,
        attrs,
    )
    _write_scalar_nc(
        f"{base_dir}/{source_id}/residual_scaling.nc",
        stats,
        "d1_std",
        contributors_one,
        attrs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to the process YAML (pilot.yaml)"
    )
    parser.add_argument(
        "--period",
        default="full",
        help=(
            "Name of a ``StatsPeriod`` from ``defaults.stats_periods`` "
            "in the config (default: ``full``). Output files land in "
            "``{output_directory}/normalization_{period}/`` (cross-source "
            "pooled) and ``{output_directory}/per_source_normalization_"
            "{period}/<source_id>/`` (per-source)."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    period = _resolve_period(cfg, args.period)
    logging.info(
        "Computing normalization for period %r [%s, %s]",
        period.name,
        period.start or "unbounded",
        period.end or "unbounded",
    )

    out_dir = cfg.output_directory.rstrip("/")
    pooled_dir = f"{out_dir}/normalization_{period.name}"
    per_source_dir = f"{out_dir}/per_source_normalization_{period.name}"

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
        stats = _compute_model_stats(zarr_paths, w2d, period)
        if stats:
            per_model[source_id] = stats
        else:
            logging.info(
                "    %s: no data in [%s, %s] for period %r; skipping",
                source_id,
                period.start or "-inf",
                period.end or "+inf",
                period.name,
            )

    if not per_model:
        raise RuntimeError(
            f"No source models contributed any data for period {period.name!r}; "
            f"check the time_subset overlap with [{period.start}, {period.end}]"
        )

    # --- Cross-source pooled outputs ---
    pooled, contributors = _pool_across_models(per_model)
    pooled_attrs = _global_attrs(list(per_model.keys()), members, period, "pooled")
    _write_scalar_nc(
        f"{pooled_dir}/centering.nc", pooled, "mean", contributors, pooled_attrs
    )
    _write_scalar_nc(
        f"{pooled_dir}/scaling.nc", pooled, "std", contributors, pooled_attrs
    )
    _write_scalar_nc(
        f"{pooled_dir}/residual_centering.nc",
        pooled,
        "d1_mean",
        contributors,
        pooled_attrs,
    )
    _write_scalar_nc(
        f"{pooled_dir}/residual_scaling.nc",
        pooled,
        "d1_std",
        contributors,
        pooled_attrs,
    )
    _write_map_nc(
        f"{pooled_dir}/time_mean_map.nc",
        pooled,
        contributors,
        lat,
        lon,
        pooled_attrs,
    )

    # --- Per-source outputs ---
    for source_id, stats in sorted(per_model.items()):
        per_src_attrs = _global_attrs(
            [source_id], {source_id: members[source_id]}, period, "per_source"
        )
        _write_per_source(per_source_dir, source_id, stats, per_src_attrs)
    logging.info(
        "Wrote per-source stats for %d sources to %s/",
        len(per_model),
        per_source_dir,
    )


if __name__ == "__main__":
    main()
