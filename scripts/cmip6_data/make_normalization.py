"""Produce pooled + per-source normalization files for the CMIP6 daily pilot.

Aggregates per-dataset ``stats.nc`` files (one per processed zarr,
written by ``compute_and_write_stats``) into:

- ``normalization_{period}/centering.nc`` / ``scaling.nc`` —
  cross-source pooled global mean / std.
- ``normalization_{period}/residual_centering.nc`` /
  ``residual_scaling.nc`` — cross-source pooled one-step-difference
  mean / std.
- ``normalization_{period}/time_mean_map.nc`` — cross-source pooled
  time-mean spatial field per variable.
- ``per_source_normalization_{period}/<source_id>/centering.nc`` /
  ``scaling.nc`` / ``residual_centering.nc`` / ``residual_scaling.nc``
  — per-source variants. Consumed by
  ``fme.core.per_source_normalizer.PerSourceNormalizationConfig`` at
  training time via the matching ``subdirectory:`` setting.

Per source the **first ensemble member** (lowest ``variant_r``) is
picked; its available experiments are pooled in time using the
per-experiment scalar mean/std and per-cell ``n_valid_map`` so the
result matches concatenating-then-computing exactly (up to the missing
boundary in d1, which we accept). Sources are then averaged with equal
weight.

The ``--period`` argument selects one of the configured
``stats_periods`` (default ``full``). Each ``stats.nc`` already carries
one entry per configured period along a ``period`` coord; this script
just selects the matching slice. No zarr opens, no time-axis slicing.

Outputs land in ``<output_directory>``. All files carry global
attributes describing provenance (source models, experiments, member
selection, period window).

Usage:
    python make_normalization.py --config configs/pilot.yaml \\
        [--period full | 1940-2014 | 1979-2014]
"""

import argparse
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from compute_stats import _SCALAR_NAMES  # noqa: E402
from config import SURFACE_AND_OCEAN_VARIABLES, ProcessConfig, StatsPeriod  # noqa: E402

# Mask variables produced by the pipeline — excluded from the
# aggregation pass, since their semantics (0/1 valid-cell) make
# standardised mean/std nonsensical. Trivial entries (mean=0, std=1)
# are injected after aggregation via :func:`_inject_trivial_norm`;
# see ``_TRIVIAL_NORM_VARS`` below.
_SKIP_VARS = frozenset(
    ("below_surface_mask",)
    + tuple(f"below_surface_mask{p}" for p in (1000, 850, 700, 500, 250, 100, 50, 10))
    + tuple(
        f"{h.output_name}_mask"
        for h in SURFACE_AND_OCEAN_VARIABLES
        if h.kind in ("ocean_surface", "seaice_surface")
    )
)

# Variables that get **trivial** normalization (mean=0, std=1) rather
# than aggregated stats. Two families fall in here, sharing the same
# rationale — for fields whose natural domain is [0, 1], the network
# is better off seeing the raw value than a (val - 0.3) / 0.5
# standardisation that scrambles the semantics:
#
# - Per-cell binary masks (``below_surface_mask*``, ``oday_*_mask``,
#   ``siday_*_mask``) — 0/1 indicators of "this variable is defined
#   at this cell". Already in ``_SKIP_VARS`` so the aggregator never
#   tries to compute real stats for them.
# - ``land_fraction`` (CMIP6 sftlf) — a static [0, 1] field that the
#   model uses to know where land is. Real aggregated stats (mean
#   ~0.3, std ~0.45) would standardise away the 0/1 interpretation;
#   trivial norm preserves the raw value the network expects.
# - ``luh2_forest`` — a static [0, 1] field carrying the LUH2 forest
#   fraction. Same rationale as ``land_fraction``: real aggregated
#   stats (mean ~0.07, std ~0.23) would standardise away the
#   semantic 0/1 interpretation.
#
# Injection happens post-aggregation so any real stats for
# ``land_fraction`` / ``luh2_forest`` (neither of which is in
# _SKIP_VARS) get *overridden* — making the convention authoritative
# regardless of upstream stats.
_TRIVIAL_NORM_VARS = frozenset(("land_fraction", "luh2_forest") + tuple(_SKIP_VARS))


def _inject_trivial_norm(stats: dict[str, dict]) -> None:
    """Override / insert mean=0, std=1, d1=0/1 entries for
    ``_TRIVIAL_NORM_VARS`` in-place.

    Used in both cohort and per-source paths so that every centering /
    scaling / residual file carries entries for the trivial-norm
    names — important because ``PerSourceNormalizationConfig.build``
    iterates over the model's full ``in_names`` and raises on
    missing keys (no implicit pass-through).
    """
    for var in _TRIVIAL_NORM_VARS:
        stats[var] = {
            "mean": 0.0,
            "std": 1.0,
            "d1_mean": 0.0,
            "d1_std": 1.0,
            "time_mean_map": None,
        }


@dataclass
class _ExperimentStats:
    """The slice of one stats.nc relevant to one (period, var)."""

    mean: float
    std: float
    d1_mean: float
    d1_std: float
    # Per-cell maps. ``time_mean_map`` / ``n_valid_map`` have shape
    # ``(lat, lon)`` for 2D vars and ``(plev, lat, lon)`` for 3D vars;
    # ``None`` for statics (use ``static_map`` instead).
    time_mean_map: Optional[np.ndarray]
    n_valid_map: Optional[np.ndarray]
    # For static vars only. Same shape as the static field.
    static_map: Optional[np.ndarray]
    # True if this is a static (no time) variable.
    is_static: bool


def _select_first_members(
    index: pd.DataFrame,
) -> dict[str, list[tuple[str, str]]]:
    """For each source_id, pick the first ensemble member and return
    its available (experiment, stats_nc_path) pairs.

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
            (r.experiment, _stats_path_for(r.output_zarr))
            for _, r in member_rows.iterrows()
        ]
    return out


def _stats_path_for(zarr_path: str) -> str:
    return zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"


def _open_stats(stats_path: str) -> xr.Dataset:
    """Open ``stats.nc`` (possibly on GCS) via a local tempfile.

    h5netcdf needs a real file handle; fsspec's GCS stream isn't
    seekable.
    """
    fs, rel = fsspec.core.url_to_fs(stats_path)
    if not fs.exists(rel):
        raise FileNotFoundError(stats_path)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        local = tmp.name
    fs.get(rel, local)
    try:
        return xr.open_dataset(local).load()
    finally:
        Path(local).unlink(missing_ok=True)


def _extract_var(
    stats: xr.Dataset, var: str, period_idx: Optional[int]
) -> Optional[_ExperimentStats]:
    """Pull all the per-variable, per-period quantities we need.

    Returns ``None`` when ``stats`` doesn't carry this variable (a
    given source may not have every variable; we silently skip).
    """
    mean_key = f"{var}__mean"
    if mean_key not in stats:
        return None
    # ``is_static`` can't be inferred from the scalar's dims —
    # compute_stats writes both static and time-varying scalars with a
    # ``period`` axis. Use the presence of ``{v}__static_map`` (only
    # emitted for statics) as the discriminator instead.
    is_static = f"{var}__static_map" in stats

    def _scalar(name: str) -> float:
        da = stats.get(f"{var}__{name}")
        if da is None:
            return float("nan")
        if "period" in da.dims and period_idx is not None:
            return float(da.isel(period=period_idx).values)
        return float(da.values)

    mean = _scalar("mean")
    std = _scalar("std")
    d1_mean = _scalar("d1_mean")
    d1_std = _scalar("d1_std")

    time_mean_map: Optional[np.ndarray] = None
    n_valid_map: Optional[np.ndarray] = None
    static_map: Optional[np.ndarray] = None

    if is_static:
        sm_key = f"{var}__static_map"
        if sm_key in stats:
            static_map = np.asarray(stats[sm_key].values, dtype=np.float32)
    else:
        tm_key = f"{var}__time_mean_map"
        nv_key = f"{var}__n_valid_map"
        if tm_key in stats and period_idx is not None:
            time_mean_map = np.asarray(
                stats[tm_key].isel(period=period_idx).values, dtype=np.float32
            )
        if nv_key in stats and period_idx is not None:
            n_valid_map = np.asarray(stats[nv_key].isel(period=period_idx).values)

    # Treat NaN d1 stats (statics, single-sample periods) as 0/0 so
    # they contribute neutrally to the pool. Matches the legacy
    # behavior of ``_stats_for_var`` returning 0.0/0.0 for those.
    if np.isnan(d1_mean):
        d1_mean = 0.0
    if np.isnan(d1_std):
        d1_std = 0.0

    return _ExperimentStats(
        mean=mean,
        std=std,
        d1_mean=d1_mean,
        d1_std=d1_std,
        time_mean_map=time_mean_map,
        n_valid_map=n_valid_map,
        static_map=static_map,
        is_static=is_static,
    )


def _period_index(stats: xr.Dataset, period_name: str) -> Optional[int]:
    """Return the index of ``period_name`` in the stats.nc ``period``
    coord, or ``None`` if the stats has no ``period`` dim (legacy
    single-period file)."""
    if "period" not in stats.coords:
        return None
    labels = [str(p) for p in stats["period"].values]
    if period_name not in labels:
        raise ValueError(
            f"period {period_name!r} not found in stats.nc "
            f"({labels}) — re-run compute_stats with the matching "
            "period configured."
        )
    return labels.index(period_name)


def _pool_scalar(
    means: list[float], stds: list[float], weights: list[float]
) -> tuple[float, float]:
    """Sample-weighted pool of scalar mean / std across experiments.

    Uses the law of total variance:
        pooled_var = E[var] + Var[E]
    where the expectation is over experiments weighted by ``weights``.
    All-zero weights collapse to an unweighted average — happens only
    for periods with no overlap, where we expect NaNs all the way.
    """
    ws = np.asarray(weights, dtype=float)
    finite = np.isfinite(np.asarray(means)) & np.isfinite(np.asarray(stds)) & (ws > 0)
    if not finite.any():
        return float("nan"), float("nan")
    m = np.asarray(means)[finite]
    s = np.asarray(stds)[finite]
    w = ws[finite] / ws[finite].sum()
    pooled_mean = float(np.sum(m * w))
    pooled_var = float(np.sum(s**2 * w) + np.sum((m - pooled_mean) ** 2 * w))
    return pooled_mean, float(np.sqrt(max(pooled_var, 0.0)))


def _pool_per_cell_mean(
    maps: list[np.ndarray], n_valids: list[np.ndarray]
) -> np.ndarray:
    """Per-cell sample-weighted average of time-mean maps.

    ``maps[i]`` is the experiment-i time mean (NaN where no samples);
    ``n_valids[i]`` is the per-cell valid count for the same experiment.
    Cells where every experiment has zero valid samples stay NaN.
    """
    stacked_m = np.stack(maps, axis=0).astype(np.float64)
    stacked_n = np.stack(n_valids, axis=0).astype(np.float64)
    # Treat NaN means as not contributing: zero out both the value and
    # its weight. (Per-cell ``time_mean`` is NaN exactly when n_valid
    # is 0 for that cell, but be defensive.)
    valid = np.isfinite(stacked_m)
    stacked_m = np.where(valid, stacked_m, 0.0)
    stacked_n = np.where(valid, stacked_n, 0.0)

    num = np.sum(stacked_m * stacked_n, axis=0)
    den = np.sum(stacked_n, axis=0)
    out = np.full_like(num, np.nan, dtype=np.float64)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out.astype(np.float32)


def _aggregate_one_source(
    stats_files: list[xr.Dataset], period_name: str
) -> dict[str, dict]:
    """Pool one source's experiments into a single per-variable stats dict.

    Returns ``{var: {mean, std, d1_mean, d1_std, time_mean_map}}``.
    ``time_mean_map`` is the static field for statics and the pooled
    per-cell time-mean otherwise.
    """
    # Resolve the period index once per file (may differ if legacy).
    period_indices = [_period_index(s, period_name) for s in stats_files]

    # Discover variables across all files (different experiments may
    # carry different variables when an augment landed only one).
    all_vars: set[str] = set()
    for s in stats_files:
        for name in s.data_vars:
            if "__" not in name:
                continue
            base, suffix = name.rsplit("__", 1)
            if suffix in _SCALAR_NAMES:
                all_vars.add(base)
    all_vars -= _SKIP_VARS

    out: dict[str, dict] = {}
    for var in sorted(all_vars):
        extracted = []
        for s, pidx in zip(stats_files, period_indices, strict=True):
            ex = _extract_var(s, var, pidx)
            if ex is None:
                continue
            extracted.append(ex)
        if not extracted:
            continue

        # Static vars: identical across experiments of the same model.
        # Take the first non-NaN representation and call it done.
        if extracted[0].is_static:
            first = extracted[0]
            out[var] = {
                "mean": first.mean,
                "std": first.std,
                "d1_mean": 0.0,
                "d1_std": 0.0,
                "time_mean_map": first.static_map,
            }
            continue

        # Time-varying: weight experiments by total valid sample count.
        # ``n_valid_map.sum()`` gives an exact area-agnostic sample
        # count — for the scalar pool this is the same weight a
        # concatenate-then-mean operation would implicitly use, up to
        # the area weights being uniform across experiments (they are,
        # since every experiment of a given model lives on the same
        # target grid).
        weights = [
            float(ex.n_valid_map.sum()) if ex.n_valid_map is not None else 0.0
            for ex in extracted
        ]
        means = [ex.mean for ex in extracted]
        stds = [ex.std for ex in extracted]
        pooled_mean, pooled_std = _pool_scalar(means, stds, weights)

        d1_means = [ex.d1_mean for ex in extracted]
        d1_stds = [ex.d1_std for ex in extracted]
        # d1 uses (n_time - 1) samples per cell, but since we don't
        # store that map separately and the offset is tiny vs the time
        # length, use the same weights. The error is O(1/n_time).
        pooled_d1_mean, pooled_d1_std = _pool_scalar(d1_means, d1_stds, weights)

        # Per-cell time-mean pool.
        if all(
            ex.time_mean_map is not None and ex.n_valid_map is not None
            for ex in extracted
        ):
            tmap = _pool_per_cell_mean(
                [ex.time_mean_map for ex in extracted],  # type: ignore[list-item]
                [ex.n_valid_map for ex in extracted],  # type: ignore[list-item]
            )
        else:
            tmap = None

        out[var] = {
            "mean": pooled_mean,
            "std": pooled_std,
            "d1_mean": pooled_d1_mean,
            "d1_std": pooled_d1_std,
            "time_mean_map": tmap,
        }
    return out


def _pool_across_models(
    per_model: dict[str, dict[str, dict]],
) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Average per-variable stats across models with equal weight.

    Mirrors the legacy behavior: every contributing model contributes
    equally, regardless of its size or experiment count.
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
        }
        # Per-cell map averaging: include only models where the map is
        # present. Pure-static vars without a map drop out cleanly.
        maps = [
            s["time_mean_map"]
            for _, s in models_with_var
            if s.get("time_mean_map") is not None
        ]
        if maps:
            pooled[var]["time_mean_map"] = np.mean(np.stack(maps, axis=0), axis=0)
        else:
            pooled[var]["time_mean_map"] = None
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
            "models with equal weight. Aggregated from per-dataset "
            "stats.nc files (no zarr re-scan)."
        )
    elif scope == "per_source":
        desc = (
            "Per-source normalization statistics for the CMIP6 daily pilot. "
            "First ensemble member, all available experiments pooled by "
            "sample weight, restricted to the configured period window."
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
        tmap = stats.get("time_mean_map")
        if tmap is None or tmap.ndim != 2:
            continue
        da = xr.DataArray(
            tmap,
            dims=["lat", "lon"],
            attrs={"contributors": ", ".join(contributors[var])},
        )
        data_vars[var] = da
    coords = {"lat": lat, "lon": lon}
    ds = xr.Dataset(data_vars, coords=coords, attrs=global_attrs)
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

    logging.info(
        "Computing normalization from %d source models: %s",
        len(members),
        ", ".join(sorted(members)),
    )

    per_model: dict[str, dict[str, dict]] = {}
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    for source_id, exps in sorted(members.items()):
        logging.info(
            "  %s (experiments %s)",
            source_id,
            ", ".join(e for e, _ in exps),
        )
        stats_files: list[xr.Dataset] = []
        for exp_name, stats_path in exps:
            try:
                ds = _open_stats(stats_path)
            except FileNotFoundError:
                logging.warning(
                    "    %s/%s: missing %s — skipping (run migrate.py to "
                    "regenerate or recompute_stats)",
                    source_id,
                    exp_name,
                    stats_path,
                )
                continue
            stats_files.append(ds)
            if lat is None and "lat" in ds.coords:
                lat = np.asarray(ds["lat"].values)
                lon = np.asarray(ds["lon"].values)
        if not stats_files:
            logging.info("    %s: no stats files available; skipping", source_id)
            continue
        stats = _aggregate_one_source(stats_files, period.name)
        for ds in stats_files:
            ds.close()
        if stats:
            per_model[source_id] = stats
        else:
            logging.info(
                "    %s: no variables passed the period %r filter",
                source_id,
                period.name,
            )

    if not per_model:
        raise RuntimeError(
            f"No source models contributed any data for period {period.name!r}; "
            f"check that stats.nc files exist and that period names line up."
        )
    if lat is None or lon is None:
        raise RuntimeError(
            "No stats.nc carried lat/lon coords — regenerate with the "
            "0.3.0→0.4.0 migration (which writes them) before aggregating."
        )

    # --- Cross-source pooled outputs ---
    pooled, contributors = _pool_across_models(per_model)
    _inject_trivial_norm(pooled)
    # ``contributors`` keyed by every var in ``pooled``; trivial-norm
    # entries get a sentinel marker so the downstream nc carries a
    # discoverable hint that the stats came from the convention,
    # not from data aggregation.
    for var in _TRIVIAL_NORM_VARS:
        contributors[var] = ["(convention: mean=0, std=1)"]
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
        # Inject trivial-norm entries here too so per-source lookup
        # for masks + land_fraction always succeeds. Without this,
        # PerSourceNormalizer.build would raise on any source that
        # doesn't publish (e.g.) ``oday_tos_mask`` in its data.
        _inject_trivial_norm(stats)
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
