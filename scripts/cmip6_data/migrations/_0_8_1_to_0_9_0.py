"""Migration 0.8.1 → 0.9.0: derive ``total_water_path`` + flag the
mask/sftlf trivial-norm convention.

Two conventions advance to 0.9.0:

1. **``total_water_path`` backfill.** The ESGF augment in 0.8.1's
   ``process_esgf.py`` only derived ``total_water_path = water_vapor_path
   + clwvi`` when ``water_vapor_path`` was pre-existing in the zarr;
   when both inputs landed in the same augment pass (the common v2
   case) the derivation was silently skipped. The augment-side bug is
   fixed in process_esgf.py; this migration backfills the 22 v2
   datasets that need it.

   For each eligible dataset (both inputs present, ``total_water_path``
   absent), the migration:

   * reads only the two input variables, computes the sum, writes
     ``total_water_path`` to the zarr via ``to_zarr(mode="a")``, and
     re-consolidates metadata;
   * computes stats *only* for the new ``total_water_path`` variable
     (subsetting the dataset before calling
     ``compute_dataset_stats``) and merges them into the existing
     per-dataset ``stats.nc``. The ~80 other variables' stats are
     read-through and re-written without recomputation.

   ``total_water_path`` is a real continuous physical field
   (kg/m²); its normalization needs real mean/std, not the trivial
   (0, 1) we use for masks. After this migration the per-dataset
   ``stats.nc`` files carry real ``total_water_path__*`` entries, so
   the next ``make_normalization.py`` aggregation pass picks them
   up in both cohort and per-source centering files.

2. **Trivial normalization for masks + land_fraction.** The cohort
   + per-source ``centering.nc`` / ``scaling.nc`` files get
   trivial mean=0/std=1 entries for the 15 mask vars and
   ``land_fraction`` (sftlf) during the next aggregation via
   ``make_normalization._inject_trivial_norm``. This migration does
   not touch per-dataset stats for those — the convention is owned
   by the aggregator. The version bump here is the marker that
   downstream consumers can rely on the new convention.

Sources that don't publish ``total_water_path`` (no
``water_vapor_path`` or no ``clwvi``) get a pure sidecar bump.
The per-source norm gap that creates (some sources have TWP, some
don't) is handled at training time by ``PerSourceNormalizer``'s
data-mask-aware pass-through — a variable a source lacks is left
unchanged for that source's batch slice rather than queried in a
nonexistent stats entry.
"""

import logging
import sys
import tempfile
from pathlib import Path

import fsspec
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_stats import area_weights_2d, compute_dataset_stats  # noqa: E402
from config import DEFAULT_STATS_PERIODS  # noqa: E402
from processing import compute_total_water_path  # noqa: E402
from schema_version import Migration  # noqa: E402

_WVP = "water_vapor_path"
_CLWVI = "clwvi"
_TWP = "total_water_path"


def _stats_path_for(zarr_path: str) -> str:
    return zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"


def _derive_total_water_path(zarr_path: str) -> None:
    """Compute total_water_path = water_vapor_path + clwvi and write
    it to the zarr.

    Reads only the two input variables (not the full dataset).
    Rechunks the sum explicitly to match ``water_vapor_path``'s
    on-disk layout (time=360 years-of-day chunks, full lat/lon)
    because the two augment passes that produced ``wvp`` and
    ``clwvi`` wrote with different chunk shapes — the day-cadence
    augmenter chunked clwvi as ``(3422, 6, 12)`` for some datasets,
    so a naive sum carries irregular dask chunks that ``to_zarr``
    rejects with "Zarr requires uniform chunk sizes". Re-consolidates
    the metadata at the end so subsequent ``open_zarr(consolidated=
    True)`` reads see the new variable.
    """
    logging.info("  deriving %s from %s + %s", _TWP, _WVP, _CLWVI)
    ds = xr.open_zarr(zarr_path, consolidated=False)
    try:
        twp = compute_total_water_path(ds[_WVP], ds[_CLWVI])
        # Use water_vapor_path's chunk layout as the canonical
        # target — that's the shape the rest of the atmospheric vars
        # in the zarr use (time=360, full lat/lon). Pulling chunk
        # sizes from the source variable means we adapt automatically
        # to whatever the dataset's actual layout is.
        wvp_chunks = ds[_WVP].chunks
        chunk_spec = {dim: max(wvp_chunks[i]) for i, dim in enumerate(ds[_WVP].dims)}
        twp = twp.chunk(chunk_spec)
        twp_ds = xr.Dataset({_TWP: twp})
        twp_ds.to_zarr(
            zarr_path,
            mode="a",
            consolidated=False,
            zarr_format=3,
            align_chunks=True,
        )
    finally:
        ds.close()
    zarr.consolidate_metadata(zarr_path)


def _append_total_water_path_stats(zarr_path: str, sidecar: dict) -> None:
    """Compute stats for just ``total_water_path`` and merge them
    into the existing per-dataset ``stats.nc``.

    The subset ``ds[[_TWP]]`` ensures ``compute_dataset_stats`` only
    iterates over the one new variable — the ~80 existing variables'
    entries are copied through via ``xr.merge`` without
    recomputation. Cost is one variable's time-axis read + scalar/map
    reduction per period.
    """
    grid_name = sidecar.get("target_grid", "")
    if not grid_name:
        raise RuntimeError(
            f"sidecar at {zarr_path} has no ``target_grid``; cannot "
            f"compute stats for {_TWP}."
        )
    ds = xr.open_zarr(zarr_path, consolidated=False)
    try:
        sub = ds[[_TWP]]
        n_lon = int(ds.sizes.get("lon", 90))
        w2d = area_weights_2d(grid_name, n_lon)
        new_stats_ds, _ = compute_dataset_stats(
            sub, w2d, periods=tuple(DEFAULT_STATS_PERIODS)
        )
    finally:
        ds.close()

    stats_path = _stats_path_for(zarr_path)
    fs, rel = fsspec.core.url_to_fs(stats_path)
    if not fs.exists(rel):
        raise FileNotFoundError(
            f"stats.nc missing for {zarr_path}; cannot append " f"{_TWP} stats."
        )

    # Read existing, drop any overlap (re-runs of this migration would
    # otherwise produce conflict-on-merge), merge in the new variable's
    # stats, write back. Preserve identity attrs from the existing file.
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp_in:
        fs.get(rel, tmp_in.name)
        existing = xr.open_dataset(tmp_in.name).load()
    overlap = [v for v in new_stats_ds.data_vars if v in existing.data_vars]
    if overlap:
        existing = existing.drop_vars(overlap)
    merged = xr.merge([existing, new_stats_ds])
    merged.attrs.update(existing.attrs)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_out:
        local = tmp_out.name
    try:
        merged.to_netcdf(local)
        with fs.open(rel, "wb") as fobj, open(local, "rb") as src:
            fobj.write(src.read())
    finally:
        Path(local).unlink(missing_ok=True)


def _apply(zarr_path: str, sidecar: dict) -> dict:
    variables_present = set(sidecar.get("variables_present", []) or [])
    can_derive = (
        _WVP in variables_present
        and _CLWVI in variables_present
        and _TWP not in variables_present
    )
    if can_derive:
        _derive_total_water_path(zarr_path)
        _append_total_water_path_stats(zarr_path, sidecar)
        variables_present.add(_TWP)
        sidecar["variables_present"] = sorted(variables_present)
        derived = True
    else:
        derived = False
        logging.info("  %s already present or inputs missing — sidecar-only bump", _TWP)

    sidecar["schema_version"] = "0.9.0"
    sidecar.setdefault("migrations", []).append(
        {
            "from": "0.8.1",
            "to": "0.9.0",
            "derived_total_water_path": derived,
        }
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.8.1",
    to_version="0.9.0",
    description=(
        "Derive total_water_path + append its real stats (per-dataset, "
        "single-variable, incremental — no full-dataset recompute) for "
        "the 22 v2 datasets that have both inputs (water_vapor_path + "
        "clwvi) but missed the derivation in the 0.8.1 augment pass. "
        "Marks the convention bump for trivial mean=0/std=1 "
        "normalization of masks + land_fraction in 0.9.0 centering.nc "
        "files (applied by make_normalization.py at aggregation time)."
    ),
    apply=_apply,
)
