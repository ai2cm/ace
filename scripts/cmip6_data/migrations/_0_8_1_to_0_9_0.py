"""Migration 0.8.1 → 0.9.0: derive ``total_water_path`` + normalize masks.

Two conventions change in this version, both affecting how derived /
trivial fields are surfaced:

1. **``total_water_path`` backfill.** The ESGF augment in 0.8.1's
   ``process_esgf.py`` only derived ``total_water_path = water_vapor_path
   + clwvi`` when ``water_vapor_path`` was *pre-existing* in the zarr
   (i.e., from Pangeo's ingest), not when it was added by the same
   augment pass via Eday.prw. That mismatch missed 22 of 26 eligible
   v2 datasets. The augment-side bug is fixed in process_esgf.py;
   this migration backfills the derivation for already-augmented
   datasets so they don't need a fresh augment pass.

2. **Mask + sftlf trivial normalization.** Mask channels
   (``below_surface_mask{plev}``, ``oday_*_mask``, ``siday_*_mask``)
   and the static ``land_fraction`` field (``sftlf``) are now
   expected to have mean=0/std=1 entries in the cohort + per-source
   ``centering.nc`` / ``scaling.nc`` files — preserving their
   0/1-valued semantics rather than standardising. The
   make_normalization.py change owns the centering-file rewrite;
   this migration doesn't touch zarrs for that convention, just
   records the version bump so consumers know the 0.9.0 stats
   files include trivial entries for those names.

Behaviour:

- If ``water_vapor_path`` and ``clwvi`` are both present in
  ``variables_present`` and ``total_water_path`` is absent: derive
  the sum, write the new variable into the zarr via
  ``to_zarr(mode="a", consolidated=False)``, re-consolidate, and
  append ``total_water_path`` stats to the existing ``stats.nc``
  (incremental — does not recompute the other ~80 variables).
- Otherwise: sidecar-only bump (schema version + audit entry).

Sidecar audit log records ``derived_total_water_path`` either as
``True`` or ``False`` to make pre/post comparisons trivial.
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


def _append_var_stats(
    zarr_path: str,
    sidecar: dict,
    var_name: str,
) -> None:
    """Compute stats for a single new variable and merge them into
    the existing ``stats.nc``.

    Why incremental: a full recompute (re-running
    ``compute_dataset_stats`` over every variable) is O(minutes) per
    dataset and pointlessly redoes the ~80 variables whose values
    didn't change. Subsetting the dataset to just ``var_name``
    before calling ``compute_dataset_stats`` keeps the migration
    fast at scale.
    """
    grid_name = sidecar.get("target_grid", "")
    if not grid_name:
        raise RuntimeError(
            f"sidecar at {zarr_path} has no ``target_grid``; cannot "
            "compute stats for the new variable."
        )
    ds = xr.open_zarr(zarr_path, consolidated=False)
    try:
        sub = ds[[var_name]]
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
            f"stats.nc missing for {zarr_path}; cannot append " f"{var_name} stats."
        )

    # Read existing stats.nc, merge in the new variable's stats,
    # write back. Drop overlapping vars from the existing dataset
    # so xr.merge doesn't conflict on coord re-emission (e.g.
    # ``period``).
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp_in:
        fs.get(rel, tmp_in.name)
        existing = xr.open_dataset(tmp_in.name).load()
    overlap = [v for v in new_stats_ds.data_vars if v in existing.data_vars]
    if overlap:
        existing = existing.drop_vars(overlap)
    merged = xr.merge([existing, new_stats_ds])
    # Preserve identity attrs the original stats.nc carried.
    merged.attrs.update(existing.attrs)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_out:
        local = tmp_out.name
    try:
        merged.to_netcdf(local)
        with fs.open(rel, "wb") as fobj, open(local, "rb") as src:
            fobj.write(src.read())
    finally:
        Path(local).unlink(missing_ok=True)


def _derive_total_water_path(zarr_path: str, sidecar: dict) -> dict:
    """Compute total_water_path and write it into the zarr.

    Returns an updated sidecar with ``total_water_path`` appended to
    ``variables_present``. Stats are appended via
    :func:`_append_var_stats`.
    """
    logging.info("  deriving %s from %s + %s", _TWP, _WVP, _CLWVI)
    ds = xr.open_zarr(zarr_path, consolidated=False)
    try:
        twp = compute_total_water_path(ds[_WVP], ds[_CLWVI])
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
    _append_var_stats(zarr_path, sidecar, _TWP)

    variables_present = sorted(set(sidecar.get("variables_present", [])) | {_TWP})
    sidecar["variables_present"] = variables_present
    return sidecar


def _apply(zarr_path: str, sidecar: dict) -> dict:
    variables_present = set(sidecar.get("variables_present", []) or [])
    can_derive = (
        _WVP in variables_present
        and _CLWVI in variables_present
        and _TWP not in variables_present
    )
    if can_derive:
        sidecar = _derive_total_water_path(zarr_path, sidecar)
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
        "Derive total_water_path for datasets that have both inputs "
        "(water_vapor_path + clwvi) but missed the derivation in the "
        "0.8.1 augment pass (22 of 26 eligible v2 datasets). Marks "
        "the convention bump for mask + sftlf trivial normalization "
        "in 0.9.0 centering.nc files."
    ),
    apply=_apply,
)
