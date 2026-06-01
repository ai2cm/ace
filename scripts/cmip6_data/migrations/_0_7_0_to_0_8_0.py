"""Migration 0.7.0 → 0.8.0: rename ``eday_ts`` → ``surface_temperature``.

Aligns the daily Eday.ts surface-temperature variable with the
SHIELD/ERA5 baseline naming. The same rename runs at fresh-ingest
time at schema 0.8.0 via the ``SurfaceAndOceanVariable`` output_name
change in ``config.py``; this migration brings any 0.7.0 zarr that
carries ``eday_ts`` into line.

Cost model. v2's ``eday_ts`` is real-daily 75-year data, ~30 MB
chunked per dataset. The migration renames the zarr array via a
storage-level prefix move (``fs.cp`` + ``fs.rm``) rather than a full
xarray-load + zarr rewrite — on GCS the copy uses the rewrite API
(server-side, no data download), which keeps the migration cheap
across the cohort (the alternative would shuffle a few GB of data
per dataset). Datasets that don't carry ``eday_ts`` (the ~60% of v2
without daily Eday.ts) become a sidecar-only bump.

Why this rename now. The 0.7.0 → 0.8.0 bump was deferred from the
config change because we expected to reprocess fresh rather than
migrate v2 in place. We later wanted to augment v2 with the missing
CFday variables (see ``process_esgf.py``'s
``_augment_day_variables``), and the augment path's schema-version
gate (``existing.schema_version != SCHEMA_VERSION`` → skip) requires
v2 to reach 0.8.0 before augmentation can run. This migration
unblocks that path.
"""

import logging
import sys
from pathlib import Path

import fsspec
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_stats import compute_and_write_stats  # noqa: E402
from config import DEFAULT_STATS_PERIODS  # noqa: E402
from schema_version import Migration  # noqa: E402

_SRC_NAME = "eday_ts"
_DST_NAME = "surface_temperature"


def _rename_zarr_array(zarr_path: str, src: str, dst: str) -> None:
    """Server-side rename of a zarr v3 array within a group.

    Implementation: ``fs.cp`` every blob from ``<zarr>/<src>/`` to
    ``<zarr>/<dst>/``, then ``fs.rm`` the source prefix. On GCS this
    is a server-side rewrite (fast, no egress). On a local filesystem
    we have to ``makedirs`` the destination subdirectories before
    each copy (gcsfs handles parents implicitly; the local backend
    doesn't).

    Re-consolidates the group's metadata at the end so subsequent
    ``xr.open_zarr(..., consolidated=True)`` reads see the new name.
    """
    fs, root = fsspec.core.url_to_fs(zarr_path)
    root = root.rstrip("/")
    src_prefix = f"{root}/{src}"
    dst_prefix = f"{root}/{dst}"

    blobs = fs.find(src_prefix)
    if not blobs:
        raise FileNotFoundError(
            f"No zarr blobs found at {src_prefix} — array missing despite "
            "appearing in the sidecar's variables_present."
        )
    for blob in blobs:
        # ``fs.find`` returns absolute paths; strip the prefix to get
        # the relative path under the array (zarr.json, c/0/0/0, ...).
        rel = blob[len(src_prefix) :].lstrip("/")
        target = f"{dst_prefix}/{rel}"
        parent = target.rsplit("/", 1)[0]
        try:
            fs.makedirs(parent, exist_ok=True)
        except (NotImplementedError, AttributeError):
            # Object stores (gcs) don't need explicit parents.
            pass
        fs.cp(blob, target)
    fs.rm(src_prefix, recursive=True)
    zarr.consolidate_metadata(zarr_path)


def _regenerate_stats(zarr_path: str, sidecar: dict) -> None:
    """Recompute stats.nc with the new variable name."""
    stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
    identity = {
        "source_id": sidecar.get("source_id", ""),
        "experiment": sidecar.get("experiment", ""),
        "variant_label": sidecar.get("variant_label", ""),
        "label": sidecar.get("label", ""),
    }
    grid_name = sidecar.get("target_grid", "")
    if not grid_name:
        raise RuntimeError(
            f"sidecar at {zarr_path} has no ``target_grid``; cannot "
            "regenerate stats."
        )
    ds = xr.open_zarr(zarr_path, consolidated=True)
    try:
        compute_and_write_stats(
            ds,
            stats_path,
            identity,
            grid_name,
            periods=tuple(DEFAULT_STATS_PERIODS),
        )
    finally:
        ds.close()


def _apply(zarr_path: str, sidecar: dict) -> dict:
    variables_present = list(sidecar.get("variables_present", []))
    if _SRC_NAME not in variables_present:
        logging.info("  no %s in this dataset; sidecar bump only", _SRC_NAME)
        sidecar["schema_version"] = "0.8.0"
        sidecar.setdefault("migrations", []).append(
            {"from": "0.7.0", "to": "0.8.0", "renamed": {}}
        )
        return sidecar

    logging.info("  renaming %s → %s in zarr", _SRC_NAME, _DST_NAME)
    _rename_zarr_array(zarr_path, _SRC_NAME, _DST_NAME)

    # Sidecar's ``variables_present`` reflects the new name. Dedup +
    # sort to match the canonical sidecar form.
    variables_present = [_DST_NAME if v == _SRC_NAME else v for v in variables_present]
    sidecar["variables_present"] = sorted(set(variables_present))

    _regenerate_stats(zarr_path, sidecar)

    sidecar["schema_version"] = "0.8.0"
    sidecar.setdefault("migrations", []).append(
        {"from": "0.7.0", "to": "0.8.0", "renamed": {_SRC_NAME: _DST_NAME}}
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.7.0",
    to_version="0.8.0",
    description=(
        "Rename eday_ts → surface_temperature for SHIELD/ERA5 baseline "
        "alignment; regenerate stats.nc. Storage-level rename via "
        "fsspec — server-side on GCS."
    ),
    apply=_apply,
)
