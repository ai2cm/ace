"""Migration 0.4.0 → 0.5.0.

Clean up publisher fill-value leaks already written to disk. The 0.4.0
stats sweep surfaced cells with magnitudes >> any physical value in
several datasets:

- **BCC-CSM2-MR/historical/r1i1p1f1**: 6400 cells on the last
  historical day filled with ``1e+36`` across all 7 plev levels
  for ``ua*``. The source Pangeo zarr declares ``_FillValue=1e+20``
  but the stored bytes are ``1e+36``, so xarray's mask_and_scale
  decode misses them.
- **CESM2/ssp245,ssp585** (5 variants): ``omon_tob`` augments leak
  raw ``9.97e+36`` (netCDF C default fill) because the source NetCDF
  has no usable ``_FillValue`` attribute.
- **FGOALS-f3-L/historical/r1i1p1f1**: ``omon_tob``/``omon_mlotst``
  with stored ``1e+35``.
- **MPI-ESM1-2-LR**: ``siday_sithick`` with values 1e+15-1e+16 on 4
  variants. The publisher's CMOR ``history`` attribute claims a
  ``-9e+33`` → ``1e+20`` fill replacement was applied, but a residue
  at 1e+15 was missed.

These are all bugs that fresh runs now catch via
:func:`processing.decode_default_fills` at source-open time. The
migration applies the same magnitude clip (``|value| >= 1e10`` → NaN)
to the already-written zarr cells, then regenerates ``stats.nc``.

The threshold is 9+ orders of magnitude above the largest physical
value any variable in this dataset takes (CO2 ppm tops out near 2500;
wind ~200 m/s), so the clip has effectively zero false-positive
surface — there's no physical variable that legitimately approaches
1e10. Each variable's affected cell count is logged + persisted in
the sidecar audit.

What's *not* fixed here: GFDL-CM4 / CMCC ``HGTsfc`` is ~20% finite
because :func:`process_one` previously selected mismatched
``orog``/``sftlf`` source grids (``gr1``/``gr2``) and the merge-then-
regrid step produced a sparse union. That's fixed in
:func:`process_one` for fresh runs but can't be repaired in place
(no source access from a migrate pod); those 10 datasets need a
full reprocess to recover their orography.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_stats import compute_and_write_stats  # noqa: E402
from config import DEFAULT_STATS_PERIODS  # noqa: E402
from processing import _FILL_VALUE_THRESHOLD  # noqa: E402
from schema_version import Migration  # noqa: E402


def _clip_fill_leaks(zarr_path: str) -> list[dict]:
    """For each float variable in ``zarr_path``, NaN cells whose
    magnitude exceeds ``_FILL_VALUE_THRESHOLD``. Returns a list of
    per-variable audit dicts.

    Reads + writes go through the zarr library directly (not xarray)
    so we don't materialise the whole dataset — large vars (e.g.
    31k-timestep ``ua250``) would blow the pod's 16 GiB RSS if loaded
    via xarray.
    """
    group = zarr.open_group(zarr_path, mode="r+")
    audit: list[dict] = []
    for var in list(group.array_keys()):
        arr = group[var]
        if arr.dtype.kind != "f":
            continue
        # Single ``.[:]`` load per variable. The largest production
        # var (``ua250`` 31k×45×90 float32 ≈ 500 MB) is well under
        # the migrate-pod's 16 GiB RSS limit.
        data = arr[:]
        bad = np.abs(data) >= _FILL_VALUE_THRESHOLD
        n_bad = int(bad.sum())
        if n_bad == 0:
            del data, bad
            continue
        max_bad = float(np.abs(data[bad]).max())
        data = data.astype(np.float32, copy=False)
        data[bad] = np.float32("nan")
        arr[:] = data
        audit.append(
            {
                "variable": var,
                "n_bad": n_bad,
                "max_abs_value": max_bad,
                "threshold": _FILL_VALUE_THRESHOLD,
            }
        )
        logging.info(
            "  %s: %d cells with |value| >= %g (max |value| %.3g) "
            "→ NaN (publisher fill-value leak)",
            var,
            n_bad,
            _FILL_VALUE_THRESHOLD,
            max_bad,
        )
        del data, bad
    return audit


def _regenerate_stats(zarr_path: str, sidecar: dict) -> None:
    """After clipping, refresh stats.nc so downstream aggregation
    sees the cleaned values."""
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
    # Re-consolidate metadata so xarray's consolidated open sees the
    # updated array bytes.
    zarr.consolidate_metadata(zarr_path)
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
    audit = _clip_fill_leaks(zarr_path)
    if audit:
        _regenerate_stats(zarr_path, sidecar)
        logging.info(
            "  regenerated stats.nc after clipping %d variables",
            len(audit),
        )
    else:
        logging.info("  no fill-value leaks detected; sidecar bump only")

    sidecar["schema_version"] = "0.5.0"
    sidecar.setdefault("migrations", []).append(
        {
            "from": "0.4.0",
            "to": "0.5.0",
            "fill_value_clips": audit,
        }
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.4.0",
    to_version="0.5.0",
    description=(
        "NaN publisher fill-value leaks (|value| >= 1e10 → NaN) and "
        "regenerate stats.nc."
    ),
    apply=_apply,
)
