"""Migration 0.5.0 → 0.6.0.

Convert temperature variables whose stored ``units`` attribute is
Celsius to Kelvin in place. ESGF augment used to skip the
:func:`processing.harmonize_temperature_to_kelvin` step that the
fresh process / process_esgf paths run, so 77 datasets ended up with
``omon_tob`` stored in degC (cohort mean ~3 K — nonsense for ocean
bottom temperature, which should be 273-280 K). This migration walks
each zarr's variables, detects degC by reading the per-variable
``units`` attribute, adds 273.15, and writes back. The augment write
path itself was fixed in the same commit so future runs no longer
need this migration.

Variables that are already Kelvin (most temperatures), non-temperature
(everything else), or mask channels are untouched. The audit logs
exactly which variables were converted per dataset.
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
from schema_version import Migration  # noqa: E402

# Tokens that count as Celsius in CMIP6 / CF metadata. Mirrors
# the table inside :func:`processing.harmonize_temperature_to_kelvin`
# — duplicated here so the migration doesn't have to import the
# private constant.
_CELSIUS_TOKENS = frozenset(
    {"degc", "degreec", "degreescelsius", "celsius", "°c", "deg_c"}
)


def _units_is_celsius(arr_attrs: dict) -> bool:
    raw = str(arr_attrs.get("units", "")).strip().lower()
    raw = raw.replace("**", "").replace("^", "")
    return raw in _CELSIUS_TOKENS


def _convert_degc_to_k(zarr_path: str) -> list[dict]:
    """For each float variable whose stored ``units`` attribute is
    Celsius, add 273.15 and overwrite the array. Returns a per-variable
    audit list.
    """
    group = zarr.open_group(zarr_path, mode="r+")
    audit: list[dict] = []
    for var in list(group.array_keys()):
        arr = group[var]
        if arr.dtype.kind != "f":
            continue
        if var.endswith("_mask"):
            continue
        if not _units_is_celsius(dict(arr.attrs)):
            continue
        data = arr[:]
        # Skip if the stored values are obviously already Kelvin —
        # defensive against an earlier partial-conversion run that
        # bumped values but didn't update the units attr.
        finite = np.isfinite(data)
        if finite.any() and float(np.nanmin(data[finite])) > 100.0:
            logging.warning(
                "  %s declares units=degC but min %.2f > 100 — already K? skipping",
                var,
                float(np.nanmin(data[finite])),
            )
            continue
        old_min = float(np.nanmin(data[finite])) if finite.any() else float("nan")
        old_max = float(np.nanmax(data[finite])) if finite.any() else float("nan")
        data = data + np.float32(273.15)
        arr[:] = data
        arr.attrs["units"] = "K"
        audit.append(
            {
                "variable": var,
                "pre_min": old_min,
                "pre_max": old_max,
            }
        )
        logging.info(
            "  %s: degC → K (added 273.15; pre-conversion range " "[%.2f, %.2f])",
            var,
            old_min,
            old_max,
        )
        del data, finite
    return audit


def _regenerate_stats(zarr_path: str, sidecar: dict) -> None:
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
    audit = _convert_degc_to_k(zarr_path)
    if audit:
        _regenerate_stats(zarr_path, sidecar)
        logging.info(
            "  regenerated stats.nc after converting %d variable(s)",
            len(audit),
        )
    else:
        logging.info("  no degC variables found; sidecar bump only")

    sidecar["schema_version"] = "0.6.0"
    sidecar.setdefault("migrations", []).append(
        {
            "from": "0.5.0",
            "to": "0.6.0",
            "degc_to_k": audit,
        }
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.5.0",
    to_version="0.6.0",
    description=(
        "Convert temperature variables stored in degC to Kelvin "
        "(augment-path harmonize_temperature_to_kelvin was missing); "
        "regenerate stats.nc."
    ),
    apply=_apply,
)
