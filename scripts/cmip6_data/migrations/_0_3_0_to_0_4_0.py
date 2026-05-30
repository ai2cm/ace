"""Migration 0.3.0 → 0.4.0.

Regenerate ``stats.nc`` per dataset with the new layout that includes
per-cell maps (``{var}__time_mean_map``, ``{var}__time_var_map``,
``{var}__n_valid_map``, ``{var}__d1_var_map``) in addition to the
existing scalar stats. Zarr data is unchanged — this is a pure
stats-file rewrite so ``make_normalization`` can aggregate pooled
per-cell quantities cheaply across the cohort without re-scanning each
zarr.

Implementation: open the zarr, call ``compute_and_write_stats`` (which
overwrites ``stats.nc`` next to the zarr). Identity and grid name come
from the existing sidecar — no config lookup required.
"""

import logging
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_stats import compute_and_write_stats  # noqa: E402
from config import DEFAULT_STATS_PERIODS  # noqa: E402
from schema_version import Migration  # noqa: E402


def _apply(zarr_path: str, sidecar: dict) -> dict:
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
            "regenerate stats without it (area_weights need the grid name)."
        )

    ds = xr.open_zarr(zarr_path, consolidated=True)
    try:
        compute_and_write_stats(
            ds, stats_path, identity, grid_name, periods=tuple(DEFAULT_STATS_PERIODS)
        )
    finally:
        ds.close()
    logging.info("  regenerated %s with per-cell maps", stats_path)

    sidecar["schema_version"] = "0.4.0"
    sidecar.setdefault("migrations", []).append(
        {
            "from": "0.3.0",
            "to": "0.4.0",
            "stats_layout": "per-cell maps added",
        }
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.3.0",
    to_version="0.4.0",
    description=(
        "Regenerate stats.nc with per-cell maps (time_mean_map, "
        "time_var_map, n_valid_map, d1_var_map) for downstream pooled "
        "normalization aggregation."
    ),
    apply=_apply,
)
