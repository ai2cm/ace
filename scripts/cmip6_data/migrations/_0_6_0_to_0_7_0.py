"""Migration 0.6.0 → 0.7.0.

One-off repair for ``BCC-CSM2-MR/historical/r1i1p1f1``. The source
Pangeo zarr's last day (2014-12-31) is corrupt across every plev
variable: ``ua`` peaks at ~88,400 m/s where surrounding days are
50-70 m/s; ``va`` ~440 m/s; ``hus`` ~0.02 in the stratosphere
(~8400× the cohort norm); ``zg`` up to 26,000 m at 1000 hPa (54×
typical). The corruption is sharp — the global mean of ``ua500``
jumps by ~3,200× in one step from 2014-12-30 to 2014-12-31, with
no warning in the preceding ten days — so dropping the bad
timestep is the cleanest action.

The other 242 datasets (sibling BCC-CSM2-MR variants, every other
source model) were checked and are clean. The values on this
particular day are unphysical *but* all below 1e10, so they escaped
the 0.4.0 → 0.5.0 fill-value net.

Three independent guards have to agree before any byte is touched,
so this can't silently truncate a clean dataset:

1. Sidecar identity tuple ``(source_id, experiment, variant_label)``
   exactly matches the target.
2. Sidecar pre-state (``n_timesteps`` and ``time_end``) matches the
   known-bad state from the v2 cohort write.
3. On-disk ``ua500[-1]`` actually contains the symptom (max |value|
   > 50000 m/s — physically impossible, observed at ~88400).

Any non-match → fail loudly. For non-target datasets the migration
is a pure no-op + version bump.
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


def _assert_known_bad_state(zarr_path: str, sidecar: dict) -> None:
    """Three-guard safety check. Fails loudly with a descriptive
    error if any guard disagrees with the known-bad target.
    """
    if sidecar.get("n_timesteps") != 23725:
        raise RuntimeError(
            f"refuse to truncate {zarr_path}: sidecar n_timesteps="
            f"{sidecar.get('n_timesteps')!r}, expected 23725 (the v2 "
            "cohort's known-bad BCC-CSM2-MR historical r1 length)."
        )
    if sidecar.get("time_end") != "2014-12-31 12:00:00":
        raise RuntimeError(
            f"refuse to truncate {zarr_path}: sidecar time_end="
            f"{sidecar.get('time_end')!r}, expected '2014-12-31 12:00:00'."
        )
    # Read just the last timestep of ua500, not the whole array, so
    # the guard cost is bounded.
    group = zarr.open_group(zarr_path, mode="r")
    ua500 = group["ua500"]
    last_day = np.asarray(ua500[-1])
    finite = np.isfinite(last_day)
    if not finite.any():
        raise RuntimeError(
            f"refuse to truncate {zarr_path}: ua500[-1] is entirely "
            "non-finite — can't verify the known-bad symptom."
        )
    max_abs = float(np.abs(last_day[finite]).max())
    if max_abs < 50000.0:
        raise RuntimeError(
            f"refuse to truncate {zarr_path}: ua500[-1] max |value| "
            f"= {max_abs:.1f} m/s, expected > 50000 (the BCC corruption "
            "signature). Refusing in case identity collision points us "
            "at a clean dataset."
        )
    logging.info(
        "  pre-state guards passed: n_timesteps=23725, "
        "time_end=2014-12-31 12:00:00, ua500[-1] max |value|=%.0f m/s",
        max_abs,
    )


def _truncate_last_timestep(zarr_path: str) -> list[dict]:
    """Shrink every time-dimensioned array by 1 along axis 0 via
    in-place ``zarr.Array.resize``. Returns one audit entry per
    truncated variable.

    Sharded arrays handle this cleanly — only the metadata's
    ``shape`` field changes. The dataset's last chunk holds 325
    timesteps; after this it holds 324. No shards are orphaned.
    """
    group = zarr.open_group(zarr_path, mode="r+")
    per_var: list[dict] = []
    for var in list(group.array_keys()):
        arr = group[var]
        dims = arr.metadata.dimension_names or ()
        if "time" not in dims:
            continue
        if dims.index("time") != 0:
            raise RuntimeError(
                f"{var}: expected ``time`` at axis 0, found at " f"{dims.index('time')}"
            )
        old_n = arr.shape[0]
        new_shape = (old_n - 1,) + tuple(arr.shape[1:])
        arr.resize(new_shape)
        per_var.append({"variable": var, "old_n": old_n, "new_n": new_shape[0]})
        logging.info("  %s: truncated time axis %d → %d", var, old_n, new_shape[0])
    return per_var


def _regenerate_stats(zarr_path: str, sidecar: dict) -> None:
    """Recompute stats.nc after truncation."""
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


def _update_time_window(zarr_path: str, sidecar: dict) -> None:
    """Refresh the sidecar's ``n_timesteps`` and ``time_end`` from
    the truncated zarr. Re-consolidates first so the per-array
    ``shape`` updates we just wrote are visible.
    """
    zarr.consolidate_metadata(zarr_path)
    ds = xr.open_zarr(zarr_path, consolidated=True)
    try:
        sidecar["n_timesteps"] = int(ds.sizes["time"])
        sidecar["time_end"] = str(ds["time"].isel(time=-1).values)
    finally:
        ds.close()


def _apply(zarr_path: str, sidecar: dict) -> dict:
    identity = (
        sidecar.get("source_id", ""),
        sidecar.get("experiment", ""),
        sidecar.get("variant_label", ""),
    )
    if identity != ("BCC-CSM2-MR", "historical", "r1i1p1f1"):
        logging.info(
            "  not the BCC-CSM2-MR/historical/r1i1p1f1 target; sidecar bump only"
        )
        audit: dict = {"skipped": True, "reason": "not_target"}
    else:
        logging.info(
            "  matched BCC-CSM2-MR/historical/r1i1p1f1; running pre-state guards"
        )
        _assert_known_bad_state(zarr_path, sidecar)
        per_var = _truncate_last_timestep(zarr_path)
        _update_time_window(zarr_path, sidecar)
        _regenerate_stats(zarr_path, sidecar)
        audit = {"variables_truncated": per_var}
        logging.info(
            "  truncated %d variable(s); new n_timesteps=%d, time_end=%s",
            len(per_var),
            sidecar["n_timesteps"],
            sidecar["time_end"],
        )

    sidecar["schema_version"] = "0.7.0"
    sidecar.setdefault("migrations", []).append(
        {
            "from": "0.6.0",
            "to": "0.7.0",
            "bcc_csm2_mr_last_day_truncation": audit,
        }
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.6.0",
    to_version="0.7.0",
    description=(
        "Truncate corrupt last timestep (2014-12-31) of "
        "BCC-CSM2-MR/historical/r1i1p1f1; no-op for all other datasets."
    ),
    apply=_apply,
)
