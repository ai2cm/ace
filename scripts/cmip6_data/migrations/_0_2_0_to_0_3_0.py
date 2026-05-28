"""Migration 0.2.0 → 0.3.0: drop ``ta_derived_layer_*`` variables.

Removes the per-layer derived temperature channels (e.g.
``ta_derived_layer_1000_850``, ..., ``ta_derived_layer_50_10``) that
the pipeline used to compute from ``zg`` + ``hus`` via the
hypsometric equation. We decided to stop carrying these as primary
training inputs — downstream consumers that need a layer-mean T can
derive it on the fly.

Implementation is just a chunk delete + metadata re-consolidation
plus a trim of the corresponding entries in ``stats.nc``. No
variable-write happens, so this migration runs in seconds per
dataset (the rate-limiter is just the GCS object-delete fan-out).
"""

import logging
import sys
from pathlib import Path

import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema_version import Migration  # noqa: E402


def _list_derived_layer_vars(zarr_path: str) -> list[str]:
    """Return the names of ``ta_derived_layer_*`` variables present on
    disk. Reads the root group directly so we don't pull the per-chunk
    metadata for the whole dataset just to enumerate variable names.
    """
    group = zarr.open_group(zarr_path, mode="r")
    return sorted(name for name in group if name.startswith("ta_derived_layer"))


def _apply(zarr_path: str, sidecar: dict) -> dict:
    layer_vars = _list_derived_layer_vars(zarr_path)
    if not layer_vars:
        sidecar["schema_version"] = "0.3.0"
        sidecar.setdefault("migrations", []).append(
            {"from": "0.2.0", "to": "0.3.0", "removed": []}
        )
        return sidecar

    group = zarr.open_group(zarr_path, mode="r+")
    for v in layer_vars:
        # Removing the group node deletes its chunk store entries; this
        # is the supported way to drop a variable in zarr v3.
        del group[v]
    zarr.consolidate_metadata(zarr_path)

    # Trim per-variable entries from stats.nc. Each removed variable
    # contributes ``<var>__mean``, ``<var>__std``, ``<var>__d1_mean``,
    # ``<var>__d1_std`` keys (and an unfilled-vs-filled split for
    # variables that have one). Drop all of them.
    stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
    try:
        stats = xr.open_dataset(stats_path, engine="h5netcdf").load()
        stats.close()
        prefixes = tuple(f"{v}__" for v in layer_vars)
        to_drop = [k for k in stats.data_vars if k.startswith(prefixes)]
        if to_drop:
            stats = stats.drop_vars(to_drop)
            stats.to_netcdf(stats_path, mode="w", engine="h5netcdf")
            logging.info(
                "  trimmed %d stats.nc entries for %d removed layer vars",
                len(to_drop),
                len(layer_vars),
            )
    except FileNotFoundError:
        logging.warning(
            "  stats.nc not present at %s — skipping stats trim", stats_path
        )
    except Exception as e:  # noqa: BLE001
        logging.warning(
            "  stats.nc trim failed during 0.2.0→0.3.0 for %s: %s",
            zarr_path,
            e,
        )

    remaining = [v for v in sidecar.get("variables_present", []) if v not in layer_vars]
    sidecar["variables_present"] = sorted(remaining)
    sidecar["schema_version"] = "0.3.0"
    sidecar.setdefault("migrations", []).append(
        {"from": "0.2.0", "to": "0.3.0", "removed": list(layer_vars)}
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.2.0",
    to_version="0.3.0",
    description=(
        "Drop ta_derived_layer_* variables from the zarr and matching stats.nc keys."
    ),
    apply=_apply,
)
