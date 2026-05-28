"""Migration 0.2.0 → 0.3.0.

Two on-disk changes:

1. **Drop derived layer-mean T**. ``ta_derived_layer_{lo}_{hi}`` was
   computed from zg + hus via the hypsometric equation. We stopped
   carrying these as primary training inputs — consumers that need a
   layer-mean T can derive it on the fly. Implementation is a
   ``del group[var]`` per variable + ``zarr.consolidate_metadata`` +
   a trim of the corresponding entries from ``stats.nc``.
2. **Re-fill below-surface plev cells with smooth flood**. Earlier
   schema versions filled below-surface cells via nearest-above-fill
   (column-wise vertical copy from the lowest above-surface level).
   The new pipeline uses ``fill_below_surface_smooth`` (the same
   smooth flood fill the model uses at runtime, via
   ``fme.core.fill``). For each per-level ``{var}{hPa}`` in
   ``(ua, va, hus, zg) × plev``, we re-NaN cells where the stored
   ``below_surface_mask{hPa}`` is 1 and re-fill via
   :func:`fill_below_surface_smooth`. The stored mask is the same one
   that drove the original fill, so re-NaN + re-fill is value-
   equivalent to running the new pipeline from scratch (verified by
   the migrated-vs-fresh regression test).
"""

import logging
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema_version import Migration  # noqa: E402

_LEVELED_VARS = ("ua", "va", "hus", "zg")
_LEVEL_RE = re.compile(r"^(ua|va|hus|zg)(\d+)$")


def _list_derived_layer_vars(zarr_path: str) -> list[str]:
    group = zarr.open_group(zarr_path, mode="r")
    return sorted(name for name in group if name.startswith("ta_derived_layer"))


def _enumerate_leveled_vars(zarr_path: str) -> list[tuple[str, str]]:
    """List ``(var_name, hpa_label)`` pairs for plev-flattened state
    variables present in the zarr (e.g. ``("ua1000", "1000")``).
    """
    group = zarr.open_group(zarr_path, mode="r")
    pairs: list[tuple[str, str]] = []
    for name in group:
        m = _LEVEL_RE.match(name)
        if m is None:
            continue
        pairs.append((name, m.group(2)))
    return pairs


def _refill_below_surface(zarr_path: str) -> list[str]:
    """For each ``{var}{hPa}`` present, re-NaN cells where the matching
    ``below_surface_mask{hPa}`` is 1 and refill with
    :func:`fill_below_surface_smooth`. Returns the list of variables
    that were refilled.

    Reads + writes go through the zarr library directly (not xarray)
    so we don't accumulate a per-variable cache across iterations.
    With xarray, ``ds[var].load()`` 32× over a 31k-timestep dataset
    runs the local process into multi-GB RSS and gets OOM-killed.
    The zarr-direct path keeps memory bounded to ~2 GB per variable.
    """
    from processing import fill_below_surface_smooth

    pairs = _enumerate_leveled_vars(zarr_path)
    if not pairs:
        return []

    group = zarr.open_group(zarr_path, mode="r+")
    refilled: list[str] = []
    for var_name, hpa in pairs:
        mask_name = f"below_surface_mask{hpa}"
        if mask_name not in group:
            # No stored mask for this level — earlier schema may have
            # skipped writing it (no orog, no NaN pattern). Without
            # the mask we can't safely re-NaN, so leave the variable.
            continue
        # Direct zarr reads, no xarray cache.
        da_vals = group[var_name][:]  # (time, lat, lon)
        mask_vals = group[mask_name][:]  # (time, lat, lon)
        # fill_below_surface_smooth expects (time, plev, lat, lon)
        # xr.DataArrays. Wrap with a singleton plev dim, call, then
        # drop it. Cheap — no zarr re-read.
        da4d = xr.DataArray(
            da_vals[:, None],
            dims=("time", "plev", "lat", "lon"),
        )
        mask4d = xr.DataArray(
            mask_vals[:, None],
            dims=("time", "plev", "lat", "lon"),
        )
        refilled_da = fill_below_surface_smooth(da4d, mask4d).isel(plev=0, drop=True)
        group[var_name][:] = refilled_da.values.astype(np.float32)
        refilled.append(var_name)
        # Help the GC release the working buffers before the next
        # variable's load — important for prod-scale (31k timesteps).
        del da_vals, mask_vals, da4d, mask4d, refilled_da
    return refilled


def _apply(zarr_path: str, sidecar: dict) -> dict:
    layer_vars = _list_derived_layer_vars(zarr_path)

    # 1. Drop derived layer-mean T variables (if present).
    if layer_vars:
        group = zarr.open_group(zarr_path, mode="r+")
        for v in layer_vars:
            del group[v]
        zarr.consolidate_metadata(zarr_path)
        logging.info("  dropped %d ta_derived_layer_* vars", len(layer_vars))

    # 2. Re-fill below-surface plev cells via smooth flood.
    refilled = _refill_below_surface(zarr_path)
    if refilled:
        zarr.consolidate_metadata(zarr_path)
        logging.info("  refilled %d level vars via smooth flood", len(refilled))

    # 3. Trim stats.nc entries for the dropped layer vars + regenerate
    # entries for the refilled vars. The refill changes values in
    # below-surface cells, so all stats keys for those vars are stale.
    stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
    if layer_vars or refilled:
        try:
            stats = xr.open_dataset(stats_path, engine="h5netcdf").load()
            stats.close()
            stale_prefixes = tuple(f"{v}__" for v in (layer_vars + refilled))
            to_drop = [k for k in stats.data_vars if k.startswith(stale_prefixes)]
            if to_drop:
                stats = stats.drop_vars(to_drop)
                stats.to_netcdf(stats_path, mode="w", engine="h5netcdf")
                logging.info(
                    "  trimmed %d stats.nc entries (%d dropped vars, %d refilled vars)",
                    len(to_drop),
                    len(layer_vars),
                    len(refilled),
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

    # 4. Sidecar bookkeeping.
    remaining = [v for v in sidecar.get("variables_present", []) if v not in layer_vars]
    sidecar["variables_present"] = sorted(remaining)
    sidecar["schema_version"] = "0.3.0"
    sidecar.setdefault("migrations", []).append(
        {
            "from": "0.2.0",
            "to": "0.3.0",
            "removed": list(layer_vars),
            "refilled": refilled,
        }
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.2.0",
    to_version="0.3.0",
    description=(
        "Drop ta_derived_layer_* vars + refill below-surface plev cells "
        "with smooth flood fill (replacing nearest-above)."
    ),
    apply=_apply,
)
