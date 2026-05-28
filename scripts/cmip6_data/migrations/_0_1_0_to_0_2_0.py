"""Migration 0.1.0 → 0.2.0: add ``log_input4mips_co2``.

Computes the natural log of the existing per-dataset
``input4mips_co2`` channel and appends it to the zarr store as
``log_input4mips_co2``. The new variable has the same dtype, shape,
and chunking as ``input4mips_co2`` (broadcast scalar over
``(time, lat, lon)``) so downstream code can treat them
interchangeably.

Unlike the 0.0.0→0.1.0 migration (which renamed existing variables
and therefore had to rewrite the full zarr), this migration only
adds a variable. We use ``to_zarr(mode="a")`` so only the new
variable's chunks land on disk — typically seconds per dataset.

Datasets that lack ``input4mips_co2`` (rare ESGF entries with no
staged external forcings) are stamped at the new version with no
on-disk change so the chain still composes.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema_version import Migration  # noqa: E402


def _apply(zarr_path: str, sidecar: dict) -> dict:
    ds = xr.open_zarr(zarr_path, consolidated=True)
    if "input4mips_co2" not in ds.data_vars:
        ds.close()
        sidecar["schema_version"] = "0.2.0"
        sidecar.setdefault("migrations", []).append(
            {"from": "0.1.0", "to": "0.2.0", "added": []}
        )
        return sidecar

    co2 = ds["input4mips_co2"]
    log_co2 = np.log(co2).astype(np.float32)
    log_co2 = log_co2.rename("log_input4mips_co2")
    log_co2.attrs = {
        "units": "ln(ppm)",
        "long_name": "natural log of global mean CO2 mole fraction",
        "description": (
            "Natural log (ln, base e) of input4mips_co2. Base matches "
            "the Myhre et al. (1998) CO2 radiative-forcing formula "
            "ΔF = 5.35 · ln(C/C₀), so a unit change in this channel "
            "corresponds to one e-fold of CO2 — the physically "
            "relevant scale for the model's response. log10 would "
            "carry the same information off the physics axis."
        ),
    }
    # Match the source variable's chunk/shard encoding so the new
    # channel is layed out identically on disk.
    log_co2.encoding = {
        k: v
        for k, v in co2.encoding.items()
        if k in ("chunks", "shards", "dtype", "compressors", "filters")
    }
    log_co2.encoding["dtype"] = "float32"

    log_co2.to_dataset().to_zarr(
        zarr_path,
        mode="a",
        consolidated=False,
        zarr_format=3,
        align_chunks=True,
    )
    zarr.consolidate_metadata(zarr_path)
    ds.close()

    # Regenerate stats.nc so log_input4mips_co2 has per-dataset stats.
    # Re-open the just-written zarr (cheap: target-resolution, no
    # upstream dask graph) to avoid retaining the migration-time
    # in-memory dataset.
    from compute_stats import compute_and_write_stats

    stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
    try:
        written = xr.open_zarr(zarr_path, consolidated=True)
        compute_and_write_stats(
            written,
            stats_path,
            identity={
                "source_id": sidecar.get("source_id", ""),
                "experiment": sidecar.get("experiment", ""),
                "variant_label": sidecar.get("variant_label", ""),
                "label": sidecar.get("label", ""),
            },
            grid_name="F22.5",
            periods=None,
        )
        written.close()
    except Exception as e:  # noqa: BLE001
        logging.warning(
            "  stats regeneration failed for %s during 0.1.0→0.2.0: %s",
            zarr_path,
            e,
        )

    sidecar["schema_version"] = "0.2.0"
    sidecar["variables_present"] = sorted(
        set(sidecar.get("variables_present", [])) | {"log_input4mips_co2"}
    )
    sidecar.setdefault("migrations", []).append(
        {"from": "0.1.0", "to": "0.2.0", "added": ["log_input4mips_co2"]}
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.1.0",
    to_version="0.2.0",
    description=("Add log_input4mips_co2 = ln(input4mips_co2); regenerate stats.nc."),
    apply=_apply,
)
