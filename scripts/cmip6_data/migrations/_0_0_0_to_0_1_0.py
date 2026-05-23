"""Migration 0.0.0 → 0.1.0: rename eday_ts → surface_temperature
and orog → HGTsfc.

Captures the schema change that introduced this framework: the daily
surface-T composite and static surface altitude were renamed in
``CMIP_TO_OUTPUT_RENAMES`` to match the SHIELD/ERA5 baseline naming.
Sidecars written before this framework had no ``schema_version``
field; those are treated as ``"0.0.0"``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema_version import Migration  # noqa: E402

_RENAMES = {
    "eday_ts": "surface_temperature",
    "orog": "HGTsfc",
}


def _apply(zarr_path: str, sidecar: dict) -> dict:
    """Open the dataset's zarr, apply ``_RENAMES`` to any matching
    variables, and rewrite the zarr in place. Regenerate ``stats.nc``
    because the old file's keys (``<var>__<stat>``) reference the
    pre-rename names. Update the sidecar's ``schema_version``,
    ``variables_present``, and append a ``migrations`` audit entry.
    """
    import xarray as xr
    from compute_stats import compute_and_write_stats
    from processing import write_zarr

    ds = xr.open_zarr(zarr_path, consolidated=True)
    applicable = {src: dst for src, dst in _RENAMES.items() if src in ds.data_vars}

    if not applicable:
        # Dataset doesn't carry either variable (rare, but possible for
        # ESGF datasets missing both). Stamp the version and continue.
        ds.close()
        sidecar["schema_version"] = "0.1.0"
        sidecar.setdefault("migrations", []).append(
            {"from": "0.0.0", "to": "0.1.0", "renamed": {}}
        )
        return sidecar

    renamed = ds.rename(applicable)
    for src, dst in applicable.items():
        # Preserve a pre-existing original_name (set during the initial
        # finalize / rename step); only stamp it ourselves if missing.
        if "original_name" not in renamed[dst].attrs:
            renamed[dst].attrs["original_name"] = src

    # Materialize before the in-place rewrite so the open handle on
    # ``zarr_path`` can be closed cleanly before we overwrite.
    renamed = renamed.load()
    ds.close()

    # Reuse the source zarr's chunking. Pull chunk_time / shard_time
    # from any time-dimensioned variable's encoding so the rewrite
    # preserves the on-disk layout.
    chunk_time, shard_time = _infer_chunks(renamed)
    # ``write_zarr`` only reads ``cfg.chunking``; the rest of
    # ``ResolvedDatasetConfig`` is irrelevant for an in-place rewrite.
    # The cast keeps mypy from blocking on the duck-typed stand-in.
    from typing import cast

    from config import ResolvedDatasetConfig

    write_zarr(
        renamed,
        zarr_path,
        cast(ResolvedDatasetConfig, _MigrationCfg(chunk_time, shard_time)),
    )

    stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
    try:
        compute_and_write_stats(
            renamed,
            stats_path,
            identity={
                "source_id": sidecar.get("source_id", ""),
                "experiment": sidecar.get("experiment", ""),
                "variant_label": sidecar.get("variant_label", ""),
                "label": sidecar.get("label", ""),
            },
            grid_name="F22.5",
            periods=None,  # uses DEFAULT_STATS_PERIODS
        )
    except Exception as e:  # noqa: BLE001
        logging.warning(
            "  stats regeneration failed for %s during 0.0.0→0.1.0: %s",
            zarr_path,
            e,
        )

    sidecar["schema_version"] = "0.1.0"
    sidecar["variables_present"] = sorted(renamed.data_vars)
    sidecar.setdefault("migrations", []).append(
        {"from": "0.0.0", "to": "0.1.0", "renamed": applicable}
    )
    return sidecar


def _infer_chunks(ds) -> tuple[int, int | None]:
    """Best-effort extraction of (chunk_time, shard_time) from an
    already-encoded dataset, used to preserve layout during the
    in-place rewrite. Defaults to ``(365, None)`` when no encoding
    information is available.
    """
    for v in ds.data_vars:
        var = ds[v]
        if "time" not in var.dims:
            continue
        time_axis = var.dims.index("time")
        chunks = var.encoding.get("chunks")
        shards = var.encoding.get("shards")
        if chunks is not None:
            return (int(chunks[time_axis]), int(shards[time_axis]) if shards else None)
    return (365, None)


class _MigrationCfg:
    """Stand-in for ``ResolvedDatasetConfig`` exposing just the
    ``chunking`` block ``write_zarr`` needs. Keeps the migration
    decoupled from the full config-resolution pipeline.
    """

    def __init__(self, chunk_time: int, shard_time: int | None) -> None:
        self.chunking = _ChunkingStub(chunk_time, shard_time)


class _ChunkingStub:
    def __init__(self, chunk_time: int, shard_time: int | None) -> None:
        self.chunk_time = chunk_time
        self.shard_time = shard_time


MIGRATION = Migration(
    from_version="0.0.0",
    to_version="0.1.0",
    description=(
        "Rename eday_ts → surface_temperature and orog → HGTsfc; "
        "regenerate stats.nc."
    ),
    apply=_apply,
)
