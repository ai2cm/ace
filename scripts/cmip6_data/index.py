"""Dataset-index schema and writers for the CMIP6 daily pilot.

The processing script builds one :class:`DatasetIndexRow` per attempted
dataset and, at the end of the run, writes:

- ``<output_directory>/index.parquet`` (if a parquet engine is
  available — pyarrow or fastparquet; logged-and-skipped otherwise)
- ``<output_directory>/index.csv`` (always)
- ``<zarr_path>/metadata.json`` sidecar alongside each successfully
  written zarr (not written when ``status != "ok"``).
"""

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import fsspec
import pandas as pd


@dataclass
class DatasetIndexRow:
    # Identity
    source_id: str
    experiment: str
    variant_label: str
    variant_r: Optional[int] = None
    variant_i: Optional[int] = None
    variant_p: Optional[int] = None
    variant_f: Optional[int] = None
    label: str = ""

    # Provenance (inputs)
    core_zstores: list[str] = field(default_factory=list)
    # Map of surface-and-ocean output_name -> source zstore URL (one entry per
    # surface-and-ocean variable actually opened for this dataset).
    surface_and_ocean_zstores: dict[str, str] = field(default_factory=dict)
    static_zstores: list[str] = field(default_factory=list)

    # Processing
    target_grid: str = ""
    native_grid_label: str = ""
    native_calendar: str = ""
    regrid_methods: dict[str, str] = field(default_factory=dict)
    mask_source: str = ""

    # Output
    output_zarr: str = ""
    n_timesteps: int = 0
    time_start: str = ""
    time_end: str = ""
    variables_present: list[str] = field(default_factory=list)

    # Status / audit
    status: str = "pending"  # "ok" | "skipped" | "failed" | "pending"
    skip_reason: str = ""
    cell_methods_mismatch: list[str] = field(default_factory=list)
    n_nan_input_cells: int = 0
    warnings: list[str] = field(default_factory=list)
    # Schema-version stamp recording which on-disk format the dataset
    # was written under. ``migrate.py`` reads this and applies the
    # registered migration chain to bring older datasets up to the
    # current ``SCHEMA_VERSION``. Sidecars written before this field
    # existed default to ``"0.0.0"`` in migrate.py.
    schema_version: str = ""
    # Where this dataset's variables came from. ``"pangeo"`` =
    # written by ``process.py``; ``"esgf"`` = written by
    # ``process_esgf.py`` without a prior Pangeo zarr; ``"pangeo+esgf"``
    # = started as a Pangeo zarr and later augmented by the ESGF
    # pipeline with variables Pangeo didn't have. See
    # ``esgf_augmented_variables`` for the per-variable list in the
    # augmented case.
    data_source: str = "pangeo"
    # Names of variables that were added by a later ESGF augment pass.
    # Always empty for ``data_source == "pangeo"``. For
    # ``data_source == "esgf"`` this lists *all* variables in the
    # dataset (the whole thing came from ESGF). For
    # ``data_source == "pangeo+esgf"`` this lists just the variables
    # ESGF added on top of the existing Pangeo zarr.
    esgf_augmented_variables: list[str] = field(default_factory=list)
    # Append-only audit log of schema migrations applied to this
    # dataset (e.g. ``[{"from": "0.1.0", "to": "0.2.0", "added": ...}]``).
    # Populated by ``migrate.py`` after each successful chain step.
    # Carried as a field on the row so the ESGF-augment path doesn't
    # silently drop the audit when it round-trips the sidecar through
    # a ``DatasetIndexRow`` instance.
    migrations: list[dict] = field(default_factory=list)


# Columns that hold list/dict values. For the flat tabular outputs
# (parquet + csv) these are JSON-encoded into strings; the JSON sidecar
# keeps them as native JSON types.
_JSON_ENCODED_FIELDS = (
    "core_zstores",
    "surface_and_ocean_zstores",
    "static_zstores",
    "regrid_methods",
    "variables_present",
    "cell_methods_mismatch",
    "warnings",
    "esgf_augmented_variables",
    "migrations",
)


def _row_to_flat_dict(row: DatasetIndexRow) -> dict:
    d = dataclasses.asdict(row)
    for k in _JSON_ENCODED_FIELDS:
        d[k] = json.dumps(d[k])
    return d


def rows_to_dataframe(rows: list[DatasetIndexRow]) -> pd.DataFrame:
    return pd.DataFrame([_row_to_flat_dict(r) for r in rows])


def write_index(rows: list[DatasetIndexRow], output_directory: str) -> None:
    """Write the central index as parquet + CSV at the root of
    ``output_directory``. CSV is always written; parquet is skipped with
    a warning if no parquet engine is installed.
    """
    df = rows_to_dataframe(rows)
    root = output_directory.rstrip("/")
    csv_path = f"{root}/index.csv"
    parquet_path = f"{root}/index.parquet"

    fs, rel = fsspec.core.url_to_fs(root)
    fs.makedirs(rel, exist_ok=True)

    df.to_csv(csv_path, index=False)
    logging.info("Wrote %d rows to %s", len(df), csv_path)

    try:
        df.to_parquet(parquet_path, index=False)
        logging.info("Wrote %d rows to %s", len(df), parquet_path)
    except Exception as e:  # noqa: BLE001
        # ImportError when pyarrow is absent; ArrowNotImplementedError
        # when pyarrow is present but built without the GCS filesystem
        # (the dev env's pyarrow ships this way). CSV is the
        # authoritative form anyway; parquet is for downstream
        # consumers that prefer typed I/O — skip silently if writing
        # fails, but log it loudly.
        logging.warning(
            "Skipped %s (%s: %s); CSV write still succeeded.",
            parquet_path,
            type(e).__name__,
            e,
        )


def write_sidecar(row: DatasetIndexRow, zarr_path: str) -> None:
    """Write a ``metadata.json`` inside a successfully written zarr.

    No-op when ``row.status != "ok"``.
    """
    if row.status != "ok":
        return
    sidecar_path = f"{zarr_path.rstrip('/')}/metadata.json"
    with fsspec.open(sidecar_path, "w") as f:
        json.dump(dataclasses.asdict(row), f, indent=2, sort_keys=True)
    logging.info("Wrote sidecar %s", sidecar_path)


def failure_record_path(output_directory: str, row: DatasetIndexRow) -> str:
    """Path to a per-dataset failure record under
    ``<output_directory>/failures/``. Kept separate from the per-zarr
    ``metadata.json`` sidecar so the presence of an ok sidecar remains
    the single source of truth for "completed". The failure record is
    the forensic trail for runs that exited without producing a zarr —
    skipped or failed — so we can tell after the fact why a dataset is
    absent from the archive (lnlqt taught us this is otherwise lost
    once pods get GC'd).
    """
    root = output_directory.rstrip("/")
    fname = f"{row.source_id}__{row.experiment}__{row.variant_label}.json"
    return f"{root}/failures/{fname}"


def write_failure_record(row: DatasetIndexRow, output_directory: str) -> None:
    """Persist a forensic record for a ``status != "ok"`` dataset.

    No-op for ok rows (those go through ``write_sidecar``). Idempotent
    — re-running on a dataset just overwrites the record with the
    latest attempt's row.
    """
    if row.status == "ok":
        return
    path = failure_record_path(output_directory, row)
    with fsspec.open(path, "w") as f:
        json.dump(dataclasses.asdict(row), f, indent=2, sort_keys=True)
    logging.info("Wrote failure record %s (status=%s)", path, row.status)


def clear_failure_record(row: DatasetIndexRow, output_directory: str) -> None:
    """Remove any prior failure record for this dataset. Called when an
    attempt succeeds (status=ok) so we don't leave a stale failure
    record next to a now-good zarr.
    """
    path = failure_record_path(output_directory, row)
    fs, rel = fsspec.core.url_to_fs(path)
    if fs.exists(rel):
        fs.rm(rel)
        logging.info("Removed stale failure record %s", path)


__all__ = [
    "DatasetIndexRow",
    "clear_failure_record",
    "failure_record_path",
    "rows_to_dataframe",
    "write_failure_record",
    "write_index",
    "write_sidecar",
]
