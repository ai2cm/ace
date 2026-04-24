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
    forcing_zstores: list[str] = field(default_factory=list)
    static_zstores: list[str] = field(default_factory=list)

    # Processing
    target_grid: str = ""
    native_grid_label: str = ""
    native_calendar: str = ""
    regrid_methods: dict[str, str] = field(default_factory=dict)
    mask_source: str = ""
    forcing_interpolation: str = ""

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


# Columns that hold list/dict values. For the flat tabular outputs
# (parquet + csv) these are JSON-encoded into strings; the JSON sidecar
# keeps them as native JSON types.
_JSON_ENCODED_FIELDS = (
    "core_zstores",
    "forcing_zstores",
    "static_zstores",
    "regrid_methods",
    "variables_present",
    "cell_methods_mismatch",
    "warnings",
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

    df.to_csv(csv_path, index=False)
    logging.info("Wrote %d rows to %s", len(df), csv_path)

    try:
        df.to_parquet(parquet_path, index=False)
        logging.info("Wrote %d rows to %s", len(df), parquet_path)
    except ImportError as e:
        logging.warning(
            "Skipped %s (%s). Install pyarrow to enable parquet output.",
            parquet_path,
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


__all__ = [
    "DatasetIndexRow",
    "rows_to_dataframe",
    "write_index",
    "write_sidecar",
]
