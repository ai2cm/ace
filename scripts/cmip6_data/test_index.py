"""Minimal tests for index.py — runnable standalone."""

import json
import tempfile
from pathlib import Path

import pandas as pd
from index import (
    DatasetIndexRow,
    clear_failure_record,
    failure_record_path,
    rows_to_dataframe,
    write_failure_record,
    write_index,
    write_sidecar,
)


def _example_row(status: str = "ok", suffix: str = "") -> DatasetIndexRow:
    return DatasetIndexRow(
        source_id="CanESM5",
        experiment="historical",
        variant_label="r1i1p1f1",
        variant_r=1,
        variant_i=1,
        variant_p=1,
        variant_f=1,
        label="CanESM5.p1",
        core_zstores=["gs://a/ua.zarr", "gs://a/va.zarr"],
        regrid_methods={"ua": "bilinear", "pr": "conservative"},
        target_grid="F22.5",
        status=status,
        output_zarr=f"gs://out/CanESM5{suffix}.zarr" if status == "ok" else "",
        skip_reason="" if status == "ok" else "missing core variable ua",
    )


def test_rows_to_dataframe_encodes_lists_as_json_strings():
    df = rows_to_dataframe([_example_row()])
    assert df.loc[0, "core_zstores"] == json.dumps(["gs://a/ua.zarr", "gs://a/va.zarr"])
    assert df.loc[0, "regrid_methods"] == json.dumps(
        {"ua": "bilinear", "pr": "conservative"}
    )
    # Scalar fields stay as scalars
    assert df.loc[0, "label"] == "CanESM5.p1"
    assert df.loc[0, "variant_p"] == 1


def test_write_index_writes_csv_always(tmp_path: Path):
    rows = [_example_row(), _example_row(status="skipped")]
    write_index(rows, str(tmp_path))
    csv_path = tmp_path / "index.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert set(df["status"]) == {"ok", "skipped"}


def test_write_sidecar_only_for_ok(tmp_path: Path):
    zarr_dir = tmp_path / "data.zarr"
    zarr_dir.mkdir()
    # ok row -> sidecar written
    write_sidecar(_example_row(), str(zarr_dir))
    sidecar = zarr_dir / "metadata.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["label"] == "CanESM5.p1"
    assert payload["core_zstores"] == ["gs://a/ua.zarr", "gs://a/va.zarr"]

    # failed row -> no sidecar
    zarr_dir2 = tmp_path / "failed.zarr"
    zarr_dir2.mkdir()
    write_sidecar(_example_row(status="failed"), str(zarr_dir2))
    assert not (zarr_dir2 / "metadata.json").exists()


def test_failure_record_path_uses_failures_subdir():
    row = _example_row(status="failed")
    p = failure_record_path("gs://bucket/v2", row)
    assert p == "gs://bucket/v2/failures/CanESM5__historical__r1i1p1f1.json"


def test_write_failure_record_persists_non_ok(tmp_path: Path):
    out = tmp_path / "v2"
    out.mkdir()
    # Skipped row → record written.
    skipped = _example_row(status="skipped")
    write_failure_record(skipped, str(out))
    rec = out / "failures" / "CanESM5__historical__r1i1p1f1.json"
    assert rec.exists()
    payload = json.loads(rec.read_text())
    assert payload["status"] == "skipped"
    assert payload["skip_reason"] == "missing core variable ua"

    # Failed row overwrites the same path with the new attempt.
    failed = _example_row(status="failed")
    write_failure_record(failed, str(out))
    payload = json.loads(rec.read_text())
    assert payload["status"] == "failed"


def test_write_failure_record_noop_for_ok(tmp_path: Path):
    out = tmp_path / "v2"
    out.mkdir()
    write_failure_record(_example_row(status="ok"), str(out))
    assert not (out / "failures").exists()


def test_clear_failure_record_removes_stale_entry(tmp_path: Path):
    out = tmp_path / "v2"
    out.mkdir()
    # Seed a prior failure.
    write_failure_record(_example_row(status="failed"), str(out))
    rec = out / "failures" / "CanESM5__historical__r1i1p1f1.json"
    assert rec.exists()
    # New attempt succeeds — should remove the prior record.
    clear_failure_record(_example_row(status="ok"), str(out))
    assert not rec.exists()


def test_clear_failure_record_noop_when_missing(tmp_path: Path):
    out = tmp_path / "v2"
    out.mkdir()
    # No prior record present — must not raise.
    clear_failure_record(_example_row(status="ok"), str(out))


def test_data_source_defaults_to_pangeo():
    row = DatasetIndexRow(
        source_id="X", experiment="historical", variant_label="r1i1p1f1"
    )
    assert row.data_source == "pangeo"
    assert row.esgf_augmented_variables == []


def test_esgf_augmented_variables_json_round_trip(tmp_path: Path):
    """The audit list must survive the sidecar write/read round-trip and
    the flat (csv/parquet) JSON encoding used by the index."""
    row = _example_row()
    row.data_source = "pangeo+esgf"
    row.esgf_augmented_variables = ["siday_sea_ice_fraction", "omon_tob"]
    zarr_dir = tmp_path / "data.zarr"
    zarr_dir.mkdir()
    write_sidecar(row, str(zarr_dir))
    payload = json.loads((zarr_dir / "metadata.json").read_text())
    assert payload["data_source"] == "pangeo+esgf"
    assert payload["esgf_augmented_variables"] == [
        "siday_sea_ice_fraction",
        "omon_tob",
    ]

    # Flat encoding for the central index.
    df = rows_to_dataframe([row])
    assert df.loc[0, "data_source"] == "pangeo+esgf"
    assert df.loc[0, "esgf_augmented_variables"] == json.dumps(
        ["siday_sea_ice_fraction", "omon_tob"]
    )


if __name__ == "__main__":
    # Simple standalone runner for machines without pytest.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_rows_to_dataframe_encodes_lists_as_json_strings()
        print("ok test_rows_to_dataframe_encodes_lists_as_json_strings")
        test_write_index_writes_csv_always(tmp)
        print("ok test_write_index_writes_csv_always")
        test_write_sidecar_only_for_ok(tmp)
        print("ok test_write_sidecar_only_for_ok")
    print("all tests passed")
