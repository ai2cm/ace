"""Targeted tests for process.py — sidecar scan and related helpers.

The full process_one path requires network access and real zarr stores;
these tests cover the unit-testable bits that have been root causes of
recent production failures.
"""

import json
from pathlib import Path

from index import DatasetIndexRow, write_sidecar
from process import _scan_all_sidecars


def _ok_row(source_id: str = "CanESM5") -> DatasetIndexRow:
    return DatasetIndexRow(
        source_id=source_id,
        experiment="historical",
        variant_label="r1i1p1f1",
        variant_r=1,
        variant_i=1,
        variant_p=1,
        variant_f=1,
        label=f"{source_id}.p1",
        target_grid="F22.5",
        status="ok",
        output_zarr=f"file:///{source_id}/data.zarr",
    )


def test_scan_all_sidecars_finds_dataset_sidecars(tmp_path: Path):
    """Standard sidecars under <model>/<exp>/<variant>/data.zarr load fine."""
    for source_id in ("CanESM5", "GFDL-CM4"):
        d = tmp_path / source_id / "historical" / "r1i1p1f1" / "data.zarr"
        d.mkdir(parents=True)
        write_sidecar(_ok_row(source_id), str(d))

    rows = _scan_all_sidecars(str(tmp_path))
    assert {r.source_id for r in rows} == {"CanESM5", "GFDL-CM4"}


def test_scan_all_sidecars_skips_external_forcings(tmp_path: Path):
    """Regression: ``external_forcings.py`` drops its own ``metadata.json``
    sidecar next to each per-scenario zarr. Those payloads don't have
    ``source_id`` / ``variant_label`` and would crash
    ``DatasetIndexRow(**filtered)`` — the scan must skip them.
    """
    # A normal dataset sidecar.
    d = tmp_path / "CanESM5" / "historical" / "r1i1p1f1" / "data.zarr"
    d.mkdir(parents=True)
    write_sidecar(_ok_row(), str(d))

    # An external-forcings sidecar that lacks source_id / variant_label.
    ef = tmp_path / "external_forcings" / "historical.zarr"
    ef.mkdir(parents=True)
    (ef / "metadata.json").write_text(
        json.dumps(
            {
                "experiment": "historical",
                "co2_url": "https://example.org/co2.csv",
                "staged_at": "2026-05-22T00:00:00Z",
            }
        )
    )

    rows = _scan_all_sidecars(str(tmp_path))
    # Only the dataset sidecar — the external-forcings one was skipped.
    assert len(rows) == 1
    assert rows[0].source_id == "CanESM5"


def test_scan_all_sidecars_empty_directory(tmp_path: Path):
    assert _scan_all_sidecars(str(tmp_path)) == []


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_scan_all_sidecars_finds_dataset_sidecars(tmp / "a")
        print("ok test_scan_all_sidecars_finds_dataset_sidecars")
        test_scan_all_sidecars_skips_external_forcings(tmp / "b")
        print("ok test_scan_all_sidecars_skips_external_forcings")
        test_scan_all_sidecars_empty_directory(tmp / "c")
        print("ok test_scan_all_sidecars_empty_directory")
    print("all tests passed")
