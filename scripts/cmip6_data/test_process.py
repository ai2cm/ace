"""Targeted tests for process.py — sidecar scan and related helpers.

The full process_one path requires network access and real zarr stores;
these tests cover the unit-testable bits that have been root causes of
recent production failures.
"""

import json
from pathlib import Path

import pandas as pd
from config import ProcessConfig
from index import DatasetIndexRow, write_sidecar
from process import _scan_all_sidecars, select_datasets


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


# ---------------------------------------------------------------------------
# Planning-time selection: drop tasks that lack enough core day variables
# so we don't burn a pod per dead-on-arrival dataset (lnlqt / 4tbbk lesson).
# ---------------------------------------------------------------------------


def _inventory_row(
    source_id: str,
    experiment: str,
    member: str,
    variable: str,
    table: str = "day",
) -> dict:
    parts = member.lstrip("r").split("i")
    r = int(parts[0])
    rest = parts[1].split("p")
    i = int(rest[0])
    rest2 = rest[1].split("f")
    p = int(rest2[0])
    f = int(rest2[1])
    return {
        "table_id": table,
        "source_id": source_id,
        "experiment_id": experiment,
        "member_id": member,
        "variant_r": r,
        "variant_i": i,
        "variant_p": p,
        "variant_f": f,
        "variable_id": variable,
        "grid_label": "gn",
        "zstore": f"gs://x/{source_id}/{experiment}/{member}/{table}/{variable}",
    }


def _make_config(
    max_core_missing: int = 3, core_vars: list[str] | None = None
) -> ProcessConfig:
    cfg = ProcessConfig(inventory_path="x", output_directory="y")
    cfg.defaults.max_core_missing = max_core_missing
    if core_vars is not None:
        cfg.defaults.core_variables = core_vars
    cfg.selection.experiments = ["historical"]
    cfg.selection.require_i = None
    cfg.selection.max_members_per_f = None
    return cfg


def test_select_datasets_drops_tasks_missing_too_many_core_vars():
    # 4 core variables required; max_core_missing=1 → datasets with
    # fewer than 3 cores are dropped at planning time.
    core_vars = ["ua", "va", "hus", "zg"]
    rows = []
    # "good": all 4 core vars → kept
    for v in core_vars:
        rows.append(_inventory_row("Good", "historical", "r1i1p1f1", v))
    # "bad": only 1 core var → 3 missing > 1 → dropped
    rows.append(_inventory_row("Bad", "historical", "r1i1p1f1", "zg"))
    # "borderline": 3 of 4 cores → 1 missing == max_core_missing → kept
    for v in ["ua", "va", "hus"]:
        rows.append(_inventory_row("Borderline", "historical", "r1i1p1f1", v))

    inventory = pd.DataFrame(rows)
    cfg = _make_config(max_core_missing=1, core_vars=core_vars)
    tasks = select_datasets(inventory, cfg)
    surviving_sources = sorted(t.source_id for t in tasks)
    assert surviving_sources == ["Borderline", "Good"]


def test_select_datasets_keeps_when_within_threshold():
    # Default max_core_missing=3 → a dataset with 5 of 8 cores passes
    # (3 missing == max_core_missing, not > max_core_missing).
    core_vars = ["ua", "va", "hus", "zg", "tas", "huss", "psl", "pr"]
    rows = [
        _inventory_row("OnlyFive", "historical", "r1i1p1f1", v) for v in core_vars[:5]
    ]
    inventory = pd.DataFrame(rows)
    cfg = _make_config(max_core_missing=3, core_vars=core_vars)
    tasks = select_datasets(inventory, cfg)
    assert len(tasks) == 1
    assert tasks[0].source_id == "OnlyFive"


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
