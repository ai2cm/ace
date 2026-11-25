import json
import sys

import eval_table
import fm_table
from _beaker_listing import OK, RUNNING, NamedExperiment
from _inference_common import (
    C96AMIP_ARCHS,
    ERA5_1979_ARCHS,
    V3_ARCH,
    V3_GRID_ARCHS,
    V3_SINGLE_SOURCE_ARCHS,
    inference_run_names,
)
from fm_table import ROWS, GantryJob


def test_tables_track_the_cohorts(monkeypatch, capsys):
    listed_names = set()
    listed_texts = []

    def list_jobs(text, names):
        listed_texts.append(text)
        # 30 jobs per run: 3 runs per single-source arch, 12 per grid arch.
        assert len(names) == (360 if text.endswith("-v3-fm-") else 90)
        assert all(name.startswith(text) for name in names)
        listed_names.update(names)
        return {}

    monkeypatch.setattr(sys, "argv", ["eval_table.py"])
    monkeypatch.setattr(eval_table, "newest_by_name", list_jobs)
    eval_table.main()

    assert listed_texts == [
        f"ace2-fm-{arch}-{arm}-"
        for arm in ("fm-c96amip", "fm-era5-1979")
        for arch in ("nc-sfno-v2", "nc-swin-v2.2-v2")
    ] + ["ace2-fm-nc-sfno-v3-fm-", "ace2-fm-nc-swin-v2.2-v3-fm-"] + [
        f"ace2-fm-{arch}-{arm}-"
        for arm in ("fm-c96amip", "fm-era5-1979")
        for arch in ("nc-sfno-v3", "nc-swin-v2.2-v3")
    ]
    training = {name for row in ROWS for name in row.names}
    assert len(training) == 12 + 24 + 12
    assert sum("-fm-era5-1979-" in name for name in training) == 12
    assert all(
        sum(name.startswith(run + "-") for name in listed_names) == 30
        for run in training
    )
    out = capsys.readouterr().out
    assert out.count("| Total |  | 0 | 0 | 0 | 0 | 180 | 180 |") == 2
    assert out.count("| Total |  | 0 | 0 | 0 | 0 | 720 | 720 |") == 1
    assert out.count("| Total |  | 0 | 0 | 0 | 0 | 360 | 360 |") == 1
    assert out.index("C96 AMIP-only") < out.index("ERA5-1979") < out.index("v3 grid")


def test_fm_table_rows_cover_the_v3_grid():
    labels = [row.label for row in ROWS]
    assert labels[:4] == [*C96AMIP_ARCHS, *ERA5_1979_ARCHS]
    assert labels[4:12] == [
        f"{base}-{arm}-{mask}"
        for base in ("sfno-v3", "swin-v2.2-v3")
        for arm in ("e3sm-balanced", "e3sm-temperature")
        for mask in ("maskon", "maskoff")
    ]
    assert labels[12:] == list(V3_SINGLE_SOURCE_ARCHS)
    assert all(len(row.names) == 3 and row.max_epochs == 120 for row in ROWS)
    grid = {name for row in ROWS[4:12] for name in row.names}
    assert grid == {
        name for arch in V3_GRID_ARCHS for name in inference_run_names(arch)
    }
    # The grid's finished sfno balanced maskoff cell is three sfno-v3 runs.
    assert set(ROWS[5].names) == {
        f"ace2-fm-nc-sfno-v3-fm-e3sm-balanced-c2-cellloss-s{seed}-maskoff"
        for seed in range(3)
    }
    assert grid & set(inference_run_names(V3_ARCH)) == set(ROWS[5].names)
    assert ROWS[12].names == (
        "ace2-fm-nc-sfno-v3-fm-c96amip-c1-cellloss-s0-maskoff",
        "ace2-fm-nc-sfno-v3-fm-c96amip-c1-cellloss-s1-maskoff",
        "ace2-fm-nc-sfno-v3-fm-c96amip-c1-cellloss-s2-maskoff",
    )


def test_fm_table_gantry_backend_reads_every_record(monkeypatch, tmp_path):
    finished = ROWS[5].names[0]
    unlisted = ROWS[5].names[1]
    grid = ROWS[4].names[0]
    v3_record = tmp_path / "v3.json"
    v3_record.write_text(
        json.dumps({finished: {"id": "A"}, unlisted: {"id": "B"}, grid: {"id": "X"}})
    )
    grid_record = tmp_path / "grid.json"
    grid_record.write_text(json.dumps({grid: {"id": "C"}}))
    listed = [
        GantryJob("A", "succeeded", "2026-11-18", f"agent-alexeyy-{finished}-abcd"),
        GantryJob("C", "queued", "2026-11-25", f"agent-alexeyy-{grid}-wxyz"),
    ]
    looked_up = []

    def gantry_job(job_id):
        looked_up.append(job_id)
        return GantryJob(job_id, "succeeded", "", f"agent-alexeyy-{unlisted}-efgh")

    monkeypatch.setattr(
        fm_table, "RECORDS", (tmp_path / "none.json", grid_record, v3_record)
    )
    monkeypatch.setattr(fm_table, "_gantry_list", lambda: listed)
    monkeypatch.setattr(fm_table, "_gantry_job", gantry_job)
    monkeypatch.setattr(fm_table, "_gantry", lambda *args: "")
    monkeypatch.setattr(fm_table, "_wandb_histories", lambda experiments: {})

    tracked, progress = fm_table.gantry_backend(ROWS[4:6])

    assert looked_up == ["B"]
    assert tracked[finished].status == OK
    assert tracked[unlisted].status == OK
    assert progress[finished][0] == 120
    assert tracked[grid].queued
    assert set(tracked) == {finished, unlisted, grid}


def test_gantry_list_reads_the_created_column(monkeypatch):
    out = (
        "01ABC  succeeded   high     2026-11-18T03:03:07+00:00  agent-alexeyy-x-3ga0\n"
    )
    monkeypatch.setattr(fm_table, "_gantry", lambda *args: out)
    [job] = fm_table._gantry_list()
    assert (job.id, job.status, job.created, job.name) == (
        "01ABC",
        "succeeded",
        "2026-11-18T03:03:07+00:00",
        "agent-alexeyy-x-3ga0",
    )


def test_eval_table_with_v3_arch_tracks_all_families(monkeypatch, capsys):
    def list_jobs(text, names):
        assert text == "ace2-fm-nc-sfno-v3-fm-"
        assert len(names) == 540
        return {}

    monkeypatch.setattr(sys, "argv", ["eval_table.py", "--arch", V3_ARCH])
    monkeypatch.setattr(eval_table, "newest_by_name", list_jobs)
    eval_table.main()

    assert "| Total |  | 0 | 0 | 0 | 0 | 540 | 540 |" in capsys.readouterr().out


def test_eval_table_tracks_v3_eval_checkpoints(monkeypatch, capsys):
    runs = inference_run_names(V3_ARCH)
    expected = {f"{run}-{ckpt}" for run in runs for ckpt in ("bestinf", "lastema")}
    running = f"{runs[0]}-bestinf"
    queued = f"{runs[0]}-lastema"

    def list_jobs(text, names):
        assert text == "ace2-fm-nc-sfno-v3-fm-"
        assert names == expected
        return {
            running: NamedExperiment(
                running, RUNNING, {"jobs": [{"status": {"started": "date"}}]}
            ),
            queued: NamedExperiment(queued, RUNNING, {"jobs": [{"status": {}}]}),
        }

    monkeypatch.setattr(
        sys, "argv", ["eval_table.py", "--arch", V3_ARCH, "--family", "eval"]
    )
    monkeypatch.setattr(eval_table, "newest_by_name", list_jobs)
    eval_table.main()

    output = capsys.readouterr().out
    assert "| eval | bestinf | 0 | 1 | 0 | 0 | 17 | 18 |" in output
    assert "| eval | lastema | 0 | 0 | 1 | 0 | 17 | 18 |" in output
    assert "| Total |  | 0 | 1 | 1 | 0 | 34 | 36 |" in output
