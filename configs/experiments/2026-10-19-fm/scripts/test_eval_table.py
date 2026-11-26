import json
import sys

import _agent_vm
import eval_table
import fm_table
import submit_inference_jobs
from _agent_vm import GantryJob
from _beaker_listing import OK, RUNNING, NamedExperiment
from _inference_common import (
    C96AMIP_ARCHS,
    ERA5_1979_ARCHS,
    V3_ARCH,
    V3_GRID_ARCHS,
    V3_SINGLE_SOURCE_ARCHS,
    inference_run_names,
)
from fm_table import ROWS


def _use_real_beaker(monkeypatch):
    """Take the real-beaker path, as off an agent VM."""
    monkeypatch.setattr(eval_table, "on_agent_vm", lambda: False)
    monkeypatch.setattr(submit_inference_jobs, "on_agent_vm", lambda: False)


def test_tables_track_the_cohorts(monkeypatch, capsys):
    _use_real_beaker(monkeypatch)
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
    monkeypatch.setattr(_agent_vm, "gantry", lambda *args: out)
    [job] = _agent_vm.gantry_list()
    assert (job.id, job.status, job.created, job.name) == (
        "01ABC",
        "succeeded",
        "2026-11-18T03:03:07+00:00",
        "agent-alexeyy-x-3ga0",
    )


def test_eval_table_with_v3_arch_tracks_all_families(monkeypatch, capsys):
    _use_real_beaker(monkeypatch)

    def list_jobs(text, names):
        assert text == "ace2-fm-nc-sfno-v3-fm-"
        assert len(names) == 540
        return {}

    monkeypatch.setattr(sys, "argv", ["eval_table.py", "--arch", V3_ARCH])
    monkeypatch.setattr(eval_table, "newest_by_name", list_jobs)
    eval_table.main()

    assert "| Total |  | 0 | 0 | 0 | 0 | 540 | 540 |" in capsys.readouterr().out


def test_eval_table_tracks_v3_eval_checkpoints(monkeypatch, capsys):
    _use_real_beaker(monkeypatch)
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


def test_eval_table_on_agent_vm(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(eval_table, "on_agent_vm", lambda: True)
    monkeypatch.setattr(submit_inference_jobs, "on_agent_vm", lambda: True)
    monkeypatch.setattr(eval_table, "EVAL_RECORD_PATH", tmp_path / "none.json")
    arch = "swin-v2.2-v3-grid"
    run = next(r for r in inference_run_names(arch) if "temperature" in r)
    paper = f"{run}-som-control-dataCO2-ens-sstslab-eval-lastema"
    short_paper = submit_inference_jobs.agent_job_name(paper)
    assert short_paper != paper
    found = {
        # Submitted elsewhere, then resubmitted from the VM: the newest counts.
        f"{run}-bestinf": [
            GantryJob("A", "failed", "2026-11-26", f"{run}-bestinf"),
            GantryJob(
                "B", "running", "2026-11-27", f"agent-alexeyy-{run}-bestinf-ab12"
            ),
        ],
        f"{run}-lastema": [GantryJob("C", "running", "t", "x")],
        short_paper: [GantryJob("D", "succeeded", "t", "x")],
        f"{run}-cm4xshield-bestinf": [GantryJob("E", "canceled", "t", "x")],
    }
    groups = []

    def find_experiments(names_by_text, recorded):
        groups.append(names_by_text)
        return found

    monkeypatch.setattr(eval_table, "find_experiments", find_experiments)
    monkeypatch.setattr(eval_table, "gantry_started", lambda job: job.id == "B")
    monkeypatch.setattr(sys, "argv", ["eval_table.py", "--arch", arch])
    eval_table.main()

    [names_by_text] = groups
    runs = inference_run_names(arch)
    short_runs = [submit_inference_jobs.agent_run_name(r) for r in runs]
    assert set(names_by_text) == {*runs, *short_runs}
    assert short_paper in names_by_text[submit_inference_jobs.agent_run_name(run)]
    out = capsys.readouterr().out
    assert "| eval | bestinf | 0 | 1 | 0 | 0 | 11 | 12 |" in out
    assert "| eval | lastema | 0 | 0 | 1 | 0 | 11 | 12 |" in out
    assert "| paper-som | lastema | 1 | 0 | 0 | 0 | 23 | 24 |" in out
    assert "| Total |  | 1 | 1 | 1 | 1 | 356 | 360 |" in out
    # Found under its real-gantry (full) name, listed under its VM name.
    cm4xshield = submit_inference_jobs.agent_job_name(f"{run}-cm4xshield-bestinf")
    assert f"{cm4xshield}  E  (canceled)" in out
