import sys

import eval_table
from _beaker_listing import RUNNING, NamedExperiment
from _inference_common import C96AMIP_ARCHS, V3_ARCH, inference_run_names
from fm_table import ROWS


def test_tables_track_the_six_c96amip_runs(monkeypatch, capsys):
    listed_names = set()

    def list_jobs(text, names):
        assert "-fm-c96amip-" in text
        assert len(names) == 90
        assert all(name.startswith(text) for name in names)
        listed_names.update(names)
        return {}

    monkeypatch.setattr(sys, "argv", ["eval_table.py"])
    monkeypatch.setattr(eval_table, "newest_by_name", list_jobs)
    eval_table.main()

    assert {row.label for row in ROWS} == C96AMIP_ARCHS.keys()
    training = {name for row in ROWS for name in row.names}
    assert len(training) == 6
    assert all(
        sum(name.startswith(run + "-") for name in listed_names) == 30
        for run in training
    )
    assert "| Total |  | 0 | 0 | 0 | 0 | 180 | 180 |" in capsys.readouterr().out


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
