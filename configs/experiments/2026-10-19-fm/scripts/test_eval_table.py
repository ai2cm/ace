import sys

import eval_table
from _beaker_listing import RUNNING, NamedExperiment
from _inference_common import V3_ARCH, inference_run_names


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
