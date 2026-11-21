import json
from collections import Counter

import submit_inference_jobs as submit
from _inference_common import (
    INFERENCE_CONFIGS_DIR,
    V3_ARCH,
    inference_run_names,
    run_train_labels,
)
from generate_inference_configs import (
    EVAL_SUITE_CONFIGS,
    FAMILIES,
    V3_C3_EVAL_CONFIG,
    family_configs,
)
from run_eval_suite import load_suite
from submit_fm_v3_jobs import RECORD_PATH as V3_RECORD_PATH


def test_v3_job_selection():
    runs = inference_run_names(V3_ARCH)
    assert len(runs) == 18
    for ckpt in ("bestinf", "lastema"):
        jobs = submit.select_jobs(["eval"], runs, ckpt, V3_ARCH)
        assert len(jobs) == len(runs)
        assert {name for _, _, name, _ in jobs} == {f"{run}-{ckpt}" for run in runs}
        for _, run, _, config in jobs:
            expected_config = (
                V3_C3_EVAL_CONFIG if "-c3-" in run else EVAL_SUITE_CONFIGS["eval"]
            )
            assert config == expected_config


def test_v3_eval_suite_checkpoint_and_labels():
    for variant, config_file in (
        ("c1", EVAL_SUITE_CONFIGS["eval"]),
        ("c2", EVAL_SUITE_CONFIGS["eval"]),
        ("c3", V3_C3_EVAL_CONFIG),
    ):
        path = str(INFERENCE_CONFIGS_DIR / config_file)
        checkpoint = (
            "/climate-default/agents/test/training_checkpoints/ema_ckpt_0120.tar"
        )
        suite = load_suite(path, checkpoint_path=checkpoint)
        labels = set()
        for entry in suite.inferences:
            assert entry.config.checkpoint_path == checkpoint
            labels.update(entry.config_data["loader"]["dataset"]["labels"])
        assert labels == {"era5", "amip" if variant == "c3" else "c96"}
        for run in inference_run_names(V3_ARCH):
            if f"-{variant}-" in run:
                assert labels <= run_train_labels(run)


def test_v3_remaining_families_use_training_labels():
    runs = inference_run_names(V3_ARCH)
    families = [family for family in FAMILIES if family != "eval"]
    configs = {
        name: cfg for family in families for name, cfg in family_configs(family).items()
    }
    train_labels = {run: run_train_labels(run) for run in runs}
    jobs = [
        job
        for ckpt in ("bestinf", "lastema")
        for job in submit.select_jobs(families, runs, ckpt, V3_ARCH)
    ]
    assert len({name for _, _, name, _ in jobs}) == 504
    assert Counter(family for family, _, _, _ in jobs) == {
        "cm4xshield": 36,
        "cm4-1pct": 36,
        "sst": 216,
        "paper-som": 72,
        "paper-era5": 144,
    }
    for family, run, _, filename in jobs:
        config = configs[filename]
        labels = (
            {
                label
                for entry in config["inferences"]
                for label in entry["config"]["loader"]["dataset"]["labels"]
            }
            if family in ("cm4xshield", "cm4-1pct")
            else set(config["labels"])
        )
        assert labels and labels <= train_labels[run]


def test_v3_resolve_runs_uses_succeeded_recorded_experiment(monkeypatch):
    succeeded, failed, missing = inference_run_names(V3_ARCH)[:3]
    records = json.loads(V3_RECORD_PATH.read_text())
    monkeypatch.setattr(
        submit,
        "list_experiments",
        lambda *_: [
            {
                "id": records[run]["id"],
                "jobs": [{"status": {"exited": "date", "exitCode": exit_code}}],
            }
            for run, exit_code in [(succeeded, 0), (failed, 1)]
        ],
    )
    ready, skipped = submit._resolve_v3_runs([succeeded, failed, missing], "ai2/ace")
    assert list(ready) == [succeeded]
    assert set(skipped) == {failed, missing}
    output = ready[succeeded]
    assert (
        output.checkpoint("bestinf", V3_ARCH)
        == records[succeeded]["best_inference_ckpt"]
    )
    assert output.checkpoint("lastema", V3_ARCH) == records[succeeded][
        "ema_ckpt_pattern"
    ].format(epoch=120)
