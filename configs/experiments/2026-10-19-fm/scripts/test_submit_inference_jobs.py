import json
from collections import Counter

import submit_inference_jobs as submit
from _inference_common import (
    INFERENCE_CONFIGS_DIR,
    SINGLE_SOURCE_ARCH_ARMS,
    V3_ARCH,
    V3_GRID_ARCHS,
    V3_GRID_RECORD_PATH,
    inference_run_names,
    run_config_path,
    run_train_labels,
)
from create_fm_c96amip_jobs import LABELS as C96AMIP_LABELS
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
    ready, skipped = submit._resolve_weka_runs(
        [succeeded, failed, missing], "ai2/ace", V3_ARCH
    )
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


def test_single_source_suites_use_the_trained_label():
    labels = {
        arch: C96AMIP_LABELS[arm] for arch, arm in SINGLE_SOURCE_ARCH_ARMS.items()
    }
    assert labels == {
        "sfno-v2-c96amip": "c96",
        "swin-v2.2-v2-c96amip": "c96",
        "sfno-v2-era5-1979": "era5",
        "swin-v2.2-v2-era5-1979": "era5",
        "sfno-v3-c96amip": "c96",
        "swin-v2.2-v3-c96amip": "c96",
        "sfno-v3-era5-1979": "era5",
        "swin-v2.2-v3-era5-1979": "era5",
    }
    for arch, label in labels.items():
        run = inference_run_names(arch)[0]
        # The v3 grid's configs may not be written yet.
        if run_config_path(run).exists():
            assert run_train_labels(run) == {label}
        for family in ("eval", "cm4xshield", "cm4-1pct"):
            [(name, filename)] = submit.family_jobs(family, run, arch=arch)
            suite = load_suite(
                str(INFERENCE_CONFIGS_DIR / filename), dataset_label=label
            )
            for entry in suite.inferences:
                assert entry.config.labels == [label]
                assert entry.config_data["loader"]["dataset"]["labels"] == [label]


def test_era5_1979_jobs_have_their_own_wandb_groups():
    for arch in ("sfno-v2-era5-1979", "swin-v2.2-v2-era5-1979"):
        assert len(inference_run_names(arch)) == 3
        for ckpt in ("bestinf", "lastema"):
            for group in submit.wandb_groups(ckpt, arch).values():
                assert "era5-1979-2026-11-18" in group
                assert "c96amip" not in group


def test_v3_grid_archs_resolve_their_runs_from_both_records(monkeypatch, tmp_path):
    v3_runs = set(inference_run_names(V3_ARCH))
    assert len(v3_runs) == 18
    for arch in V3_GRID_ARCHS:
        runs = inference_run_names(arch)
        assert len(runs) == 12
        # All 5-source runs get the in-sample CM4/X-SHiELD suites (v2 variant).
        jobs = submit.select_jobs(["cm4xshield"], runs, "bestinf", arch)
        assert {config for *_, config in jobs} == {
            "ace-eval-suite-config-4deg-cm4-xshield-v2.yaml"
        }
        assert len(jobs) == 12
        for ckpt in ("bestinf", "lastema"):
            for group in submit.wandb_groups(ckpt, arch).values():
                assert "v3-grid-2026-11-25" in group
    sfno_grid = inference_run_names("sfno-v3-grid")
    shared = [run for run in sfno_grid if run in v3_runs]
    assert shared == [
        f"ace2-fm-nc-sfno-v3-fm-e3sm-balanced-c2-cellloss-s{seed}-maskoff"
        for seed in range(3)
    ]
    assert submit.WEKA_RECORDS["sfno-v3-grid"] == (V3_GRID_RECORD_PATH, V3_RECORD_PATH)

    grid_record = tmp_path / "grid.json"
    v3_record = tmp_path / "v3.json"
    new = sfno_grid[0]

    def entry(exp_id):
        return {
            "id": exp_id,
            "best_inference_ckpt": f"/w/{exp_id}/best.tar",
            "ema_ckpt_pattern": f"/w/{exp_id}/ema_{{epoch:04d}}.tar",
        }

    grid_record.write_text(json.dumps({new: entry("N")}))
    v3_record.write_text(json.dumps({shared[0]: entry("S")}))
    monkeypatch.setitem(submit.WEKA_RECORDS, "sfno-v3-grid", (grid_record, v3_record))
    succeeded = {"jobs": [{"status": {"exited": "t", "exitCode": 0}}]}
    monkeypatch.setattr(
        submit,
        "list_experiments",
        lambda workspace, text: [{"id": "N", **succeeded}, {"id": "S", **succeeded}],
    )
    ready, skipped = submit._resolve_weka_runs(sfno_grid, "ai2/ace", "sfno-v3-grid")
    assert set(ready) == {new, shared[0]}
    assert ready[shared[0]].checkpoint("lastema", "sfno-v3-grid") == "/w/S/ema_0120.tar"
    assert skipped == {
        run: "not in grid.json / v3.json" for run in sfno_grid if run not in ready
    }
