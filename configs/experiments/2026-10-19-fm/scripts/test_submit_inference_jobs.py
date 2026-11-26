import argparse
import json
from collections import Counter

import _agent_vm
import submit_inference_jobs as submit
import yaml
from _agent_vm import AGENT_NAME_LIMIT, GantryJob
from _inference_common import (
    INFERENCE_CONFIGS_DIR,
    SINGLE_SOURCE_ARCH_ARMS,
    V3_ARCH,
    V3_GRID_ARCHS,
    V3_GRID_RECORD_PATH,
    V3_SINGLE_SOURCE_ARCHS,
    inference_run_names,
    run_config_path,
    run_train_labels,
)
from create_fm_c96amip_jobs import LABELS as C96AMIP_LABELS
from create_fm_v3_jobs import ERA5_STORE
from generate_inference_configs import (
    EVAL_SUITE_CONFIGS,
    FAMILIES,
    V3_C3_EVAL_CONFIG,
    family_configs,
    weka_mounts,
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


def _use_real_beaker(monkeypatch):
    """Take the real-beaker path, as off an agent VM."""
    monkeypatch.setattr(submit, "on_agent_vm", lambda: False)


def test_v3_resolve_runs_uses_succeeded_recorded_experiment(monkeypatch):
    _use_real_beaker(monkeypatch)
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
    _use_real_beaker(monkeypatch)
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


def _on_agent_vm(monkeypatch, tmp_path):
    """Take the agent VM path, with an empty eval record in `tmp_path`."""
    monkeypatch.setattr(submit, "on_agent_vm", lambda: True)
    monkeypatch.setattr(submit, "EVAL_RECORD_PATH", tmp_path / "eval.json")


def _weka_entry(exp_id):
    return {
        "id": exp_id,
        "best_inference_ckpt": f"/climate-default/agents/a/{exp_id}/best.tar",
        "ema_ckpt_pattern": f"/climate-default/agents/a/{exp_id}/ema_{{epoch:04d}}.tar",
    }


def test_agent_vm_resolves_runs_from_gantry_status(monkeypatch, tmp_path):
    _on_agent_vm(monkeypatch, tmp_path)
    arch = "swin-v2.2-v3-grid"
    done, failed_last, running, unknown, unrecorded = inference_run_names(arch)[:5]
    record = tmp_path / "grid.json"
    record.write_text(
        json.dumps({run: _weka_entry(run[-12:]) for run in inference_run_names(arch)})
    )
    monkeypatch.setitem(submit.WEKA_RECORDS, arch, (record,))
    statuses = {
        done[-12:]: ("succeeded", 0),
        # Succeeded at the experiment level, but its latest job did not exit 0.
        failed_last[-12:]: ("succeeded", 1),
        running[-12:]: ("queued", None),
    }
    looked_up = []

    def experiment_job(exp_id):
        looked_up.append(exp_id)
        if exp_id not in statuses:
            return None
        status, code = statuses[exp_id]
        return GantryJob(exp_id, status, "", f"agent-alexeyy-x-{exp_id}", code)

    monkeypatch.setattr(submit, "experiment_job", experiment_job)
    args = argparse.Namespace(
        arch=arch,
        run=[done, failed_last, running, unknown],
        use_map=False,
        beaker_workspace="ai2/ace",
    )
    ready, skipped = submit.resolve_runs(args)
    assert list(ready) == [done]
    assert ready[done].checkpoint("lastema", arch).endswith("/ema_0120.tar")
    assert set(skipped) == {failed_last, running, unknown}
    assert "failed" in skipped[failed_last]
    assert "running" in skipped[running]
    assert "not found" in skipped[unknown]
    assert unrecorded[-12:] not in looked_up

    # --run selects exactly the named run.
    looked_up.clear()
    ready, skipped = submit.resolve_runs(
        argparse.Namespace(**{**vars(args), "run": [running]})
    )
    assert (ready, list(skipped), looked_up) == ({}, [running], [running[-12:]])


def _jobs(arch, run, ckpts=("bestinf", "lastema")):
    return [
        submit.Job(
            submit.agent_job_name(name), family, config, "-", "g", name, "/c", run, ckpt
        )
        for ckpt in ckpts
        for family, _, name, config in submit.select_jobs(FAMILIES, [run], ckpt, arch)
    ]


def test_agent_vm_job_names_fit_the_proxy():
    all_names: list[str] = []
    for arch in [*V3_GRID_ARCHS, *V3_SINGLE_SOURCE_ARCHS]:
        for run in inference_run_names(arch):
            jobs = _jobs(arch, run)
            assert len(jobs) == 30
            for job in jobs:
                assert len(job.name) <= AGENT_NAME_LIMIT == 80
                if len(job.wandb_name) <= AGENT_NAME_LIMIT:
                    assert job.name == job.wandb_name
                else:
                    assert job.name.startswith(submit.agent_run_name(run) + "-")
                    # The proxy lowercases experiment names.
                    assert job.name == job.name.lower()
            all_names += [job.name for job in jobs]
    # Unique across every run, family and checkpoint of every v3 arch.
    assert len(set(all_names)) == len(all_names)


def test_agent_vm_names_of_submitted_jobs_unchanged():
    record = submit.read_record(submit.EVAL_RECORD_PATH)
    assert record
    for name, entry in record.items():
        assert submit.agent_job_name(entry["wandb_name"]) == name
        assert entry["experiment"].startswith(_agent_vm.AGENT_PREFIX + name + "-")


def test_agent_vm_job_name_shortens_run_then_kind():
    run = "ace2-fm-nc-swin-v2.2-v3-fm-e3sm-temperature-c2-cellloss-s0-maskoff"
    short_run = "ace2-fm-nc-swin-v2.2-v3-fm-e3sm-temp-c2-s0-maskoff"
    assert submit.agent_run_name(run) == short_run
    assert submit.agent_job_name(f"{run}-cm4xshield-lastema") == (
        f"{short_run}-cm4xshield-lastema"
    )
    paper = f"{run}-era5-control-dataCO2-10yr-sstprescribed-eval-lastema"
    assert submit.agent_job_name(paper) == f"{short_run}-era5-ctl-10yr-lastema"
    som = f"{run}-som-abrupt-4xCO2-ens-sstslab-eval"
    assert submit.agent_job_name(som) == f"{short_run}-som-4xco2-ens"


def test_agent_vm_skips_jobs_found_under_either_name(monkeypatch, tmp_path):
    _on_agent_vm(monkeypatch, tmp_path)
    arch = "swin-v2.2-v3-grid"
    run = "ace2-fm-nc-swin-v2.2-v3-fm-e3sm-temperature-c2-cellloss-s2-maskoff"
    jobs = {job.wandb_name: job for job in _jobs(arch, run)}
    long_name = f"{run}-era5-control-dataCO2-10yr-sstprescribed-eval-lastema"
    long_job = jobs[long_name]
    short_run = submit.agent_run_name(run)
    assert long_job.name == f"{short_run}-era5-ctl-10yr-lastema"
    from_vm = jobs[f"{run}-bestinf"]
    from_vm_failed = jobs[f"{run}-lastema"]
    elsewhere = jobs[f"{run}-cm4xshield-bestinf"]
    recorded = jobs[f"{run}-sst-c96-p0k"]
    submit.write_record({recorded.name: {"id": "R"}}, submit.EVAL_RECORD_PATH)
    monkeypatch.setattr(
        _agent_vm,
        "gantry_list",
        lambda: [
            GantryJob("A", "queued", "t", f"agent-alexeyy-{from_vm.name}-ab1z"),
            GantryJob("B", "failed", "t", f"agent-alexeyy-{from_vm_failed.name}-x2yz"),
            GantryJob("C", "running", "t", f"agent-alexeyy-{long_job.name}-q9q9"),
        ],
    )
    monkeypatch.setattr(
        _agent_vm,
        "gantry_job",
        lambda exp_id: GantryJob(exp_id, "running", "t", "") if exp_id == "R" else None,
    )
    listed = []

    def beaker_listing(text, workspace):
        listed.append(text)
        # Submitted with the real gantry: full name, gantry's hex suffix.
        return [GantryJob("D", "succeeded", "t", f"{elsewhere.name}-1a2b")]

    monkeypatch.setattr(_agent_vm, "beaker_listing", beaker_listing)
    existing = submit.agent_existing_jobs(list(jobs.values()), "ai2/ace")
    assert existing == {from_vm.name, long_job.name, elsewhere.name, recorded.name}
    assert sorted(listed) == sorted([run, short_run])


def test_agent_vm_listing_at_the_cap_is_redone_per_name(monkeypatch):
    names = {"run-a-bestinf", "run-a-lastema"}
    monkeypatch.setattr(_agent_vm, "gantry_list", lambda: [])

    def beaker_listing(text, workspace):
        if text == "run-a":
            return [GantryJob(str(i), "failed", "", "other") for i in range(100)]
        return [GantryJob(text, "succeeded", "", text)]

    monkeypatch.setattr(_agent_vm, "beaker_listing", beaker_listing)
    found = _agent_vm.find_experiments({"run-a": names})
    assert {name: [job.id for job in jobs] for name, jobs in found.items()} == {
        name: [name] for name in names
    }


def test_agent_vm_gantry_command_uses_supported_flags_only():
    supported = {
        "--name", "--description", "--task-name", "--cluster", "--gpus",
        "--shared-memory", "--priority", "--min-runtime", "--beaker-image",
        "--env", "--env-secret", "--install", "--system-python", "--weka",
        "--branch", "--allow-dirty", "--dry-run",
    }  # fmt: skip
    run = "ace2-fm-nc-sfno-v3-fm-c96amip-c1-cellloss-s0-maskoff"
    checkpoint = "/climate-default/agents/alexeyy/x/training_checkpoints/best.tar"
    for job in _jobs("sfno-v3-c96amip", run, ("bestinf",)):
        job = job._replace(checkpoint=checkpoint)
        cmd = submit.agent_gantry_command(job, "c96", "normal", ["ai2/titan"], True)
        sep = cmd.index("--")
        flags = {arg for arg in cmd[2:sep] if arg.startswith("--")}
        assert flags <= supported, flags - supported
        assert cmd[:4] == [submit.GANTRY, "run", "--branch", submit.BRANCH]
        assert "WANDB_API_KEY=wandb-api-key-ai2cm-sa" in cmd
        assert f"WANDB_NAME={job.wandb_name}" in cmd
        assert cmd[cmd.index("--name") + 1] == job.name
        command = cmd[sep + 1 :]
        config = f"configs/experiments/2026-10-19-fm/inference_configs/{job.config}"
        assert config in command
        if submit.MODES[job.family] == "suite":
            assert command[-4:] == [
                "--checkpoint-path",
                checkpoint,
                "--dataset-label",
                "c96",
            ]
        else:
            assert command[-3:] == [
                "--override",
                f"checkpoint_path={checkpoint}",
                "labels=[c96]",
            ]


def test_generated_configs_name_their_weka_stores():
    for path in INFERENCE_CONFIGS_DIR.glob("*.yaml"):
        text = path.read_text()
        if not text.startswith("# Generated by generate_inference_configs.py"):
            continue
        mounts = weka_mounts(yaml.safe_load(text))
        for mount in mounts:
            assert f"# WEKA_MOUNT: {mount}\n" in text
        if path.name.startswith("ace-paper-era5-"):
            assert mounts == [ERA5_STORE]
