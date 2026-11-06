"""Generate the 2026-11-06 FM sfno-v3 run configs (wandb group ace2-fm-2026-11-06).

10 runs: 2 data arms x 5 seeds, input masking off. Each v3 run forks the
matching nc-sfno-v2 run (create_fm_v2_jobs.py, the "parent") at epoch 40 and
trains the rest of the parent's schedule with a cheaper multi-step recipe:

    parent ace2-fm-nc-sfno-v2-fm-{balanced,e3sm-balanced}-s{k}-maskoff
    child  ace2-fm-nc-sfno-v3-fm-{balanced,e3sm-balanced}-s{k}-maskoff

The child config is the parent run config with only these changes (data,
label sampling, model, batch size, EMA decay, validation, the weight-1
aimip_checkpoint_era5 selection entry and logging are copied verbatim):

- `stepper_training.parameter_init.weights_path: /parent_ckpt.tar`, the
  parent's `training_checkpoints/ema_ckpt_0040.tar` (EMA weights after epoch
  40), mounted from the parent's Beaker result dataset by run-ace-anneal.sh
  (submit_fm_v3_jobs.py finds the dataset). As for the anneal runs
  (create_anneal_jobs.py), parameter init loads stepper weights only: a fresh
  optimizer, an EMA re-initialised from the loaded weights, epoch counter 0.
- `max_epochs: 60` (child epoch e is parent epoch 40 + e; the parent runs 100).
- LR: the parent's remaining schedule shifted by 40 epochs: 1e-4 constant for
  epochs 1-30, then linear to 0 over epochs 31-60 (ConstantLR total_iters 30,
  LinearLR 1 -> 0 total_iters 30, milestones [30], stepped per epoch).
- `stepper_training.n_forward_steps`: 1, 2 or 4 steps drawn per batch with
  probability .6/.2/.2 from the first epoch, with the loss on the last step
  only (`optimize_last_step_only: true`); create_anneal_jobs.MULTI_STEP. This
  replaces the parent's 1 -> 2 (epoch 41) -> 4 (epoch 71) step schedule with
  the loss on every step.
- EMA checkpoints every 10 epochs on the 10-epoch boundaries,
  `ema_checkpoint_save_epochs: {start: 10, step: 10}`: epochs 10, 20, ..., 60
  (the slice runs over range(max_epochs + 1), so its index is the 1-based
  epoch number), i.e. parent-equivalent epochs 50, 60, ..., 100.
- The every-10-epoch inline inference entries (10year_oosample_varying_co2_era5,
  long_46year_varying_co2_era5) run on the same epochs as the EMA checkpoints,
  `epochs: {start: 9, step: 10}`: the inline inference slice runs over the list
  of epochs 1..max_epochs, so index 9 is epoch 10, giving epochs 10, 20, ...,
  60. (The parent's {start: 0, step: 10} runs them on epochs 1, 11, ..., 91,
  one epoch after each of its EMA checkpoints 10, 20, ..., 100.)
- `evaluate_before_training: true`: validation and every inline inference
  entry also run at epoch 0 on the loaded parent ema_ckpt_0040 weights, so the
  fork point (parent-equivalent epoch 40) has the same diagnostics.

Smoke config (`--smoke`) goes to run_configs/v3-smoke/, which no submit script
globs except submit_fm_v3_jobs.py --smoke: the s0 fm-e3sm-balanced config cut
to 3 epochs (LR constant for 2, linear for 1), 256 training samples per epoch,
EMA checkpoints and every inline inference entry every epoch (heavy ones
shortened to 365 steps), checkpoint_every_n_batches 100.

The configs are written by a YAML round trip of the parents (themselves
generated), so no hand-written comments are lost. Each file starts with a
`# KEY: value` header (create_anneal_jobs.read_header) naming the job, the
parent job and the parent checkpoint.

Usage:
    python create_fm_v3_jobs.py [--smoke]
"""

import argparse
import copy
import pathlib

import yaml
from create_anneal_jobs import MOUNTED_PARENT_CHECKPOINT, MULTI_STEP

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
SMOKE_CONFIGS_DIR = RUN_CONFIGS_DIR / "v3-smoke"
CONFIG_PREFIX = "ace-train-config-4deg-"
JOB_PREFIX = "ace2-fm-"

ARCH = "nc-sfno-v3"
PARENT_ARCH = "nc-sfno-v2"
ARMS = ("fm-balanced", "fm-e3sm-balanced")
SEEDS = (0, 1, 2, 3, 4)
SMOKE_ARM = "fm-e3sm-balanced"
SMOKE_SEED = 0

FORK_EPOCH = 40
PARENT_CHECKPOINT = f"training_checkpoints/ema_ckpt_{FORK_EPOCH:04d}.tar"
MAX_EPOCHS = 60
CONSTANT_EPOCHS = 30
DECAY_EPOCHS = MAX_EPOCHS - CONSTANT_EPOCHS
SAVE_EVERY = 10
# Same epochs, two indexing conventions (see the module docstring).
EMA_CHECKPOINT_SAVE_EPOCHS = {"start": SAVE_EVERY, "step": SAVE_EVERY}
SPARSE_INFERENCE_EPOCHS = {"start": SAVE_EVERY - 1, "step": SAVE_EVERY}
SELECTION_ENTRY = "aimip_checkpoint_era5"

SMOKE_MAX_EPOCHS = 3
SMOKE_INFERENCE_STEPS = 365


def parent_job_name(arm: str, seed: int) -> str:
    return f"{JOB_PREFIX}{PARENT_ARCH}-{arm}-s{seed}-maskoff"


def parent_config_path(arm: str, seed: int) -> pathlib.Path:
    return RUN_CONFIGS_DIR / f"{CONFIG_PREFIX}{PARENT_ARCH}-{arm}-s{seed}-maskoff.yaml"


def run_config_name(arm: str, seed: int, smoke: bool = False) -> str:
    suffix = "-smoke" if smoke else ""
    return f"{CONFIG_PREFIX}{ARCH}-{arm}-s{seed}-maskoff{suffix}.yaml"


def run_config_path(arm: str, seed: int, smoke: bool = False) -> pathlib.Path:
    directory = SMOKE_CONFIGS_DIR if smoke else RUN_CONFIGS_DIR
    return directory / run_config_name(arm, seed, smoke)


def job_name(arm: str, seed: int, smoke: bool = False) -> str:
    # Same as submit_fm_jobs.config_to_job_name(run_config_name(...)).
    return f"{JOB_PREFIX}{ARCH}-{arm}-s{seed}-maskoff{'-smoke' if smoke else ''}"


def lr_scheduler(constant_epochs: int, decay_epochs: int) -> dict:
    def sched(type_, **kwargs):
        return {"type": type_, "kwargs": kwargs, "step_each_iteration": False}

    return {
        "schedulers": [
            sched("ConstantLR", factor=1.0, total_iters=constant_epochs),
            sched(
                "LinearLR", start_factor=1.0, end_factor=0.0, total_iters=decay_epochs
            ),
        ],
        "milestones": [constant_epochs],
    }


def make_config(parent: dict) -> dict:
    cfg = copy.deepcopy(parent)
    assert "resume_results" not in cfg
    assert cfg["max_epochs"] == 100
    assert cfg["ema_checkpoint_save_epochs"] == {"start": 10, "step": 10}
    assert "evaluate_before_training" not in cfg
    cfg["max_epochs"] = MAX_EPOCHS
    cfg["ema_checkpoint_save_epochs"] = dict(EMA_CHECKPOINT_SAVE_EPOCHS)
    cfg["evaluate_before_training"] = True

    for entry in cfg["inference"]:
        if entry["name"] == SELECTION_ENTRY:
            assert entry["weight"] == 1.0
            assert entry["epochs"] == {"start": 0, "step": 1}
        else:
            assert entry["weight"] == 0.0, entry["name"]
            assert entry["epochs"] == {"start": 0, "step": 10}, entry["name"]
            entry["epochs"] = dict(SPARSE_INFERENCE_EPOCHS)

    # The parent's schedule: ConstantLR for 70 epochs, then LinearLR 1 -> 0
    # over 30. Forked at epoch 40, 30 constant epochs remain.
    scheduler = cfg["optimization"]["scheduler"]
    assert scheduler["milestones"] == [70]
    assert [s["type"] for s in scheduler["schedulers"]] == ["ConstantLR", "LinearLR"]
    cfg["optimization"]["scheduler"] = lr_scheduler(CONSTANT_EPOCHS, DECAY_EPOCHS)

    training = cfg["stepper_training"]
    assert "parameter_init" not in training
    assert training["n_forward_steps"]["milestones"][0]["epoch"] == FORK_EPOCH + 1
    training["parameter_init"] = {"weights_path": MOUNTED_PARENT_CHECKPOINT}
    training.update(copy.deepcopy(MULTI_STEP))
    return cfg


def make_smoke_config(parent: dict) -> dict:
    cfg = make_config(parent)
    cfg["max_epochs"] = SMOKE_MAX_EPOCHS
    cfg["optimization"]["scheduler"] = lr_scheduler(SMOKE_MAX_EPOCHS - 1, 1)
    cfg["ema_checkpoint_save_epochs"] = {"start": 1, "step": 1}
    cfg["checkpoint_every_n_batches"] = 100
    cfg["train_loader"]["label_sampling"]["samples_per_epoch"] = 256
    for entry in cfg["inference"]:
        entry["epochs"] = {"start": 0, "step": 1}
        if entry["name"] != SELECTION_ENTRY:
            entry["n_forward_steps"] = SMOKE_INFERENCE_STEPS
    return cfg


def describe(arm: str, seed: int, smoke: bool) -> str:
    epochs, decay = (SMOKE_MAX_EPOCHS, 1) if smoke else (MAX_EPOCHS, DECAY_EPOCHS)
    return (
        f"ACE FM sfno-v3{' SMOKE' if smoke else ''}: {epochs} epochs from "
        f"{parent_job_name(arm, seed)} ema_ckpt_{FORK_EPOCH:04d}, LR 1e-4 "
        f"constant {epochs - decay} then linear to 0 over {decay}, "
        "1/2/4 forward steps (p .6/.2/.2), loss on last step, masking off"
    )


def header(arm: str, seed: int, smoke: bool) -> str:
    lines = [
        "Generated by create_fm_v3_jobs.py; do not edit by hand.",
        f"JOB_NAME: {job_name(arm, seed, smoke)}",
        f"PARENT_CONFIG: run_configs/{parent_config_path(arm, seed).name}",
        f"PARENT_JOB: {parent_job_name(arm, seed)}",
        f"PARENT_CHECKPOINT: {PARENT_CHECKPOINT} -> {MOUNTED_PARENT_CHECKPOINT}",
        f"DESCRIPTION: {describe(arm, seed, smoke)}",
        "See create_fm_v3_jobs.py for every change from the parent.",
    ]
    return "".join(f"# {line}\n" for line in lines)


def write_config(arm: str, seed: int, smoke: bool) -> None:
    parent = yaml.safe_load(parent_config_path(arm, seed).read_text())
    cfg = make_smoke_config(parent) if smoke else make_config(parent)
    path = run_config_path(arm, seed, smoke)
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=1000)
    path.write_text(header(arm, seed, smoke) + body)
    print(f"Wrote {path.relative_to(HERE)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Also write the smoke config."
    )
    args = parser.parse_args()
    for arm in ARMS:
        for seed in SEEDS:
            write_config(arm, seed, smoke=False)
    if args.smoke:
        SMOKE_CONFIGS_DIR.mkdir(exist_ok=True)
        write_config(SMOKE_ARM, SMOKE_SEED, smoke=True)


if __name__ == "__main__":
    main()
