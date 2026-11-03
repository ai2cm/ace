"""Generate the anneal fine-tuning configs of the 2026-10-19 nc-sfno balanced runs.

Each anneal run starts from the last EMA checkpoint (`ema_ckpt_0150.tar`) of
one finished balanced nc-sfno parent run and trains for 20 more epochs with a
linearly decaying learning rate, input masking off and an arm-specific label
mix (ARMS below). Every arm runs on each of the 5 parent seeds: 6 x 5 = 30 runs.

arm  parent         label_sampling weights                     stepper_training
a0   -s{k}-maskon   cm4 .40, c96 .25, era5 .25, xshield .10    as parent (1 step)
a1   -s{k}-maskon   era5 .50, cm4 .267, c96 .167, xshield .067 as parent
a2   -s{k}-maskon   era5 .70, cm4 .16, c96 .10, xshield .04    as parent
a3   -s{k}-maskon   era5 1.0, others 0 (weight 0: not drawn)   as parent
a4   -s{k}-maskon   as a1                                      1/2/4 steps (p .6/.2/.2),
                                                               loss on last step only
a5   -s{k}-maskoff  as a1                                      as parent

Changes from the parent config (everything else is copied unchanged):

- `stepper_training.parameter_init.weights_path: /parent_ckpt.tar`, the
  parent's `training_checkpoints/ema_ckpt_0150.tar`, which the run script
  mounts from the parent's Beaker result dataset (PARENT_DATASET header line).
  Parameter init loads stepper weights only: a fresh optimizer, an EMA
  re-initialised from the loaded weights, epoch counter at 0.
- `max_epochs: 20` and `optimization.scheduler`: LinearLR from factor 1 to 0
  over 20 epoch steps, so epoch e trains at lr 1e-4 * (1 - (e-1)/20).
- `stepper.step.config.input_dropout.default.max_masked_vars: 0`; the
  presence channels (`include_channel_mask_inputs: true`) are kept so the
  network's input channels match the parent weights.
- `ema_checkpoint_save_epochs: {start: 16, step: 1}`: ema_ckpt_0016.tar ..
  ema_ckpt_0020.tar (the slice runs over range(max_epochs + 1), so the index
  is the 1-based epoch number).
- `inference`: only the 5-year `aimip_checkpoint_era5` (weight 1) and
  `aimip_checkpoint_c96` (weight 0) entries, every epoch, as in the parent.

Configs are written to run_configs/anneal/<parent config stem>-anneal-<arm>.yaml
by loading the parent YAML, patching it and dumping it again (the parents are
themselves generated, so no hand-written comments are lost). `--smoke` also
writes the 2-epoch smoke-test configs (`-smoke` suffix).

Usage:
    python create_anneal_jobs.py
"""

import argparse
import copy
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import yaml

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
ANNEAL_DIR = RUN_CONFIGS_DIR / "anneal"
BEAKER_MAP_PATH = HERE / "wandb_to_beaker_map.json"

CONFIG_PREFIX = "ace-train-config-4deg-"
JOB_PREFIX = "ace2-fm-"
PARENT_STEM = "nc-sfno-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-balanced"
SEEDS = (0, 1, 2, 3, 4)

PARENT_CHECKPOINT = "training_checkpoints/ema_ckpt_0150.tar"
MOUNTED_PARENT_CHECKPOINT = "/parent_ckpt.tar"

MAX_EPOCHS = 20
EMA_CHECKPOINT_SAVE_EPOCHS = {"start": 16, "step": 1}
SCHEDULER = {
    "type": "LinearLR",
    "kwargs": {"start_factor": 1.0, "end_factor": 0.0, "total_iters": MAX_EPOCHS},
    "step_each_iteration": False,
}
KEPT_INFERENCE = ("aimip_checkpoint_era5", "aimip_checkpoint_c96")

SMOKE_MAX_EPOCHS = 2
SMOKE_EMA_CHECKPOINT_SAVE_EPOCHS = {"start": 1, "step": 1}
SMOKE_ARMS = ("a1", "a4")
SMOKE_SEED = 0

PARENT_WEIGHTS = {"cm4": 0.40, "c96": 0.25, "era5": 0.25, "xshield": 0.10}
ERA5_50 = {"era5": 0.50, "cm4": 0.267, "c96": 0.167, "xshield": 0.067}
ERA5_70 = {"era5": 0.70, "cm4": 0.16, "c96": 0.10, "xshield": 0.04}
ERA5_ONLY = {"era5": 1.0, "cm4": 0.0, "c96": 0.0, "xshield": 0.0}

# ACE2.2's fine-tune draws 1, 2, 4, 12 or 20 steps (p .6 .2 .1 .05 .05) with
# the loss on the last step; capped at 4 steps, the mass above 4 goes to 4.
MULTI_STEP = {
    "n_forward_steps": {
        "outcomes": [
            {"steps": 1, "probability": 0.6},
            {"steps": 2, "probability": 0.2},
            {"steps": 4, "probability": 0.2},
        ]
    },
    "optimize_last_step_only": True,
}


@dataclass(frozen=True)
class Arm:
    name: str
    parent_mask: str  # "on" or "off"
    weights: dict[str, float]
    stepper_training: dict[str, Any] | None
    description: str


ARMS = {
    arm.name: arm
    for arm in (
        Arm("a0", "on", PARENT_WEIGHTS, None, "parent label mix"),
        Arm("a1", "on", ERA5_50, None, "era5 .50"),
        Arm("a2", "on", ERA5_70, None, "era5 .70"),
        Arm("a3", "on", ERA5_ONLY, None, "era5 only"),
        Arm("a4", "on", ERA5_50, MULTI_STEP, "era5 .50, 1/2/4-step rollouts"),
        Arm("a5", "off", ERA5_50, None, "era5 .50, mask-off parent"),
    )
}


def parent_job_name(seed: int, mask: str) -> str:
    return f"{JOB_PREFIX}{PARENT_STEM}-s{seed}-mask{mask}"


def parent_config_path(seed: int, mask: str) -> pathlib.Path:
    return RUN_CONFIGS_DIR / f"{CONFIG_PREFIX}{PARENT_STEM}-s{seed}-mask{mask}.yaml"


def anneal_suffix(arm: str, smoke: bool) -> str:
    return f"-anneal-{arm}" + ("-smoke" if smoke else "")


def anneal_job_name(arm: str, seed: int, smoke: bool = False) -> str:
    return parent_job_name(seed, ARMS[arm].parent_mask) + anneal_suffix(arm, smoke)


def anneal_config_path(arm: str, seed: int, smoke: bool = False) -> pathlib.Path:
    stem = parent_config_path(seed, ARMS[arm].parent_mask).stem
    return ANNEAL_DIR / f"{stem}{anneal_suffix(arm, smoke)}.yaml"


def describe(arm: Arm, seed: int, smoke: bool) -> str:
    weights = ", ".join(f"{k} {v:g}" for k, v in arm.weights.items())
    steps = (
        "1/2/4 forward steps (p .6/.2/.2), loss on last step"
        if arm.stepper_training
        else "1 forward step"
    )
    epochs = SMOKE_MAX_EPOCHS if smoke else MAX_EPOCHS
    return (
        f"ACE FM anneal arm {arm.name}{' SMOKE' if smoke else ''} ({arm.description}): "
        f"{epochs} epochs from {parent_job_name(seed, arm.parent_mask)} "
        f"ema_ckpt_0150, LR 1e-4 linear to 0 over 20 epochs, masking off, "
        f"label weights {weights}, {steps}"
    )


def make_config(parent: dict, arm: Arm, smoke: bool) -> dict:
    config = copy.deepcopy(parent)
    if "resume_results" in config:
        raise ValueError("parent config sets resume_results")
    config["max_epochs"] = SMOKE_MAX_EPOCHS if smoke else MAX_EPOCHS
    config["ema_checkpoint_save_epochs"] = dict(
        SMOKE_EMA_CHECKPOINT_SAVE_EPOCHS if smoke else EMA_CHECKPOINT_SAVE_EPOCHS
    )
    if "scheduler" in config["optimization"]:
        raise ValueError("parent config already sets a scheduler")
    config["optimization"]["scheduler"] = copy.deepcopy(SCHEDULER)

    step_config = config["stepper"]["step"]["config"]
    if step_config.get("include_channel_mask_inputs") is not True:
        raise ValueError("parent config must set include_channel_mask_inputs")
    step_config["input_dropout"] = {"default": {"max_masked_vars": 0}}

    by_name = {entry["name"]: entry for entry in config["inference"]}
    config["inference"] = [by_name[name] for name in KEPT_INFERENCE]
    for entry in config["inference"]:
        if entry["epochs"] != {"start": 0, "step": 1}:
            raise ValueError(f"{entry['name']} does not run every epoch")

    label_sampling = config["train_loader"]["label_sampling"]
    if set(label_sampling["weights"]) != set(arm.weights):
        raise ValueError(f"label sets differ: {label_sampling['weights']}")
    label_sampling["weights"] = dict(arm.weights)

    training = config["stepper_training"]
    if "parameter_init" in training:
        raise ValueError("parent config already sets parameter_init")
    training["parameter_init"] = {"weights_path": MOUNTED_PARENT_CHECKPOINT}
    if arm.stepper_training is not None:
        training.update(copy.deepcopy(arm.stepper_training))
    return config


def header(arm: Arm, seed: int, smoke: bool, dataset_id: str) -> str:
    lines = [
        "Generated by create_anneal_jobs.py; do not edit by hand.",
        f"JOB_NAME: {anneal_job_name(arm.name, seed, smoke)}",
        f"PARENT_JOB: {parent_job_name(seed, arm.parent_mask)}",
        f"PARENT_DATASET: {dataset_id}",
        f"PARENT_CHECKPOINT: {PARENT_CHECKPOINT} -> {MOUNTED_PARENT_CHECKPOINT}",
        f"DESCRIPTION: {describe(arm, seed, smoke)}",
    ]
    return "".join(f"# {line}\n" for line in lines)


def read_header(path: pathlib.Path) -> dict[str, str]:
    """The `# KEY: value` header lines written by `header`."""
    fields = {}
    for line in path.read_text().splitlines():
        if not line.startswith("# "):
            break
        key, sep, value = line[2:].partition(": ")
        if sep:
            fields[key] = value
    return fields


def write_config(arm_name: str, seed: int, smoke: bool, beaker_map: dict) -> None:
    arm = ARMS[arm_name]
    parent = yaml.safe_load(parent_config_path(seed, arm.parent_mask).read_text())
    dataset_id = beaker_map[parent_job_name(seed, arm.parent_mask)]
    config = make_config(parent, arm, smoke)
    path = anneal_config_path(arm_name, seed, smoke)
    body = yaml.safe_dump(config, sort_keys=False, width=1000)
    path.write_text(header(arm, seed, smoke, dataset_id) + body)
    print(f"Wrote {path.relative_to(HERE)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Also write the smoke configs of arms {SMOKE_ARMS}, seed {SMOKE_SEED}.",
    )
    args = parser.parse_args()
    ANNEAL_DIR.mkdir(exist_ok=True)
    beaker_map = json.loads(BEAKER_MAP_PATH.read_text())
    for arm_name in ARMS:
        for seed in SEEDS:
            write_config(arm_name, seed, False, beaker_map)
    if args.smoke:
        for arm_name in SMOKE_ARMS:
            write_config(arm_name, SMOKE_SEED, True, beaker_map)


if __name__ == "__main__":
    main()
