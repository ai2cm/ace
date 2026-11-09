"""Generate the 2026-11-06 FM swin-v2.2-v3 run configs (wandb group ace2-fm-2026-11-06).

10 runs: 2 data arms x 5 seeds, input masking off, trained from scratch. Each
run is the matching nc-swin-v2.2-v2 run config (create_fm_v2_jobs.py, the
"parent") with the step curriculum of the sfno-v3 runs (create_fm_v3_jobs.py)
over the full 100 epochs, and HGTsfc roundtripped where it is inconsistent
with PRESsfc:

    parent ace2-fm-nc-swin-v2.2-v2-fm-{balanced,e3sm-balanced}-s{k}-maskoff
    run    ace2-fm-nc-swin-v2.2-v3-fm-{balanced,e3sm-balanced}-s{k}-maskoff

Changes from the parent (everything else, including max_epochs 100, the LR
schedule (2-epoch warmup, 1e-4 constant to epoch 70, linear to 0 by 100),
data, label sampling, model and EMA decay, is copied verbatim):

- `stepper_training.n_forward_steps`: 1 step for epochs 1-40, then 1, 2 or 4
  steps drawn per batch with probability .6/.2/.2 (create_anneal_jobs.MULTI_STEP)
  from epoch 41, with the loss on the last step only
  (`optimize_last_step_only: true`). This matches what an sfno-v3 run sees on
  the 100-epoch scale: its v2 parent's single-step epochs 1-40, then the v3
  child's sampled steps. It replaces the parent's 1 -> 2 (epoch 41) -> 4
  (epoch 71) schedule with the loss on every step.
- `sht_roundtrip_variables: [HGTsfc]` on every training dataset labelled cm4
  (CM4 piControl, CM4 1pctCO2, CM4-like AM4 random-CO2) or xshield. Their
  PRESsfc was spherical-harmonic roundtripped upstream but HGTsfc was not, so
  grid-scale PRESsfc does not follow HGTsfc (R^2 0.61 / 0.57 vs 0.99 for ERA5
  and C96); roundtripping HGTsfc restores it (R^2 0.992, about -10 Pa/m). The
  roundtrip is a projection, so it would leave an already-roundtripped field
  unchanged. Validation and inline inference use ERA5 and C96 only.
- EMA checkpoints every 5 epochs, `ema_checkpoint_save_epochs: {start: 5,
  step: 5}` (5, 10, ..., 100), as in create_fm_v3_jobs.py. The heavy inline
  inference entries (10year_oosample_varying_co2_era5,
  long_46year_varying_co2_era5) keep the parent's every-10-epoch schedule
  (epochs 1, 11, ..., 91).

Differences from sfno-v3 that remain: no fork (so the optimizer and EMA are
not reset at epoch 40), EMA checkpoints every 5 epochs from the start, and
heavy inline inference every 10 epochs rather than every 5.

Smoke config (`--smoke`) goes to run_configs/v3-smoke/: the s0
fm-e3sm-balanced config cut to 3 epochs, with sampled steps from epoch 2,
256 training samples per epoch, EMA checkpoints and every inline inference
entry every epoch (heavy ones shortened to 365 steps) and
checkpoint_every_n_batches 100.

Usage:
    python create_fm_v3_swin_jobs.py [--smoke]
"""

import argparse
import copy
import pathlib

import yaml
from create_anneal_jobs import MULTI_STEP
from create_fm_v3_jobs import (
    ARMS,
    EMA_CHECKPOINT_SAVE_EPOCHS,
    SEEDS,
    SELECTION_ENTRY,
    SMOKE_ARM,
    SMOKE_CONFIGS_DIR,
    SMOKE_INFERENCE_STEPS,
    SMOKE_SEED,
)

HERE = pathlib.Path(__file__).parent
EXP_DIR = HERE.parent  # experiment root (configs, records); scripts live in scripts/
RUN_CONFIGS_DIR = EXP_DIR / "run_configs"
CONFIG_PREFIX = "ace-train-config-4deg-"

ARCH = "nc-swin-v2.2-v3"
PARENT_ARCH = "nc-swin-v2.2-v2"
SINGLE_STEP_EPOCHS = 40
ROUNDTRIP_LABELS = ("cm4", "xshield")
ROUNDTRIP_VARIABLES = ["HGTsfc"]
SMOKE_MAX_EPOCHS = 3


def parent_config_path(arm: str, seed: int) -> pathlib.Path:
    return RUN_CONFIGS_DIR / f"{CONFIG_PREFIX}{PARENT_ARCH}-{arm}-s{seed}-maskoff.yaml"


def run_config_path(arm: str, seed: int, smoke: bool = False) -> pathlib.Path:
    directory = SMOKE_CONFIGS_DIR if smoke else RUN_CONFIGS_DIR
    suffix = "-smoke" if smoke else ""
    return directory / f"{CONFIG_PREFIX}{ARCH}-{arm}-s{seed}-maskoff{suffix}.yaml"


def n_forward_steps(single_step_epochs: int) -> dict:
    return {
        "start_value": 1,
        "milestones": [
            {
                "epoch": single_step_epochs + 1,
                "value": copy.deepcopy(MULTI_STEP["n_forward_steps"]),
            }
        ],
    }


def make_config(parent: dict) -> dict:
    cfg = copy.deepcopy(parent)
    assert cfg["max_epochs"] == 100
    assert cfg["ema_checkpoint_save_epochs"] == {"start": 10, "step": 10}
    cfg["ema_checkpoint_save_epochs"] = dict(EMA_CHECKPOINT_SAVE_EPOCHS)

    for entry in cfg["inference"]:
        if entry["name"] == SELECTION_ENTRY:
            assert entry["epochs"] == {"start": 0, "step": 1}
        else:
            assert entry["epochs"] == {"start": 0, "step": 10}, entry["name"]

    training = cfg["stepper_training"]
    assert "parameter_init" not in training
    assert training["n_forward_steps"]["milestones"][0]["epoch"] == (
        SINGLE_STEP_EPOCHS + 1
    )
    training["n_forward_steps"] = n_forward_steps(SINGLE_STEP_EPOCHS)
    training["optimize_last_step_only"] = MULTI_STEP["optimize_last_step_only"]

    n_roundtripped = 0
    for dataset in cfg["train_loader"]["dataset"]["concat"]:
        assert "sht_roundtrip_variables" not in dataset
        if set(dataset["labels"]) & set(ROUNDTRIP_LABELS):
            dataset["sht_roundtrip_variables"] = list(ROUNDTRIP_VARIABLES)
            n_roundtripped += 1
    assert n_roundtripped > 0
    return cfg


def make_smoke_config(parent: dict) -> dict:
    cfg = make_config(parent)
    cfg["max_epochs"] = SMOKE_MAX_EPOCHS
    cfg["ema_checkpoint_save_epochs"] = {"start": 1, "step": 1}
    cfg["checkpoint_every_n_batches"] = 100
    cfg["stepper_training"]["n_forward_steps"] = n_forward_steps(1)
    cfg["train_loader"]["label_sampling"]["samples_per_epoch"] = 256
    for entry in cfg["inference"]:
        entry["epochs"] = {"start": 0, "step": 1}
        if entry["name"] != SELECTION_ENTRY:
            entry["n_forward_steps"] = SMOKE_INFERENCE_STEPS
    return cfg


def header(arm: str, seed: int, smoke: bool) -> str:
    lines = [
        "Generated by create_fm_v3_swin_jobs.py; do not edit by hand.",
        f"Parent: run_configs/{parent_config_path(arm, seed).name}",
        "See create_fm_v3_swin_jobs.py for every change from the parent.",
    ]
    if smoke:
        lines.append("Smoke test: 3 epochs, sampled steps from epoch 2.")
    return "".join(f"# {line}\n" for line in lines)


def write_config(arm: str, seed: int, smoke: bool) -> None:
    parent = yaml.safe_load(parent_config_path(arm, seed).read_text())
    cfg = make_smoke_config(parent) if smoke else make_config(parent)
    path = run_config_path(arm, seed, smoke)
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=1000)
    path.write_text(header(arm, seed, smoke) + body)
    print(f"Wrote {path.relative_to(EXP_DIR)}")


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
