"""Generate the 2026-11-05 FM v2 run configs (wandb group ace2-fm-2026-11-05).

20 runs: 2 architectures x 2 data arms x 5 seeds, all with input masking off.
Each run config is derived from the matching maskoff balanced run config of the
2026-10-19 cells (the "parent"), so everything not listed below is copied from
the parent verbatim: loss, corrector, normalization groups, global-mean
removal, optimizer and weight decay, AMP, checkpoint_every_n_batches, and for
swin float32_matmul_precision and max_consecutive_non_finite_losses.

Architectures (tag directly after `ace-train-config-4deg-`):
- nc-sfno-v2: NoiseConditionedSFNO, embed_dim 1024 (parent 512).
- nc-swin-v2.2-v2: NoiseConditionedSwinTransformer, embed_dim 512 (parent 256).

Data arms:
- `fm-balanced` (ablation): exactly the parent's data, year splits, label
  weights and v2 grouped normalization stats. Parent:
  nc-*-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-balanced-s0-maskoff.
- `fm-e3sm-balanced` (new): the parent -e3sm-balanced config (ablation data +
  E3SMv3 piControl, v3 stats with the e3sm group, label weights cm4 .33,
  e3sm .21, c96 .19, era5 .18, xshield .09) plus the c96 SHiELD AMIP ic_0002
  member (same year split as ic_0001) and c96 SOM members ic_0002..ic_0005 at
  1x/2x/4xCO2, except 1xCO2-ic_0005, which is the held-out member that the
  paper SOM control / abrupt-4xCO2 evaluations initialize from
  (generate_inference_configs.SOM_CONTROL_FILE).

Changes from the parents, identical for both arms:
- max_epochs 100; EMA decay 0.9999 with validate_using_ema; EMA checkpoints
  every 10 epochs (10, 20, ..., 100).
- Multi-step curriculum: 1 forward step for epochs 1-40, 2 for 41-70, 4 for
  71-100, with the loss on every step (optimize_last_step_only false).
  Validation evaluates only the steps the train stepper would for the epoch.
- LR 1e-4 constant to epoch 70, then linear to 0 over epochs 71-100 (stepped
  per epoch). Swin keeps the parent's 2-epoch linear warmup.
- Both architectures run fme's default torch.compile mode (reduce-overhead,
  CUDA graphs). Multi-step training under CUDA graphs needs the fix from
  ai2cm/ace#1610 (accumulated grads cloned out of the CUDA-graph pool),
  merged into this branch; before it, the SFNO smoke failed at the 2-step
  milestone with "accessing tensor output of CUDAGraphs that has been
  overwritten by a subsequent run".
- Inline inference: only aimip_checkpoint_era5 (every epoch, weight 1.0) and,
  every 10 epochs, 10year_oosample_varying_co2_era5 and
  long_46year_varying_co2_era5.

Smoke configs (`--smoke`) go to run_configs/v2-smoke/, which
submit_fm_v2_jobs.select_configs never globs. They are the s0 new-arm configs
cut to a 3-epoch run that exercises 1, 2 and 4 forward steps (milestones at
epochs 2 and 3), the LR schedule, EMA checkpoints every epoch and every inline
inference entry every epoch (heavy ones shortened), with a few hundred
training samples per epoch.

The configs are written by a YAML round trip, so the parents' comments are
not carried over; the header of each file points back to its parent.

Usage:
    python create_fm_v2_jobs.py [--smoke]
"""

import argparse
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
SMOKE_CONFIGS_DIR = RUN_CONFIGS_DIR / "v2-smoke"
CONFIG_PREFIX = "ace-train-config-4deg-"

SEEDS = (0, 1, 2, 3, 4)
MAX_EPOCHS = 100

# arch tag -> (parent arch tag, embed_dim)
ARCHS = {
    "nc-sfno-v2": ("nc-sfno", 1024),
    "nc-swin-v2.2-v2": ("nc-swin-v2.2", 512),
}
# arm name (in the config name) -> parent data stem
ARMS = {
    "fm-balanced": "fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-balanced",
    "fm-e3sm-balanced": "fm-a2-cond-era5-c96-piC-1pct-cm4-xshield-e3sm-balanced",
}
NEW_DATA_ARM = "fm-e3sm-balanced"

KEPT_INFERENCE = (
    "aimip_checkpoint_era5",
    "10year_oosample_varying_co2_era5",
    "long_46year_varying_co2_era5",
)

C96_AMIP_IC2 = {
    "data_path": (
        "/climate-default/"
        "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset"
    ),
    "file_pattern": "ic_0002.zarr",
}
C96_AMIP_TRAIN_SPLITS = (
    ("1979-01-01", "1993-12-31"),
    ("1995-01-01", "2008-12-31"),
)
SOM_DATA_PATH = (
    "/climate-default/"
    "2026-06-08-vertically-resolved-4deg-daily-c96-shield-som-ensemble-fme-dataset"
)
SOM_SUBSET = ("2031-01-02", "2041-01-01")
# Held out: the paper SOM evaluations start from this member.
SOM_HELD_OUT = "1xCO2-ic_0005.zarr"
SOM_NEW_MEMBERS = tuple(
    f"{co2}xCO2-ic_{ic:04d}.zarr"
    for co2 in (1, 2, 4)
    for ic in (2, 3, 4, 5)
    if f"{co2}xCO2-ic_{ic:04d}.zarr" != SOM_HELD_OUT
)


def parent_path(arch: str, arm: str) -> pathlib.Path:
    parent_arch, _ = ARCHS[arch]
    return RUN_CONFIGS_DIR / f"{CONFIG_PREFIX}{parent_arch}-{ARMS[arm]}-s0-maskoff.yaml"


def run_config_name(arch: str, arm: str, seed: int, smoke: bool = False) -> str:
    return (
        f"{CONFIG_PREFIX}{arch}-{arm}-s{seed}-maskoff{'-smoke' if smoke else ''}.yaml"
    )


def _zarr_entry(data_path, file_pattern, start, stop, label="c96") -> dict:
    return {
        "data_path": data_path,
        "file_pattern": file_pattern,
        "engine": "zarr",
        "subset": {"start_time": start, "stop_time": stop},
        "labels": [label],
    }


def new_data_entries() -> list[dict]:
    entries = [
        _zarr_entry(C96_AMIP_IC2["data_path"], C96_AMIP_IC2["file_pattern"], a, b)
        for a, b in C96_AMIP_TRAIN_SPLITS
    ]
    entries += [_zarr_entry(SOM_DATA_PATH, f, *SOM_SUBSET) for f in SOM_NEW_MEMBERS]
    return entries


def lr_scheduler(warmup: bool) -> dict:
    def sched(type_, **kwargs):
        return {"type": type_, "kwargs": kwargs, "step_each_iteration": False}

    decay = sched("LinearLR", start_factor=1.0, end_factor=0.0, total_iters=30)
    if warmup:
        return {
            "schedulers": [
                sched("LinearLR", start_factor=0.01, end_factor=1.0, total_iters=2),
                sched("ConstantLR", factor=1.0, total_iters=68),
                decay,
            ],
            "milestones": [2, 70],
        }
    return {
        "schedulers": [sched("ConstantLR", factor=1.0, total_iters=70), decay],
        "milestones": [70],
    }


def make_config(arch: str, arm: str, seed: int) -> dict:
    cfg = yaml.safe_load(parent_path(arch, arm).read_text())
    _, embed_dim = ARCHS[arch]

    cfg["seed"] = seed
    cfg["max_epochs"] = MAX_EPOCHS
    cfg["ema"] = {"decay": 0.9999}
    cfg["validate_using_ema"] = True
    cfg["ema_checkpoint_save_epochs"] = {"start": 10, "step": 10}

    by_name = {entry["name"]: entry for entry in cfg["inference"]}
    cfg["inference"] = [by_name[name] for name in KEPT_INFERENCE]
    assert by_name["aimip_checkpoint_era5"]["weight"] == 1.0
    assert by_name["aimip_checkpoint_era5"]["epochs"] == {"start": 0, "step": 1}
    for name in KEPT_INFERENCE[1:]:
        assert by_name[name]["epochs"] == {"start": 0, "step": 10}, name

    training = cfg["stepper_training"]
    training["n_forward_steps"] = {
        "start_value": 1,
        "milestones": [{"epoch": 41, "value": 2}, {"epoch": 71, "value": 4}],
    }
    training["optimize_last_step_only"] = False
    assert "pushforward_steps" not in training

    assert isinstance(cfg["validation"], dict)
    cfg["validation"]["evaluate_all_steps"] = False

    optimization = cfg["optimization"]
    had_warmup = "scheduler" in optimization
    if had_warmup:
        first = optimization["scheduler"]["schedulers"][0]
        assert first["type"] == "LinearLR" and first["kwargs"]["total_iters"] == 2
    optimization["scheduler"] = lr_scheduler(warmup=had_warmup)

    step_config = cfg["stepper"]["step"]["config"]
    assert step_config["include_channel_mask_inputs"] is True
    assert "input_dropout" not in step_config
    step_config["builder"]["config"]["embed_dim"] = embed_dim

    if arm == NEW_DATA_ARM:
        concat = cfg["train_loader"]["dataset"]["concat"]
        existing = {(d["data_path"], d["file_pattern"]) for d in concat}
        for entry in new_data_entries():
            assert (entry["data_path"], entry["file_pattern"]) not in existing
        # Insert after the last c96 entry so the c96 members stay together.
        last_c96 = max(i for i, d in enumerate(concat) if d["labels"] == ["c96"])
        concat[last_c96 + 1 : last_c96 + 1] = new_data_entries()
    return cfg


def make_smoke_config(arch: str) -> dict:
    cfg = make_config(arch, NEW_DATA_ARM, seed=0)
    cfg["max_epochs"] = 3
    cfg["ema_checkpoint_save_epochs"] = {"start": 1, "step": 1}
    cfg["checkpoint_every_n_batches"] = 100
    cfg["stepper_training"]["n_forward_steps"] = {
        "start_value": 1,
        "milestones": [{"epoch": 2, "value": 2}, {"epoch": 3, "value": 4}],
    }
    cfg["train_loader"]["label_sampling"]["samples_per_epoch"] = 256
    for entry in cfg["inference"]:
        entry["epochs"] = {"start": 0, "step": 1}
        if entry["name"] != "aimip_checkpoint_era5":
            entry["n_forward_steps"] = 365
    return cfg


def header(arch: str, arm: str, smoke: bool) -> str:
    parent = parent_path(arch, arm).name
    lines = [
        "# Generated by create_fm_v2_jobs.py; do not edit by hand.",
        f"# Parent: run_configs/{parent}",
        "# See create_fm_v2_jobs.py for every change from the parent.",
    ]
    if smoke:
        lines.append("# Smoke test: 3 epochs, 256 samples/epoch, all inference.")
    return "\n".join(lines) + "\n"


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Also write smoke configs."
    )
    args = parser.parse_args()
    for arch in ARCHS:
        for arm in ARMS:
            for seed in SEEDS:
                path = RUN_CONFIGS_DIR / run_config_name(arch, arm, seed)
                path.write_text(
                    header(arch, arm, False) + dump(make_config(arch, arm, seed))
                )
                print(f"Wrote {path.relative_to(HERE)}")
        if args.smoke:
            SMOKE_CONFIGS_DIR.mkdir(exist_ok=True)
            path = SMOKE_CONFIGS_DIR / run_config_name(
                arch, NEW_DATA_ARM, 0, smoke=True
            )
            path.write_text(
                header(arch, NEW_DATA_ARM, True) + dump(make_smoke_config(arch))
            )
            print(f"Wrote {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
