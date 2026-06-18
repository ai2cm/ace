"""Generate var-masking training configs from the base SFNO configs.

One-factor-at-a-time (OFAT) design, anchored at mask10-uniform / co2-default:
  - sfno:     baseline only (mask0, co2-default). 1 run.
  - nc-sfno:  mask dose-response at co2-default (uniform mask0/5/10/20 plus one
              bernoulli robustness point, mask0.11), then a co2 sweep
              (co2-0.4/0.8) at both the anchor mask (mask10) and the bernoulli
              point (mask0.11). 9 runs total.

This trades away mask x co2 interaction terms for far fewer runs; add targeted
interaction cells later only if the main effects look surprising. The full
mask/co2 catalogs below remain available if you want to expand the design.

global_mean_co2 is always an input channel (in_names + next_step_forcing_names).

Naming: f"{stem}-{mask}-{mask_type}-{co2}", e.g.
  ace-train-config-4deg-AIMIP-nc-sfno-mask10-uniform-co2-0.4

All runs: 150 epochs, 8-epoch LinearLR warmup, constant LR, 8-epoch PolynomialLR
cooldown. A pre_cooldown checkpoint is saved at epoch 142 (right before cooldown).
Residual prediction is always off; global mean removal is always on.
"""

import argparse
import copy
import pathlib

import yaml

WANDB_PROJECT = "VarMasking4"
WANDB_ENTITY = "ai2cm"
WANDB_PREFIX = "ace2-var-mask-"  # stripped from wandb run names before comparison
WANDB_SUFFIX = "-v4"  # stripped from wandb run names before comparison
CONFIG_PREFIX = (
    "ace-train-config-4deg-AIMIP-"  # stripped from config stems before comparison
)

MAX_EPOCHS = 150
WARMUP_EPOCHS = 8
COOLDOWN_EPOCHS = 8
PRE_COOLDOWN_CHECKPOINT_EPOCH = MAX_EPOCHS - COOLDOWN_EPOCHS  # 142

CO2_FIELD = "global_mean_co2"

# Mask catalog: name_segment -> (mask_type_segment, base_input_dropout_without_co2)
MASK_CATALOG: dict[str, tuple[str, dict]] = {
    f"mask{n}": ("uniform", {"kind": "uniform", "min_vars": 0, "max_vars": n})
    for n in (0, 5, 10, 20, 30)
} | {
    f"mask{rate}": ("bernoulli", {"kind": "per_variable", "rate": rate})
    for rate in (0.055, 0.11, 0.22, 0.33)
}

# CO2 catalog: name_segment -> co2_rate (None = default, no override)
CO2_CATALOG: dict[str, float | None] = {
    "co2-0.4": 0.4,
    "co2-0.8": 0.8,
    "co2-0.9": 0.9,
    "co2-default": None,
}

# One-factor-at-a-time design. Anchor = mask10-uniform / co2-default.
# nc-sfno: mask dose-response at co2-default (uniform 0/5/10/20 + one bernoulli
# robustness point), then a co2 sweep at each anchor mask. sfno: baseline only.
# (mask_name, co2_name) pairs.
ANCHOR_MASKS: list[str] = ["mask10", "mask0.11", "mask20", "mask0.22", "mask30", "mask0.33"]
CO2_ABLATIONS: list[str] = ["co2-0.4", "co2-0.8", "co2-0.9"]
DOSE_RESPONSE_RUNS: list[tuple[str, str]] = [
    ("mask0", "co2-default"),
    ("mask5", "co2-default"),
    ("mask10", "co2-default"),  # anchor
    ("mask20", "co2-default"),
    ("mask30", "co2-default"),
    ("mask0.11", "co2-default"),  # bernoulli ~ uniform mask10, family check
    ("mask0.22", "co2-default"),  # bernoulli ~ uniform mask20, family check
    ("mask0.33", "co2-default"),  # bernoulli ~ uniform mask30, family check
]
# co2 sweep at each anchor mask.
NC_SFNO_RUNS: list[tuple[str, str]] = DOSE_RESPONSE_RUNS + [
    (mask_name, co2_name) for mask_name in ANCHOR_MASKS for co2_name in CO2_ABLATIONS
]
SFNO_RUNS: list[tuple[str, str]] = [
    ("mask0", "co2-default"),
]

HERE = pathlib.Path(__file__).parent
BASE_CONFIG_STEMS = [
    "ace-train-config-4deg-AIMIP-sfno",
    "ace-train-config-4deg-AIMIP-nc-sfno",
]


def _add_co2(step_cfg: dict) -> None:
    for name_key in ["next_step_forcing_names", "in_names"]:
        names = list(step_cfg[name_key])
        if CO2_FIELD not in names:
            names.append(CO2_FIELD)
        step_cfg[name_key] = names


def _apply_common_settings(
    step_cfg: dict,
    input_dropout: dict,
) -> None:
    _add_co2(step_cfg)
    step_cfg["input_dropout"] = dict(input_dropout)
    step_cfg["residual_prediction"] = False
    step_cfg["include_channel_mask_inputs"] = True
    step_cfg["global_mean_removal"] = {
        "kind": "shared",
        "append_as_input": True,
    }


def _set_wandb_project(cfg: dict) -> None:
    cfg["logging"]["project"] = WANDB_PROJECT


def _set_training_duration(cfg: dict) -> None:
    cfg["max_epochs"] = MAX_EPOCHS
    constant_iters = MAX_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS
    cfg["optimization"]["scheduler"] = {
        "schedulers": [
            {
                "type": "LinearLR",
                "kwargs": {
                    "start_factor": 0.01,
                    "end_factor": 1.0,
                    "total_iters": WARMUP_EPOCHS,
                },
                "step_each_iteration": False,
            },
            {
                "type": "ConstantLR",
                "kwargs": {"factor": 1.0, "total_iters": constant_iters},
                "step_each_iteration": False,
            },
            {
                "type": "PolynomialLR",
                "kwargs": {"power": 0.5, "total_iters": COOLDOWN_EPOCHS},
                "step_each_iteration": False,
            },
        ],
        "milestones": [WARMUP_EPOCHS, MAX_EPOCHS - COOLDOWN_EPOCHS],
    }
    cfg["pre_cooldown_checkpoint_epoch"] = PRE_COOLDOWN_CHECKPOINT_EPOCH


def _fetch_wandb_run_names() -> set[str]:
    import wandb  # lazy import: only needed with --delete-if-in-wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    names = set()
    for run in runs:
        name = run.name
        if WANDB_PREFIX and name.startswith(WANDB_PREFIX):
            name = name[len(WANDB_PREFIX) :]
        if WANDB_SUFFIX and name.endswith(WANDB_SUFFIX):
            name = name[: -len(WANDB_SUFFIX)]
        names.add(name)
    return names


def _write_config(
    cfg: dict,
    out_path: pathlib.Path,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
) -> None:
    stem = out_path.stem
    if CONFIG_PREFIX and stem.startswith(CONFIG_PREFIX):
        stem = stem[len(CONFIG_PREFIX) :]
    if wandb_run_names is not None and stem in wandb_run_names:
        if out_path.exists():
            out_path.unlink()
            print(f"Deleted {out_path.name} (run exists in wandb)")
        else:
            print(f"Skipped {out_path.name} (run exists in wandb, no file)")
        return
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return
    _set_wandb_project(cfg)
    _set_training_duration(cfg)
    with out_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}")


def _runs_for(stem: str) -> list[tuple[str, str]]:
    return NC_SFNO_RUNS if "nc-sfno" in stem else SFNO_RUNS


def generate_configs(
    base: dict,
    stem: str,
    existing_only: bool = False,
    wandb_run_names: set[str] | None = None,
) -> None:
    for mask_name, co2_name in _runs_for(stem):
        mask_type, mask_base = MASK_CATALOG[mask_name]
        co2_rate = CO2_CATALOG[co2_name]
        cfg = copy.deepcopy(base)
        name = f"{stem}-{mask_name}-{mask_type}-{co2_name}"
        out_path = HERE / f"{name}.yaml"
        input_dropout = dict(mask_base)
        if co2_rate is not None:
            input_dropout["co2_rate"] = co2_rate
        step_cfg = cfg["stepper"]["step"]["config"]
        _apply_common_settings(step_cfg, input_dropout)
        _write_config(cfg, out_path, existing_only, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite generated YAML files that already exist.",
    )
    parser.add_argument(
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            f"Delete config files whose run name already exists in "
            f"{WANDB_ENTITY}/{WANDB_PROJECT}."
        ),
    )
    args = parser.parse_args()

    base_yamls = {HERE / f"{stem}.yaml" for stem in BASE_CONFIG_STEMS}
    for yaml_path in HERE.glob("*.yaml"):
        if yaml_path not in base_yamls:
            yaml_path.unlink()
            print(f"Removed {yaml_path.name}")

    wandb_run_names: set[str] | None = None
    if args.delete_if_in_wandb:
        print(f"Fetching run names from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_run_names = _fetch_wandb_run_names()
        print(f"Found {len(wandb_run_names)} existing runs.")
        if wandb_run_names:
            print("Sample run names (after suffix strip):")
            for name in sorted(wandb_run_names):
                print(f"  {name}")

    for stem in BASE_CONFIG_STEMS:
        base_config = HERE / f"{stem}.yaml"
        with base_config.open() as f:
            base = yaml.safe_load(f)
        generate_configs(
            base,
            stem,
            existing_only=args.existing_only,
            wandb_run_names=wandb_run_names,
        )


if __name__ == "__main__":
    main()
