"""Generate var-masking training configs from the base SFNO config.

Ablation dimensions (fully crossed):
  - Model:    sfno, nc-sfno
  - Masking:  mask40 (uniform 1..40), mask5 (uniform 1..5), mask0 (no masking)
  - GMR:      gmron (global mean removal enabled), gmroff (disabled)
  - CO2:      no suffix (no CO2 field), -co2 (global_mean_co2 added as input)
  - Steps:    -steps1 (n_forward_steps=1), -steps2 (n_forward_steps=2)
  - IID:      no suffix (masks shared across ensemble members, default),
              -iid (independent mask per ensemble member, nc-sfno only,
              requires masking to be active)
  - Noise:    no suffix (noise conditioning in layernorm enabled, nc-sfno only),
              -nonoise (noise_embed_dim=0, noise conditioning disabled)

IID rule: -iid variants are only generated for nc-sfno models with active
masking (mask5, mask40). mask0-iid would be a no-op and is skipped.

Noise rule: -nonoise variants are only generated for nc-sfno models (sfno has
no noise conditioning to disable).

Together, nc-sfno produces a 2x2 over (iid/non-iid) x (noise/nonoise).

CO2 rule: when a -co2 variant is generated, global_mean_co2 is added to
in_names and next_step_forcing_names but is never masked.

All runs: 150 epochs, constant LR (no warmup or cooldown).
Residual prediction is always off.
"""

import argparse
import copy
import pathlib

import yaml

WANDB_PROJECT = "VarMasking3"
WANDB_ENTITY = "ai2cm"
WANDB_PREFIX = "ace2-var-mask-"  # stripped from wandb run names before comparison
WANDB_SUFFIX = "-v3"  # stripped from wandb run names before comparison
CONFIG_PREFIX = (
    "ace-train-config-4deg-AIMIP-"  # stripped from config stems before comparison
)

MAX_EPOCHS = 150

CO2_FIELD = "global_mean_co2"
CO2_VALS = [False]
GMR_VALS = [True]
N_FORWARD_STEPS = [1, 2]

# (name_suffix, input_dropout_config_or_None)
MASK_CONFIGS: list[tuple[str, dict | None]] = [
    ("mask10", {"kind": "uniform", "min_vars": 1, "max_vars": 10}),
    ("mask0", None),
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
    gmr_on: bool,
    input_dropout: dict | None,
    co2: bool = False,
    iid: bool = False,
    noise_conditioning: bool = True,
) -> None:
    if co2:
        _add_co2(step_cfg)
    if input_dropout is not None:
        dropout_cfg = dict(input_dropout)
        if iid:
            dropout_cfg["shared_across_ensemble"] = False
        step_cfg["input_dropout"] = dropout_cfg
    else:
        step_cfg.pop("input_dropout", None)
    if not noise_conditioning:
        step_cfg["builder"]["config"]["noise_embed_dim"] = 0
        step_cfg["builder"]["config"]["noise_type"] = "gaussian"
    step_cfg["residual_prediction"] = False
    step_cfg["include_channel_mask_inputs"] = True
    if gmr_on:
        step_cfg["global_mean_removal"] = {
            "kind": "shared",
            "append_as_input": True,
        }
    else:
        step_cfg.pop("global_mean_removal", None)


def _set_wandb_project(cfg: dict) -> None:
    cfg["logging"]["project"] = WANDB_PROJECT


def _set_training_duration(cfg: dict) -> None:
    cfg["max_epochs"] = MAX_EPOCHS
    cfg["optimization"]["scheduler"] = {
        "schedulers": [
            {
                "type": "ConstantLR",
                "kwargs": {"factor": 1.0, "total_iters": MAX_EPOCHS},
                "step_each_iteration": False,
            }
        ],
        "milestones": [],
    }
    cfg.pop("pre_cooldown_checkpoint_epoch", None)


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


def generate_configs(
    base: dict,
    stem: str,
    existing_only: bool = False,
    wandb_run_names: set[str] | None = None,
    iid_vals: list[bool] | None = None,
    noise_conditioning_vals: list[bool] | None = None,
) -> None:
    if iid_vals is None:
        iid_vals = [False]
    if noise_conditioning_vals is None:
        noise_conditioning_vals = [True]
    for mask_suffix, input_dropout in MASK_CONFIGS:
        for gmr_on in GMR_VALS:
            for co2 in CO2_VALS:
                for n_steps in N_FORWARD_STEPS:
                    for iid in iid_vals:
                        if iid and input_dropout is None:
                            continue  # mask0-iid would be a no-op
                        for noise_conditioning in noise_conditioning_vals:
                            cfg = copy.deepcopy(base)
                            gmr_suffix = "gmron" if gmr_on else "gmroff"
                            co2_suffix = "-co2" if co2 else ""
                            iid_suffix = "-iid" if iid else ""
                            noise_suffix = "-nonoise" if not noise_conditioning else ""
                            name = (
                                f"{stem}-{mask_suffix}-{gmr_suffix}"
                                f"{co2_suffix}-steps{n_steps}{iid_suffix}{noise_suffix}"
                            )
                            out_path = HERE / f"{name}.yaml"
                            step_cfg = cfg["stepper"]["step"]["config"]
                            _apply_common_settings(
                                step_cfg,
                                gmr_on,
                                input_dropout,
                                co2=co2,
                                iid=iid,
                                noise_conditioning=noise_conditioning,
                            )
                            if not noise_conditioning:
                                cfg["optimization"]["max_grad_norm"] = 0.3
                            else:
                                cfg["optimization"].pop("max_grad_norm", None)
                            cfg["stepper_training"]["n_forward_steps"] = n_steps
                            loss_type = cfg["stepper_training"]["loss"]["type"]
                            cfg["stepper_training"]["optimize_last_step_only"] = (
                                loss_type == "EnsembleLoss"
                            )
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
            sample = sorted(wandb_run_names)
            print(f"Sample run names (after suffix strip):")
            for name in sample:
                print(f"  {name}")

    for stem in BASE_CONFIG_STEMS:
        base_config = HERE / f"{stem}.yaml"
        with base_config.open() as f:
            base = yaml.safe_load(f)
        iid_vals = [False, True] if "nc-sfno" in stem else [False]
        noise_conditioning_vals = [True, False] if "nc-sfno" in stem else [True]
        generate_configs(
            base,
            stem,
            existing_only=args.existing_only,
            wandb_run_names=wandb_run_names,
            iid_vals=iid_vals,
            noise_conditioning_vals=noise_conditioning_vals,
        )


if __name__ == "__main__":
    main()
