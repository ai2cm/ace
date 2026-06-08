"""Generate var-masking training configs from the base SFNO config.

Produces configs for two masking schemes:

Bernoulli (suffix -bernoulli):
  One config per (mask_rate, gmr):
  - mask_rates: 0.05, 0.1, 0.2, 0.3, 0.4
    (applied as input_dropout.per_variable.default_rate)
  - gmr: gmron (global mean removal enabled) / gmroff (disabled)
  All variables share per_variable.default_rate; the shared
  global_mean_removal reference is protected when gmr is enabled.
  Forcing variables are masked the same as all other variables.

Jeremy/uniform (suffix -uniform):
  One config per (gmr):
  - uniform masking from 0 to all eligible variables
  - gmr: gmron / gmroff
  Forcing variables are eligible for masking.
  The shared global_mean_removal reference is protected when gmr is enabled.

CO2 variants are generated for every Bernoulli and uniform config:
  - suffix -co2-mask: global_mean_co2 is masked the same way as other variables
  - suffix -co2-nomask: global_mean_co2 is excluded from input dropout

No-masking (suffix -mask0.00):
  One config per (gmr, co2_mode):
  - gmr: gmron / gmroff
  - co2: no suffix (no co2 field), -co2-mask, -co2-nomask

All runs: 150 epochs (4 warmup, 142 constant, 4 cooldown).
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
WARMUP_EPOCHS = 7
COOLDOWN_EPOCHS = 7
COOLDOWN_START_EPOCH = MAX_EPOCHS - COOLDOWN_EPOCHS
PRE_COOLDOWN_CHECKPOINT_EPOCH = COOLDOWN_START_EPOCH

FORCING_VARS = [
    "land_fraction",
    "ocean_fraction",
    "sea_ice_fraction",
    "DSWRFtoa",
    "HGTsfc",
]
SHARED_GMR_REFERENCE_FIELD = "surface_temperature"
CO2_FIELD = "global_mean_co2"

MASK_RATES = [0.05, 0.1, 0.2, 0.3, 0.4]
GMR_VALS = [True, False]
NO_MASKING_GMR_VALS = [True, False]
NO_MASKING_CO2_MODES = [True, False]  # True = CO2 added as input, False = no CO2
CO2_MODES: list[str] = []

HERE = pathlib.Path(__file__).parent
BASE_CONFIG_STEMS = [
    "ace-train-config-4deg-AIMIP-sfno",
    "ace-train-config-4deg-AIMIP-nc-sfno",
]


def build_bernoulli_input_dropout(mask_rate: float) -> dict:
    per_variable: dict[str, float] = {"default_rate": mask_rate}
    return {"per_variable": per_variable}


def build_uniform_input_dropout() -> dict:
    return {"uniform": {"min_vars": "min", "max_vars": "max"}}


def _protect_shared_gmr_reference(input_dropout: dict) -> dict:
    """Prevent input dropout from masking the shared GMR reference field."""
    return _exclude_from_input_dropout(input_dropout, SHARED_GMR_REFERENCE_FIELD)


def _exclude_from_input_dropout(input_dropout: dict, name: str) -> dict:
    input_dropout = copy.deepcopy(input_dropout)
    if "uniform" in input_dropout:
        uniform = dict(input_dropout["uniform"])
        ignore_vars = list(uniform.get("ignore_vars", []))
        if name not in ignore_vars:
            ignore_vars.append(name)
        uniform["ignore_vars"] = ignore_vars
        input_dropout["uniform"] = uniform
        per_variable = dict(input_dropout.get("per_variable", {}))
        if per_variable.get("default_rate", 0.0) > 0.0:
            per_variable[name] = 0.0
            input_dropout["per_variable"] = per_variable
    else:
        per_variable = dict(input_dropout.get("per_variable", {}))
        per_variable[name] = 0.0
        input_dropout["per_variable"] = per_variable
    return input_dropout


def _add_co2(step_cfg: dict) -> None:
    for name_key in ["next_step_forcing_names", "in_names"]:
        names = list(step_cfg[name_key])
        if CO2_FIELD not in names:
            names.append(CO2_FIELD)
        step_cfg[name_key] = names


def _unmask_co2(input_dropout: dict) -> dict:
    return _exclude_from_input_dropout(input_dropout, CO2_FIELD)


def _co2_suffix(co2_mode: str | None) -> str:
    return "" if co2_mode is None else f"-co2-{co2_mode}"


def _apply_common_settings(
    step_cfg: dict,
    gmr_on: bool,
    input_dropout: dict,
    co2_mode: str | None = None,
) -> None:
    if co2_mode is not None:
        _add_co2(step_cfg)
        if co2_mode == "nomask":
            input_dropout = _unmask_co2(input_dropout)
        elif co2_mode != "mask":
            raise ValueError(f"Invalid CO2 mode: {co2_mode}")
    if gmr_on:
        input_dropout = _protect_shared_gmr_reference(input_dropout)
    step_cfg["input_dropout"] = input_dropout
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


def _set_pre_cooldown_checkpoint_epoch(cfg: dict) -> None:
    cfg["pre_cooldown_checkpoint_epoch"] = PRE_COOLDOWN_CHECKPOINT_EPOCH


def _set_training_duration(cfg: dict) -> None:
    cfg["max_epochs"] = MAX_EPOCHS
    schedulers = cfg["optimization"]["scheduler"]["schedulers"]
    main_epochs = MAX_EPOCHS - WARMUP_EPOCHS - COOLDOWN_EPOCHS
    schedulers[1]["kwargs"]["total_iters"] = main_epochs
    cfg["optimization"]["scheduler"]["milestones"] = [
        WARMUP_EPOCHS,
        COOLDOWN_START_EPOCH,
    ]


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
    _set_pre_cooldown_checkpoint_epoch(cfg)
    _set_training_duration(cfg)
    with out_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}")


def generate_bernoulli_configs(
    base: dict,
    stem: str,
    co2_mode: str | None = None,
    existing_only: bool = False,
    wandb_run_names: set[str] | None = None,
) -> None:
    for mask_rate in MASK_RATES:
        for gmr_on in GMR_VALS:
            cfg = copy.deepcopy(base)
            gmr_suffix = "gmron" if gmr_on else "gmroff"
            name = (
                f"{stem}"
                f"-mask{mask_rate:.2f}"
                f"-{gmr_suffix}-bernoulli"
                f"{_co2_suffix(co2_mode)}"
            )
            out_path = HERE / f"{name}.yaml"
            step_cfg = cfg["stepper"]["step"]["config"]
            _apply_common_settings(
                step_cfg,
                gmr_on,
                build_bernoulli_input_dropout(mask_rate),
                co2_mode=co2_mode,
            )
            _write_config(cfg, out_path, existing_only, wandb_run_names)


def generate_uniform_configs(
    base: dict,
    stem: str,
    co2_mode: str | None = None,
    existing_only: bool = False,
    wandb_run_names: set[str] | None = None,
) -> None:
    for gmr_on in GMR_VALS:
        cfg = copy.deepcopy(base)
        gmr_suffix = "gmron" if gmr_on else "gmroff"
        name = f"{stem}" f"-maskall" f"-{gmr_suffix}-uniform" f"{_co2_suffix(co2_mode)}"
        out_path = HERE / f"{name}.yaml"
        step_cfg = cfg["stepper"]["step"]["config"]
        _apply_common_settings(
            step_cfg,
            gmr_on,
            build_uniform_input_dropout(),
            co2_mode=co2_mode,
        )
        _write_config(cfg, out_path, existing_only, wandb_run_names)


def generate_co2_variants(
    base: dict,
    stem: str,
    existing_only: bool = False,
    wandb_run_names: set[str] | None = None,
) -> None:
    """Generate -co2-mask and -co2-nomask configs."""
    for co2_mode in CO2_MODES:
        generate_bernoulli_configs(
            base,
            stem,
            co2_mode=co2_mode,
            existing_only=existing_only,
            wandb_run_names=wandb_run_names,
        )
        generate_uniform_configs(
            base,
            stem,
            co2_mode=co2_mode,
            existing_only=existing_only,
            wandb_run_names=wandb_run_names,
        )


def generate_no_masking_configs(
    base: dict,
    stem: str,
    existing_only: bool = False,
    wandb_run_names: set[str] | None = None,
) -> None:
    """Generate no-masking configs with GMR on/off and CO2 on/off."""
    for gmr_on in NO_MASKING_GMR_VALS:
        for co2_on in NO_MASKING_CO2_MODES:
            cfg = copy.deepcopy(base)
            gmr_suffix = "gmron" if gmr_on else "gmroff"
            co2_suffix = "-co2" if co2_on else ""
            name = f"{stem}" f"-mask0.00" f"-{gmr_suffix}-bernoulli" f"{co2_suffix}"
            out_path = HERE / f"{name}.yaml"
            step_cfg = cfg["stepper"]["step"]["config"]
            _apply_common_settings(
                step_cfg,
                gmr_on,
                build_bernoulli_input_dropout(0.0),
                co2_mode="mask" if co2_on else None,
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
        generate_no_masking_configs(
            base,
            stem,
            existing_only=args.existing_only,
            wandb_run_names=wandb_run_names,
        )
        generate_bernoulli_configs(
            base,
            stem,
            existing_only=args.existing_only,
            wandb_run_names=wandb_run_names,
        )
        generate_uniform_configs(
            base,
            stem,
            existing_only=args.existing_only,
            wandb_run_names=wandb_run_names,
        )
        generate_co2_variants(
            base,
            stem,
            existing_only=args.existing_only,
            wandb_run_names=wandb_run_names,
        )


if __name__ == "__main__":
    main()
