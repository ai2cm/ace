"""Generate var-masking training configs from the nc-sfno base configs.

Configs are generated for both base models (c96 and era5); versioning is ``-v2``
throughout (via ``WANDB_SUFFIX``). Two config groups are written to
``run_configs/`` (emptied first) per model, 16 per model:

  A. uniform mask-level sweep (``default`` scheme, ``max_masked_vars``): 0, 5,
     10, 30, crossed with co2 bernoulli masking (co2default off / co2bern90 drop
     ``global_mean_co2`` w.p. 0.9). 4 x 2 = 8 configs.

  B. mask-all probability gate: the ``default`` uniform scheme masks *all*
     variables (``max_masked_vars`` = N, the input-channel count, which caps at
     the pool size so every fired draw drops the whole pool) and is gated by a
     ``probability`` that it fires on any given batch. Probabilities 1.00, 0.80,
     0.50, 0.20 crossed with the same co2 axis. 4 x 2 = 8 configs.

global_mean_co2 is already an input channel in the base configs
(in_names + next_step_forcing_names).
"""

import argparse
import copy
import pathlib
from typing import NamedTuple

import yaml

WANDB_ENTITY = "ai2cm"
WANDB_PREFIX = "ace2-var-mask-"  # stripped from wandb run names before comparison
WANDB_SUFFIX = "-v2"  # stripped from wandb run names before comparison
CONFIG_PREFIX = "ace-train-config-4deg-"  # stripped from config stems

CO2_FIELD = "global_mean_co2"

# Group A factors: uniform mask-level sweep x co2 bernoulli masking.
MASK_LEVELS = [0, 5, 10, 30]  # uniform default max_masked_vars
CO2_OPTIONS = {"co2default": None, "co2bern90": 0.9}

# Group B factor: probability the mask-all uniform scheme fires on a batch.
MASK_PROBABILITIES = [1.00, 0.80, 0.50, 0.20]

HERE = pathlib.Path(__file__).parent
BASELINE_CONFIGS_DIR = HERE / "baseline_configs"
RUN_CONFIGS_DIR = HERE / "run_configs"


class BaseModel(NamedTuple):
    stem: str  # base config filename stem (also the generated config prefix)
    project: str  # wandb project for this model's runs


# One entry per base config. Source of truth shared with the seed generator.
BASE_MODELS = [
    BaseModel("ace-train-config-4deg-nc-sfno-c96", "VarMaskingC96"),
    BaseModel("ace-train-config-4deg-nc-sfno-era5", "VarMaskingERA5"),
]

# Kept for backward compat with scripts/run-ace-train.sh default.
WANDB_PROJECT = BASE_MODELS[0].project


def config_name_to_run_name(name: str) -> str:
    """Wandb run name for a generated config stem (no .yaml)."""
    suffix = name.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


def config_to_project(config_filename: str) -> str:
    """Wandb project for a generated config, from its base-model stem."""
    stem = pathlib.Path(config_filename).stem
    for model in BASE_MODELS:
        if stem.startswith(model.stem):
            return model.project
    raise ValueError(f"no BASE_MODELS entry matches config {config_filename}")


def _fetch_wandb_run_names(project: str) -> set[str]:
    import wandb  # lazy import: only needed with --delete-if-in-wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{project}")
    return {run.name for run in runs}


def _build_input_dropout(
    mask_level: int,
    co2_rate: float | None,
    probability: float | None = None,
) -> dict:
    groups = []
    if co2_rate is not None:
        groups.append({"variables": [CO2_FIELD], "masking": {"rate": co2_rate}})
    default: dict = {"max_masked_vars": mask_level}
    if probability is not None:
        default["probability"] = probability
    dropout: dict = {"default": default}
    if groups:
        dropout["override_groups"] = groups
    return dropout


def _apply_settings(cfg: dict, input_dropout: dict, project: str) -> None:
    step_cfg = cfg["stepper"]["step"]["config"]
    step_cfg["input_dropout"] = input_dropout
    step_cfg["include_channel_mask_inputs"] = True
    cfg["logging"]["project"] = project


def _write_config(
    base: dict,
    dropout: dict,
    name: str,
    project: str,
    wandb_run_names: set[str] | None = None,
) -> None:
    out_path = RUN_CONFIGS_DIR / f"{name}.yaml"
    if wandb_run_names is not None and config_name_to_run_name(name) in wandb_run_names:
        if out_path.exists():
            out_path.unlink()
            print(f"Deleted {out_path.name} (run exists in wandb)")
        else:
            print(f"Skipped {out_path.name} (run exists in wandb)")
        return
    cfg = copy.deepcopy(base)
    _apply_settings(cfg, dropout, project)
    out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"Wrote {out_path.name}")


def generate_configs(fetch_wandb: bool = False) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for yaml_path in RUN_CONFIGS_DIR.glob("*.yaml"):
        yaml_path.unlink()
        print(f"Removed {yaml_path.name}")

    for model in BASE_MODELS:
        base_config = BASELINE_CONFIGS_DIR / f"{model.stem}.yaml"
        base = yaml.safe_load(base_config.read_text())
        in_names = list(base["stepper"]["step"]["config"]["in_names"])
        # N caps max_masked_vars at the pool size, so a fired draw masks all vars.
        mask_all = len(in_names)

        wandb_run_names: set[str] | None = None
        if fetch_wandb:
            print(f"Fetching run names from {WANDB_ENTITY}/{model.project}...")
            wandb_run_names = _fetch_wandb_run_names(model.project)
            print(f"Found {len(wandb_run_names)} existing runs.")

        # Group A: uniform mask-level sweep x co2.
        for mask_level in MASK_LEVELS:
            for co2_name, co2_rate in CO2_OPTIONS.items():
                dropout = _build_input_dropout(mask_level, co2_rate)
                name = f"{model.stem}-mask{mask_level}-{co2_name}"
                _write_config(base, dropout, name, model.project, wandb_run_names)

        # Group B: mask-all uniform scheme, probability-gated, x co2.
        for co2_name, co2_rate in CO2_OPTIONS.items():
            for probability in MASK_PROBABILITIES:
                dropout = _build_input_dropout(
                    mask_all, co2_rate, probability=probability
                )
                name = f"{model.stem}-mask{mask_all}-{co2_name}-mask{probability:.2f}"
                _write_config(base, dropout, name, model.project, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            "Skip (and delete any existing) configs whose run name already "
            "exists in the model's wandb project."
        ),
    )
    args = parser.parse_args()

    generate_configs(fetch_wandb=args.delete_if_in_wandb)


if __name__ == "__main__":
    main()
