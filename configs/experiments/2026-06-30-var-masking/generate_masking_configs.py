"""Generate var-masking training configs from the c96 nc-sfno base config.

Two config groups are written to ``run_configs/`` (emptied first), 16 total:

  A. uniform mask-level sweep (``default`` scheme, ``max_masked_vars``): 0, 5,
     10, 30, crossed with co2 bernoulli masking (co2default off / co2bern90 drop
     ``global_mean_co2`` w.p. 0.9). 4 x 2 = 8 configs.

  B. mask-all probability gate: the ``default`` uniform scheme masks *all*
     variables (``max_masked_vars`` = N, the input-channel count, which caps at
     the pool size so every fired draw drops the whole pool) and is gated by a
     ``probability`` that it fires on any given batch. Probabilities 1.00, 0.80,
     0.50, 0.20 crossed with the same co2 axis. 4 x 2 = 8 configs.

global_mean_co2 is already an input channel in the base config
(in_names + next_step_forcing_names).
"""

import argparse
import copy
import pathlib

import yaml

WANDB_PROJECT = "VarMaskingC96"
WANDB_ENTITY = "ai2cm"
WANDB_PREFIX = "ace2-var-mask-"  # stripped from wandb run names before comparison
WANDB_SUFFIX = "-v1"  # stripped from wandb run names before comparison
CONFIG_PREFIX = "ace-train-config-4deg-"  # stripped from config stems

CO2_FIELD = "global_mean_co2"

# Group A factors: uniform mask-level sweep x co2 bernoulli masking.
MASK_LEVELS = [0, 5, 10, 30]  # uniform default max_masked_vars
CO2_OPTIONS = {"co2default": None, "co2bern90": 0.9}

# Group B factor: probability the mask-all uniform scheme fires on a batch.
NOMASK_PROBABILITIES = [1.00, 0.80, 0.50, 0.20]

HERE = pathlib.Path(__file__).parent
BASE_CONFIG = HERE / "baseline_configs" / "ace-train-config-4deg-nc-sfno-c96.yaml"
STEM = "ace-train-config-4deg-nc-sfno-c96"
RUN_CONFIGS_DIR = HERE / "run_configs"


def config_name_to_run_name(name: str) -> str:
    """Wandb run name for a generated config stem (no .yaml)."""
    suffix = name.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


def _fetch_wandb_run_names() -> set[str]:
    import wandb  # lazy import: only needed with --delete-if-in-wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
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


def _apply_settings(cfg: dict, input_dropout: dict) -> None:
    step_cfg = cfg["stepper"]["step"]["config"]
    step_cfg["input_dropout"] = input_dropout
    step_cfg["include_channel_mask_inputs"] = True
    cfg["logging"]["project"] = WANDB_PROJECT


def _write_config(
    base: dict,
    dropout: dict,
    name: str,
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
    _apply_settings(cfg, dropout)
    out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"Wrote {out_path.name}")


def generate_configs(wandb_run_names: set[str] | None = None) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    in_names = list(base["stepper"]["step"]["config"]["in_names"])
    # N caps max_masked_vars at the pool size, so a fired draw masks all vars.
    mask_all = len(in_names)

    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for yaml_path in RUN_CONFIGS_DIR.glob("*.yaml"):
        yaml_path.unlink()
        print(f"Removed {yaml_path.name}")

    # Group A: uniform mask-level sweep x co2.
    for mask_level in MASK_LEVELS:
        for co2_name, co2_rate in CO2_OPTIONS.items():
            dropout = _build_input_dropout(mask_level, co2_rate)
            name = f"{STEM}-mask{mask_level}-{co2_name}"
            _write_config(base, dropout, name, wandb_run_names)

    # Group B: mask-all uniform scheme, probability-gated, x co2.
    for co2_name, co2_rate in CO2_OPTIONS.items():
        for probability in NOMASK_PROBABILITIES:
            dropout = _build_input_dropout(mask_all, co2_rate, probability=probability)
            name = f"{STEM}-mask{mask_all}-{co2_name}-nomask{probability:.2f}"
            _write_config(base, dropout, name, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            f"Skip (and delete any existing) configs whose run name already "
            f"exists in {WANDB_ENTITY}/{WANDB_PROJECT}."
        ),
    )
    args = parser.parse_args()

    wandb_run_names: set[str] | None = None
    if args.delete_if_in_wandb:
        print(f"Fetching run names from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_run_names = _fetch_wandb_run_names()
        print(f"Found {len(wandb_run_names)} existing runs.")

    generate_configs(wandb_run_names)


if __name__ == "__main__":
    main()
