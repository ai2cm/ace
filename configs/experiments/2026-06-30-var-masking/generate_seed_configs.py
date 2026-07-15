"""Generate seed-replicate training configs from the nc-sfno era5 base config.

For a chosen subset of the var-masking sweep this writes ``n_seeds`` copies of
each config (default 5), differing only in the top-level ``seed`` field, so the
same masking scheme can be trained multiple times to estimate run-to-run spread.
Versioning is ``-v1`` throughout (inherited from ``WANDB_SUFFIX``).

Configs are written into ``run_configs/`` (only ``*-seed*.yaml`` files are
cleared first, leaving the other experiments' configs untouched). The masking
subset, each crossed with the co2 axis (co2default off / co2bern90 drops
``global_mean_co2`` w.p. 0.9) and with the seeds, is:

  - mask0:  no masking (``max_masked_vars`` = 0).
  - mask10: uniform 0-10 masking (``max_masked_vars`` = 10).
  - mask40: uniform 0-40 masking (``max_masked_vars`` = 40) gated by a 25%
            ``probability`` of firing on any given batch.

With the default 5 seeds this is 3 x 2 x 5 = 30 configs.
"""

import argparse
import copy
from typing import NamedTuple

import yaml
from generate_masking_configs import (
    BASE_CONFIG_FILENAME,
    BASE_CONFIG_STEM,
    BASELINE_CONFIGS_DIR,
    CO2_OPTIONS,
    RUN_CONFIGS_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    _apply_settings,
    _build_input_dropout,
    _fetch_wandb_run_names,
    config_name_to_run_name,
)

DEFAULT_N_SEEDS = 5


class SeedGroup(NamedTuple):
    label: str
    mask_level: int | None  # ignored when mask_all is True
    mask_all: bool  # if True, max_masked_vars = len(in_names)
    probability: float | None


# Masking subset to replicate across seeds.
SEED_GROUPS = [
    SeedGroup("mask0", 0, False, None),
    SeedGroup("mask10", 10, False, None),
    SeedGroup("mask40", 40, False, 0.25),
]


def _write_config(
    base: dict,
    dropout: dict,
    seed: int,
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
    cfg["seed"] = seed
    out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"Wrote {out_path.name}")


def generate_configs(
    n_seeds: int = DEFAULT_N_SEEDS,
    fetch_wandb: bool = False,
) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for yaml_path in RUN_CONFIGS_DIR.glob("*-seed*.yaml"):
        yaml_path.unlink()
        print(f"Removed {yaml_path.name}")

    base_config = BASELINE_CONFIGS_DIR / BASE_CONFIG_FILENAME
    base = yaml.safe_load(base_config.read_text())
    in_names = list(base["stepper"]["step"]["config"]["in_names"])
    # N caps max_masked_vars at the pool size, so a fired draw masks all vars.
    mask_all = len(in_names)

    wandb_run_names: set[str] | None = None
    if fetch_wandb:
        print(f"Fetching run names from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_run_names = _fetch_wandb_run_names(WANDB_PROJECT)
        print(f"Found {len(wandb_run_names)} existing runs.")

    for group in SEED_GROUPS:
        mask_level = mask_all if group.mask_all else group.mask_level
        assert mask_level is not None
        probability = group.probability
        for co2_name, co2_rate in CO2_OPTIONS.items():
            dropout = _build_input_dropout(
                mask_level, co2_rate, probability=probability
            )
            base_name = f"{BASE_CONFIG_STEM}-{group.label}-{co2_name}"
            if probability is not None:
                base_name = f"{base_name}-mask{probability:.2f}"
            for seed in range(n_seeds):
                name = f"{base_name}-seed{seed}"
                _write_config(base, dropout, seed, name, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help=f"Number of seeds per config (default: {DEFAULT_N_SEEDS}).",
    )
    parser.add_argument(
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            "Skip (and delete any existing) configs whose run name already "
            "exists in the model's wandb project."
        ),
    )
    args = parser.parse_args()

    generate_configs(args.n_seeds, fetch_wandb=args.delete_if_in_wandb)


if __name__ == "__main__":
    main()
