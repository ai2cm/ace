"""Generate seed-replicate training configs from the nc-sfno era5 base config.

For a chosen subset of the var-masking sweep this writes ``n_seeds`` copies of
each config (default 5), differing only in the top-level ``seed`` field, so the
same masking scheme can be trained multiple times to estimate run-to-run spread.
Each generated config name ends in ``-v1`` or ``-v2`` (``--version`` selects
which baseline config to source from, default v1; see
``baseline_configs/versions.md``). The ``global_mean_removal`` stepper config
is kept as in the baseline for v1 (no gmr axis there, unlike
``generate_masking_configs.py``); v2 sweeps both gmron/gmroff (see
``gmr_options_for_version``).

Configs are written into ``run_configs/`` (only ``*-seed*.yaml`` files are
cleared first, leaving the other experiments' configs untouched). The masking
subset, each crossed with the co2 axis (co2default off / co2bern75 drops
``global_mean_co2`` w.p. 0.75) and with the seeds, is:

  - mask0:  no masking (``max_masked_vars`` = 0).
  - mask30: uniform 0-30 masking (``max_masked_vars`` = 30).

With the default 5 seeds this is 2 x 2 x 5 = 20 configs for v1:
mask30-co2bern75, mask30-co2default, mask0-co2bern75, mask0-co2default.
For v2, global_mean_co2 is not an input channel (see
``baseline_configs/versions.md``), so the co2 axis is dropped, but the gmr
axis is added instead: 2 x 1 x 2 x 5 = 20 configs: mask30-gmron,
mask30-gmroff, mask0-gmron, mask0-gmroff (each co2default).
"""

import argparse
import copy
from typing import NamedTuple

import yaml
from generate_masking_configs import (
    BASE_CONFIG_FILENAMES,
    BASE_CONFIG_STEM,
    BASELINE_CONFIGS_DIR,
    DEFAULT_VERSION,
    GMR_OPTIONS,
    RUN_CONFIGS_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    _apply_settings,
    _fetch_wandb_run_names,
    co2_options_for_version,
    config_name_to_run_name,
)

DEFAULT_N_SEEDS = 5


def gmr_options_for_version(version: str) -> dict[str, bool]:
    """GMR_OPTIONS, restricted to keep-baseline-only for v1.

    No gmr axis for v1 (``global_mean_removal`` is kept as in the baseline,
    as before, with no gmr token in the config name); v2 sweeps both
    gmron/gmroff, as in ``generate_masking_configs.py``.
    """
    if version == "v1":
        return {"": True}
    return GMR_OPTIONS


class SeedGroup(NamedTuple):
    label: str
    mask_level: int


# Masking subset to replicate across seeds.
SEED_GROUPS = [
    SeedGroup("mask0", 0),
    SeedGroup("mask20", 20),
]


def iter_train_configs(
    version: str, n_seeds: int = DEFAULT_N_SEEDS
) -> list[tuple[str, dict]]:
    """``(name, config)`` for every seed-replicate training run of ``version``.

    Built in memory from the baseline config (no files written), so callers
    (e.g. generate_eval_configs.py) can enumerate every version's runs without
    the on-disk configs of that version being present.
    """
    base = yaml.safe_load(
        (BASELINE_CONFIGS_DIR / BASE_CONFIG_FILENAMES[version]).read_text()
    )
    configs: list[tuple[str, dict]] = []
    co2_options = co2_options_for_version(version)
    gmr_options = gmr_options_for_version(version)
    for group in SEED_GROUPS:
        for co2_name, co2_rate in co2_options.items():
            for gmr_name, keep_gmr in gmr_options.items():
                gmr_token = f"{gmr_name}-" if gmr_name else ""
                base_name = f"{BASE_CONFIG_STEM}-{gmr_token}{group.label}-{co2_name}"
                for seed in range(n_seeds):
                    name = f"{base_name}-seed{seed}-{version}"
                    cfg = copy.deepcopy(base)
                    _apply_settings(cfg, group.mask_level, co2_rate, keep_gmr)
                    cfg["seed"] = seed
                    configs.append((name, cfg))
    return configs


def _write_config(
    name: str,
    cfg: dict,
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
    out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"Wrote {out_path.name}")


def generate_configs(
    n_seeds: int = DEFAULT_N_SEEDS,
    fetch_wandb: bool = False,
    version: str = DEFAULT_VERSION,
) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for yaml_path in RUN_CONFIGS_DIR.glob("*-seed*.yaml"):
        yaml_path.unlink()
        print(f"Removed {yaml_path.name}")

    wandb_run_names: set[str] | None = None
    if fetch_wandb:
        print(f"Fetching run names from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_run_names = _fetch_wandb_run_names(WANDB_PROJECT)
        print(f"Found {len(wandb_run_names)} existing runs.")

    for name, cfg in iter_train_configs(version, n_seeds):
        _write_config(name, cfg, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help=f"Number of seeds per config (default: {DEFAULT_N_SEEDS}).",
    )
    parser.add_argument(
        "--version",
        "-v",
        choices=sorted(BASE_CONFIG_FILENAMES),
        default=DEFAULT_VERSION,
        help=f"Baseline config version to sweep from (default: {DEFAULT_VERSION}).",
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

    generate_configs(
        args.n_seeds, fetch_wandb=args.delete_if_in_wandb, version=args.version
    )


if __name__ == "__main__":
    main()
