"""Generate energy-corrector / GMR / 1940-data ablation configs.

Full factorial (2x2x2 = 8) over three binary factors, applied to the base
nc-sfno config
(configs/experiments/2026-05-27-var-masking/ace-train-config-4deg-AIMIP-nc-sfno.yaml).

Factors and their base-config state:
  - energy corrector (total_energy_budget_correction):  OFF
  - global mean removal (global_mean_removal):          ON
  - training data start:                                1979

Each factor has an on/off label segment used in the filename:
  energy corrector:    -econ   (total_energy_budget_correction: constant_temperature)
                       -ecoff  (no total_energy_budget_correction)
  global mean removal: -gmron  (global_mean_removal: shared, appended)
                       -gmroff (no global_mean_removal)
  training data:       -1940on  (train data extended back to 1940-01-03)
                       -1940off (train data starts 1979)

ERA5 earliest available timestamp is 1940-01-03 (1940-01-01/02 are missing),
verified via gsutil against
gs://vcm-ml-intermediate/2026-04-17-era5-4deg-8layer-daily-1940-2025.

Naming: f"{stem}-{ec}-{gmr}-{data}", e.g.
  ace-train-config-4deg-AIMIP-nc-sfno-econ-gmron-1940off
"""

import argparse
import copy
import itertools
import pathlib

import yaml

WANDB_PROJECT = "VarMasking4"

# Earliest ERA5 timestamp in the dataset (1940-01-01/02 are absent).
EARLIEST_ERA5_TIME = "1940-01-03"

HERE = pathlib.Path(__file__).parent
BASE_CONFIG_STEM = "ace-train-config-4deg-AIMIP-nc-sfno"


def _set_wandb_project(cfg: dict) -> None:
    cfg["logging"]["project"] = WANDB_PROJECT


def _energy_corrector(cfg: dict, enabled: bool) -> None:
    corrector = cfg["stepper"]["step"]["config"]["corrector"]
    if enabled:
        corrector["total_energy_budget_correction"] = {
            "method": "constant_temperature",
        }
    else:
        corrector.pop("total_energy_budget_correction", None)


def _global_mean_removal(cfg: dict, enabled: bool) -> None:
    step_cfg = cfg["stepper"]["step"]["config"]
    if enabled:
        step_cfg["global_mean_removal"] = {
            "kind": "shared",
            "append_as_input": True,
        }
    else:
        step_cfg.pop("global_mean_removal", None)


def _train_from_1940(cfg: dict, enabled: bool) -> None:
    """Extend the earliest training subset back to 1940 (no-op when disabled)."""
    if not enabled:
        return
    concat = cfg["train_loader"]["dataset"]["concat"]
    # Earliest subset is the first concat entry; pull its start back to 1940.
    earliest = min(concat, key=lambda entry: entry["subset"]["start_time"])
    earliest["subset"]["start_time"] = EARLIEST_ERA5_TIME


# Each factor: (factor function, {label_segment: enabled_flag}). Order of the
# label dicts (on first) is the order the factor varies in the filename.
FACTORS: list[tuple] = [
    (_energy_corrector, {"econ": True, "ecoff": False}),
    (_global_mean_removal, {"gmron": True, "gmroff": False}),
    (_train_from_1940, {"1940on": True, "1940off": False}),
]


def _write_config(cfg: dict, out_path: pathlib.Path, existing_only: bool) -> None:
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return
    _set_wandb_project(cfg)
    with out_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}")


def generate_configs(base: dict, existing_only: bool = False) -> None:
    # Cartesian product over each factor's (label, enabled) choices.
    choices = [list(labels.items()) for _, labels in FACTORS]
    for combo in itertools.product(*choices):
        cfg = copy.deepcopy(base)
        labels = []
        for (factor_fn, _), (label, enabled) in zip(FACTORS, combo):
            factor_fn(cfg, enabled)
            labels.append(label)
        out_path = HERE / f"{BASE_CONFIG_STEM}-{'-'.join(labels)}.yaml"
        _write_config(cfg, out_path, existing_only)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite generated YAML files that already exist.",
    )
    args = parser.parse_args()

    base_config = HERE / f"{BASE_CONFIG_STEM}.yaml"
    with base_config.open() as f:
        base = yaml.safe_load(f)
    generate_configs(base, existing_only=args.existing_only)


if __name__ == "__main__":
    main()
