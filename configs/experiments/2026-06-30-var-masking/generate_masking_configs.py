"""Generate var-masking training configs from the c96 nc-sfno base config.

Full-factorial design over three orthogonal masking factors, 16 runs total:

  1. uniform mask level (``default`` scheme, ``max_masked_vars``): 0, 5, 10, 30
  2. co2 bernoulli masking: co2default (off) or co2bern90 (drop
     ``global_mean_co2`` w.p. 0.9)
  3. variable-group bernoulli masking: vgdefault (off) or vgbern20 (each of the
     air_temperature, specific_total_water, eastward_wind, northward_wind
     families is its own group, dropped together w.p. 0.2)

4 x 2 x 2 = 16 configs, written to ``run_configs/``.

global_mean_co2 is already an input channel in the base config
(in_names + next_step_forcing_names). We add pre_cooldown_checkpoint_epoch so a
checkpoint is saved right before cooldown starts.
"""

import argparse
import copy
import pathlib
import re

import yaml

WANDB_PROJECT = "VarMasking"
WANDB_ENTITY = "ai2cm"
WANDB_PREFIX = "ace2-var-mask-"  # stripped from wandb run names before comparison
WANDB_SUFFIX = "-v1"  # stripped from wandb run names before comparison
CONFIG_PREFIX = "ace-train-config-4deg-"  # stripped from config stems

CO2_FIELD = "global_mean_co2"

# Full-factorial design factors.
MASK_LEVELS = [0, 5, 10, 30]  # uniform default max_masked_vars
CO2_OPTIONS = {"co2default": None, "co2bern90": 0.9}
VAR_GROUP_OPTIONS = {"vgdefault": False, "vgbern20": True}

# Variable families masked as bernoulli groups when var-group masking is on.
VAR_GROUP_PREFIXES = [
    "air_temperature",
    "specific_total_water",
    "eastward_wind",
    "northward_wind",
]
VAR_GROUP_RATE = 0.2

HERE = pathlib.Path(__file__).parent
BASE_CONFIG = HERE / "baseline_configs" / "ace-train-config-4deg-nc-sfno-c96.yaml"
STEM = "ace-train-config-4deg-nc-sfno-c96"
RUN_CONFIGS_DIR = HERE / "run_configs"


def _family_channels(prefix: str, in_names: list[str]) -> list[str]:
    """Return level-indexed channels of a variable family, in level order."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    matches = sorted(
        (int(m.group(1)), name) for name in in_names if (m := pattern.match(name))
    )
    if not matches:
        raise ValueError(f"no channels matching {prefix!r}_<n> in in_names")
    return [name for _, name in matches]


def _build_input_dropout(
    mask_level: int,
    co2_rate: float | None,
    use_var_groups: bool,
    in_names: list[str],
) -> dict:
    groups = []
    if co2_rate is not None:
        groups.append({"variables": [CO2_FIELD], "masking": {"rate": co2_rate}})
    if use_var_groups:
        groups += [
            {
                "variables": _family_channels(prefix, in_names),
                "masking": {"rate": VAR_GROUP_RATE},
            }
            for prefix in VAR_GROUP_PREFIXES
        ]
    dropout: dict = {"default": {"max_masked_vars": mask_level}}
    if groups:
        dropout["override_groups"] = groups
    return dropout


def _apply_settings(cfg: dict, input_dropout: dict) -> None:
    step_cfg = cfg["stepper"]["step"]["config"]
    step_cfg["input_dropout"] = input_dropout
    step_cfg["include_channel_mask_inputs"] = True
    cfg["logging"]["project"] = WANDB_PROJECT
    # Save a checkpoint right before cooldown starts (cooldown milestone).
    cfg["pre_cooldown_checkpoint_epoch"] = cfg["optimization"]["scheduler"][
        "milestones"
    ][-1]


def generate_configs() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    in_names = list(base["stepper"]["step"]["config"]["in_names"])

    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for yaml_path in RUN_CONFIGS_DIR.glob("*.yaml"):
        yaml_path.unlink()
        print(f"Removed {yaml_path.name}")

    for mask_level in MASK_LEVELS:
        for co2_name, co2_rate in CO2_OPTIONS.items():
            for vg_name, use_var_groups in VAR_GROUP_OPTIONS.items():
                cfg = copy.deepcopy(base)
                dropout = _build_input_dropout(
                    mask_level, co2_rate, use_var_groups, in_names
                )
                _apply_settings(cfg, dropout)
                name = f"{STEM}-mask{mask_level}-{co2_name}-{vg_name}"
                out_path = RUN_CONFIGS_DIR / f"{name}.yaml"
                out_path.write_text(
                    yaml.dump(cfg, default_flow_style=False, sort_keys=False)
                )
                print(f"Wrote {out_path.name}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    generate_configs()


if __name__ == "__main__":
    main()
