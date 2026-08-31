"""Generate the masking-information arm x seed training configs.

Three arms, each differing only in how the per-channel presence mask reaches
the network, crossed with ``n_seeds`` seed replicates:

  - ``concat``: presence indicators appended as extra input channels
    (``include_channel_mask_inputs``).
  - ``film``: presence vector injected via FiLM conditioning in the SFNO
    (``builder.config.condition_on_channel_mask``).
  - ``both``: both mechanisms at once.

Everything else is fixed: uniform input masking of up to
``MAX_MASKED_VARS`` channels, no ``global_mean_co2`` input (so no co2 axis),
global-mean removal on, and the v3 (non band-limited) NoiseConditionedSFNO.
This reproduces the v4/v5/double comparison in
``ai2cm/VarMaskingInfoComparison`` without co2 as an input.

Naming: f"{CONFIG_PREFIX}{STEM}-{arm}-seed{seed}.yaml", e.g.
  ace-train-config-4deg-AIMIP-nc-sfno-mask20-uniform-noco2-concat-seed0.yaml
mapping to the wandb run name
  ace2-var-mask-nc-sfno-mask20-uniform-noco2-concat-seed0

Configs are written into ``run_configs/`` (only this generator's own
``*-seed*.yaml`` outputs are cleared first).

Usage:
    python generate_masking_configs.py [--n-seeds N]
"""

import argparse
import copy
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
BASE_CONFIG = HERE / "baseline_configs" / "ace2-varmask-batch-comparison-base.yaml"
RUN_CONFIGS_DIR = HERE / "run_configs"

CONFIG_PREFIX = "ace-train-config-4deg-AIMIP-"
WANDB_PREFIX = "ace2-var-mask-"
WANDB_SUFFIX = ""
WANDB_PROJECT = "VarMaskingInfoComparison"
WANDB_ENTITY = "ai2cm"

# Fixed sweep coordinates, present in the name so runs stay self-describing
# alongside the co2-bearing runs already in the wandb project.
STEM = "nc-sfno-mask20-uniform-noco2"
MAX_MASKED_VARS = 20
DEFAULT_N_SEEDS = 5

# arm token -> (include_channel_mask_inputs, condition_on_channel_mask)
ARMS: dict[str, tuple[bool, bool]] = {
    "concat": (True, False),
    "film": (False, True),
    "both": (True, True),
}


def config_to_run_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


def _apply_arm(cfg: dict, arm: str) -> None:
    """Set the two masking-information flags for ``arm``.

    Both flags are written explicitly (including the ``False`` side) so each
    generated config states which mechanism it uses rather than relying on the
    dataclass defaults.
    """
    include_channel_mask_inputs, condition_on_channel_mask = ARMS[arm]
    step_cfg = cfg["stepper"]["step"]["config"]
    step_cfg["include_channel_mask_inputs"] = include_channel_mask_inputs
    step_cfg["builder"]["config"]["condition_on_channel_mask"] = (
        condition_on_channel_mask
    )


def iter_train_configs(n_seeds: int = DEFAULT_N_SEEDS) -> list[tuple[str, dict]]:
    """``(name, config)`` for every arm x seed training run.

    Built in memory from the base config (no files written), so callers can
    enumerate the sweep without the generated configs being present on disk.
    """
    with BASE_CONFIG.open() as f:
        base = yaml.safe_load(f)

    # The masking level lives in the base config but is asserted here so the
    # value baked into every run name cannot silently disagree with it.
    base_masking = base["stepper"]["step"]["config"]["input_dropout"]
    if base_masking != {"default": {"max_masked_vars": MAX_MASKED_VARS}}:
        raise ValueError(
            f"base config input_dropout {base_masking} does not match the "
            f"uniform max_masked_vars={MAX_MASKED_VARS} this sweep names"
        )

    configs: list[tuple[str, dict]] = []
    for arm in ARMS:
        for seed in range(n_seeds):
            cfg = copy.deepcopy(base)
            _apply_arm(cfg, arm)
            cfg["seed"] = seed
            name = f"{STEM}-{arm}-seed{seed}"
            configs.append((name, cfg))
    return configs


def _write_configs(configs: list[tuple[str, dict]]) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for stale in RUN_CONFIGS_DIR.glob(f"{CONFIG_PREFIX}*-seed*.yaml"):
        stale.unlink()
        print(f"Deleted {stale.name}")
    for name, cfg in configs:
        out_path = RUN_CONFIGS_DIR / f"{CONFIG_PREFIX}{name}.yaml"
        with out_path.open("w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote {out_path.name} -> {config_to_run_name(out_path.name)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help=f"Number of seed replicates per arm (default: {DEFAULT_N_SEEDS}).",
    )
    args = parser.parse_args()

    configs = iter_train_configs(args.n_seeds)
    _write_configs(configs)
    print(f"\n{len(configs)} configs ({len(ARMS)} arms x {args.n_seeds} seeds).")


if __name__ == "__main__":
    main()
