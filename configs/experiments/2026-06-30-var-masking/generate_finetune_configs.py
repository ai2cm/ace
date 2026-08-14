"""Generate multi-step fine-tuning configs for the paper-final var-masking runs.

Each fine-tune config is the run's **exact 1-step pre-training config** (the
config.yaml the checkpoint was trained with, cached under
``pretrain_source_configs/``) with only three changes:

  1. ``stepper_training.n_forward_steps`` is swapped from ``1`` to the multi-step
     probability schedule used by the ERA5 baseline multi-step fine-tune
     (``configs/baselines/era5/ace-train-config-multi-step-finetuning.yaml``):
     {1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05}. This schedule is the *only*
     thing taken from the ERA5 baseline recipe.
  2. ``stepper_training.parameter_init.weights_path`` is added so training starts
     from the pre-trained checkpoint (mounted at ``/weights``).
  3. ``max_epochs`` is capped at FT_MAX_EPOCHS (fine-tuning is short; pre-training
     ran 150).

Everything else -- inference suite, training/validation windows, optimizer
(FusedAdam), EnsembleLoss (crps 0.9 / energy 0.1, no extra weights), EMA,
masking, global-mean-removal, model architecture -- is copied verbatim from
pre-training, so the fine-tune differs from pre-training only in that it rolls
out multiple steps over a short schedule.

Checkpoint dataset IDs (for the ``/weights`` mount) are resolved from
``wandb_to_beaker_map.json`` (refresh with ``update_beaker_map.py``).

Usage:
    python generate_finetune_configs.py [--source-map PATH] [--existing-only]
"""

import argparse
import copy
import json
import pathlib

import yaml
from generate_masking_configs import CONFIG_PREFIX, RUN_CONFIGS_DIR, WANDB_PREFIX

HERE = pathlib.Path(__file__).parent
DEFAULT_SOURCE_MAP = HERE / "wandb_to_beaker_map.json"
PRETRAIN_CONFIGS_DIR = HERE / "pretrain_source_configs"

# Checkpoint file loaded for fine-tuning, matching the ERA5 baseline recipe.
CHECKPOINT_NAME = "training_checkpoints/best_ckpt.tar"

# Suffix distinguishing a fine-tune config/run from its pre-training source.
FT_SUFFIX = "-mstepft"

# Fine-tuning is short; cap it well below pre-training's max_epochs (150).
FT_MAX_EPOCHS = 20

# The one thing taken from the ERA5 baseline multi-step fine-tuning config: the
# n_forward_steps probability schedule. Everything else comes from pre-training.
MULTISTEP_SCHEDULE = {
    "outcomes": [
        {"steps": 1, "probability": 0.6},
        {"steps": 2, "probability": 0.2},
        {"steps": 4, "probability": 0.1},
        {"steps": 12, "probability": 0.05},
        {"steps": 20, "probability": 0.05},
    ]
}

# Paper-final source run per (global-mean-removal, masking) cell. gmron-mask0
# uses seed2 as an interim stand-in because the intended gmron-mask0-seed1 run
# had not produced a succeeded checkpoint when these configs were generated;
# swap it once seed1 finishes (update SELECTED_SOURCES, drop its cached config
# into pretrain_source_configs/, and re-run).
SELECTED_SOURCES = {
    "gmroff-mask0": "ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5",
    "gmroff-mask20": "ace2-var-mask-nc-sfno-era5-gmroff-mask20-seed1-v5",
    "gmron-mask0": "ace2-var-mask-nc-sfno-era5-gmron-mask0-seed2-v5",
    "gmron-mask20": "ace2-var-mask-nc-sfno-era5-gmron-mask20-seed0-v5",
}


def source_run_to_config_stem(source_run_name: str) -> str:
    """Config stem for a source run name.

    ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5
    -> ace-train-config-4deg-nc-sfno-era5-gmroff-mask0-seed1-v5-mstepft
    """
    suffix = source_run_name.removeprefix(WANDB_PREFIX)
    return f"{CONFIG_PREFIX}{suffix}{FT_SUFFIX}"


def _to_finetune(pretrain_cfg: dict) -> dict:
    """Return a fine-tune config: pre-training config with only the multi-step
    schedule, checkpoint weight-loading, and shorter max_epochs changed.
    """
    cfg = copy.deepcopy(pretrain_cfg)
    st = cfg.setdefault("stepper_training", {})
    st["n_forward_steps"] = copy.deepcopy(MULTISTEP_SCHEDULE)
    st["parameter_init"] = {"weights_path": f"/weights/{CHECKPOINT_NAME}"}
    cfg["max_epochs"] = FT_MAX_EPOCHS
    return cfg


def generate_finetune_config(
    source_run_name: str,
    beaker_dataset_id: str,
    existing_only: bool,
) -> None:
    out_path = RUN_CONFIGS_DIR / f"{source_run_to_config_stem(source_run_name)}.yaml"
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return

    pretrain_path = PRETRAIN_CONFIGS_DIR / f"{source_run_name}.yaml"
    if not pretrain_path.exists():
        raise FileNotFoundError(
            f"missing cached pre-training config {pretrain_path.name} in "
            f"{PRETRAIN_CONFIGS_DIR} — fetch it with "
            f"`beaker dataset stream-file {beaker_dataset_id} config.yaml`."
        )
    pretrain_cfg = yaml.safe_load(pretrain_path.read_text())
    cfg = _to_finetune(pretrain_cfg)

    header = (
        f"# arg: --dataset {beaker_dataset_id}:/weights\n"
        f"# source pre-training run: {source_run_name}\n"
        "# = that run's 1-step pre-training config, with only: n_forward_steps"
        " swapped for the\n#   multi-step schedule, parameter_init added, and"
        f" max_epochs capped at {FT_MAX_EPOCHS}\n#   (see"
        " generate_finetune_configs.py).\n"
    )
    with out_path.open("w") as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}  (weights: {beaker_dataset_id})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-map",
        metavar="PATH",
        default=str(DEFAULT_SOURCE_MAP),
        help=(
            "JSON mapping pre-training run name -> Beaker dataset ID"
            f" (default: {DEFAULT_SOURCE_MAP})."
        ),
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only overwrite fine-tune configs that already exist.",
    )
    args = parser.parse_args()

    source_map = json.loads(pathlib.Path(args.source_map).read_text())

    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    missing = []
    for cell, source_run_name in SELECTED_SOURCES.items():
        dataset_id = source_map.get(source_run_name)
        if dataset_id is None:
            missing.append((cell, source_run_name))
            continue
        generate_finetune_config(source_run_name, dataset_id, args.existing_only)

    if missing:
        lines = "\n".join(f"  {cell}: {run}" for cell, run in missing)
        raise SystemExit(
            "No Beaker dataset ID in the source map for:\n"
            f"{lines}\n"
            "Refresh it with update_beaker_map.py (source run must have a "
            "succeeded job)."
        )


if __name__ == "__main__":
    main()
