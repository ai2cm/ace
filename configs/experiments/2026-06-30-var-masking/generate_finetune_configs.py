"""Generate multi-step fine-tuning configs for the paper-final var-masking runs.

Takes the four paper-final v5 pre-trained checkpoints (one per
global-mean-removal x masking cell) and, for each, writes a training config that
loads that checkpoint and continues training with the *exact* multi-step
fine-tuning recipe used by the ERA5 baseline
(``configs/baselines/era5/ace-train-config-multi-step-finetuning.yaml``): a
``n_forward_steps`` probability schedule over {1, 2, 4, 12, 20},
``optimize_last_step_only``, EnsembleLoss (crps 0.9 / energy 0.1, h500 weight
5.0), AdamW lr 1e-4, and ``max_epochs: 40``.

The model architecture (NoiseConditionedSFNO-512, global-mean-removal, input
masking, channel-mask inputs, ...) is not re-specified here: ``stepper`` is a
``checkpoint_path`` reference, so the full stepper config is reconstructed from
the pre-trained checkpoint and every var-masking cell keeps its own
architecture. ``stepper_training.parameter_init.weights_path`` then loads those
same weights as the fine-tuning starting point.

The only deliberate deviations from the ERA5 baseline recipe are:
  - ``logging.project`` is ``VarMasking8`` (not ``ace``) so the fine-tunes group
    with the pre-training runs and the existing eval tooling can find them;
  - the pre-trained checkpoint dataset mounted at ``/weights`` differs per cell.

Everything else -- data windows, inference protocol, optimizer, EMA, loss, and
the ``n_forward_steps`` schedule -- is copied verbatim from the ERA5 baseline
fine-tuning config.

Checkpoint dataset IDs are resolved from ``wandb_to_beaker_map.json`` (refresh
it with ``update_beaker_map.py`` if a source run's succeeded job changed).

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

# Checkpoint file loaded for fine-tuning, matching the ERA5 baseline recipe.
CHECKPOINT_NAME = "training_checkpoints/best_ckpt.tar"

# Suffix distinguishing a fine-tune config/run from its pre-training source.
FT_SUFFIX = "-mstepft"

# Paper-final source run per (global-mean-removal, masking) cell. gmron-mask0
# uses seed2 as an interim stand-in because the intended gmron-mask0-seed1 run
# had not produced a succeeded checkpoint when these configs were generated;
# swap it once seed1 finishes (update SELECTED_SOURCES and re-run).
SELECTED_SOURCES = {
    "gmroff-mask0": "ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5",
    "gmroff-mask20": "ace2-var-mask-nc-sfno-era5-gmroff-mask20-seed1-v5",
    "gmron-mask0": "ace2-var-mask-nc-sfno-era5-gmron-mask0-seed2-v5",
    "gmron-mask20": "ace2-var-mask-nc-sfno-era5-gmron-mask20-seed0-v5",
}


def _era5_baseline_finetune_recipe() -> dict:
    """The ERA5 baseline multi-step fine-tuning recipe, minus the per-cell
    checkpoint mount points (filled in by ``generate_finetune_config``).

    Copied verbatim from
    ``configs/baselines/era5/ace-train-config-multi-step-finetuning.yaml`` except
    ``logging.project`` (``VarMasking8`` so the fine-tunes group with the
    experiment) and the ``stepper``/``parameter_init`` weights paths.
    """
    dataset = {
        "data_path": "/climate-default/",
        "file_pattern": "2026-03-19-era5-1deg-8layer-1940-2025.zarr",
        "engine": "zarr",
    }
    return {
        "seed": 0,
        "experiment_dir": "/results",
        "save_checkpoint": True,
        "validate_using_ema": True,
        "max_epochs": 40,
        "ema": {"decay": 0.999},
        "inference": [
            {
                "n_forward_steps": 7300,
                "forward_steps_in_memory": 40,
                "loader": {
                    "start_indices": {
                        "times": [
                            "1996-01-01T00:00:00",
                            "1996-02-15T00:00:00",
                            "1996-04-01T00:00:00",
                            "1996-05-15T00:00:00",
                            "1996-07-01T00:00:00",
                            "1996-08-15T00:00:00",
                            "1996-10-01T00:00:00",
                            "1996-11-15T00:00:00",
                        ]
                    },
                    "dataset": copy.deepcopy(dataset),
                    "num_data_workers": 8,
                },
                "aggregator": {
                    "histogram": {"enabled": True},
                    "time_mean_reference_data": "/statsdata/time-mean.nc",
                },
            }
        ],
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": True,
            "log_to_file": True,
            "project": "VarMasking8",
            "entity": "ai2cm",
        },
        "train_loader": {
            "batch_size": 8,
            "num_data_workers": 8,
            "prefetch_factor": 2,
            "dataset": {
                "concat": [
                    {**copy.deepcopy(dataset), "subset": {"stop_time": "1995-12-31"}},
                    {
                        **copy.deepcopy(dataset),
                        "subset": {
                            "start_time": "2011-01-01",
                            "stop_time": "2019-12-31",
                        },
                    },
                    {
                        **copy.deepcopy(dataset),
                        "subset": {"start_time": "2021-01-01"},
                    },
                ]
            },
        },
        "validation": {
            "loader": {
                "batch_size": 32,
                "num_data_workers": 8,
                "prefetch_factor": 2,
                "dataset": {
                    **copy.deepcopy(dataset),
                    "subset": {
                        "start_time": "1996-01-01",
                        "stop_time": "1997-12-31",
                    },
                },
            }
        },
        "optimization": {
            "use_gradient_accumulation": True,
            "enable_automatic_mixed_precision": False,
            "lr": 0.0001,
            "optimizer_type": "AdamW",
            "kwargs": {"fused": True, "weight_decay": 0.01},
        },
        "stepper_training": {
            "n_ensemble": 2,
            "parameter_init": {"weights_path": f"/weights/{CHECKPOINT_NAME}"},
            "n_forward_steps": {
                "outcomes": [
                    {"steps": 1, "probability": 0.6},
                    {"steps": 2, "probability": 0.2},
                    {"steps": 4, "probability": 0.1},
                    {"steps": 12, "probability": 0.05},
                    {"steps": 20, "probability": 0.05},
                ]
            },
            "optimize_last_step_only": True,
            "loss": {
                "type": "EnsembleLoss",
                "weights": {"h500": 5.0},
                "kwargs": {"crps_weight": 0.9, "energy_score_weight": 0.1},
            },
        },
        "stepper": {"checkpoint_path": f"/weights/{CHECKPOINT_NAME}"},
    }


def source_run_to_config_stem(source_run_name: str) -> str:
    """Config stem for a source run name.

    ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5
    -> ace-train-config-4deg-nc-sfno-era5-gmroff-mask0-seed1-v5-mstepft
    """
    suffix = source_run_name.removeprefix(WANDB_PREFIX)
    return f"{CONFIG_PREFIX}{suffix}{FT_SUFFIX}"


def generate_finetune_config(
    source_run_name: str,
    beaker_dataset_id: str,
    existing_only: bool,
) -> None:
    out_path = RUN_CONFIGS_DIR / f"{source_run_to_config_stem(source_run_name)}.yaml"
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return

    cfg = _era5_baseline_finetune_recipe()
    stats_dataset = "andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019"
    header = (
        f"# arg: --dataset {stats_dataset}:/statsdata\n"
        f"# arg: --dataset {beaker_dataset_id}:/weights\n"
        f"# source pre-training run: {source_run_name}\n"
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
