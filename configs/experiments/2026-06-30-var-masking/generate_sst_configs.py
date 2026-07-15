"""Generate SST-perturbation inference configs for the VarMasking8 (ERA5)
checkpoints.

Free-running inference configs are written to ``run_configs/``, one per
constant SST perturbation level, using the ERA5 reanalysis zarr forcing
dataset native to the checkpoints. The perturbation levels are p0k / p2k /
p4k. Each config mounts its checkpoint at ``/ckpt.tar`` (supplied per-run by
submit_sst_jobs.py) and runs a prognostic forecast with the SST forcing
shifted by a constant amplitude.

The configs are run-agnostic: the per-run checkpoint dataset is provided at
submit time, so the same configs are reused across every training run. They
are consumed by ``python -m fme.ace.inference`` (via run-ace-inference.sh),
not by the evaluator suite.
"""

import argparse
import pathlib
from typing import NamedTuple

import yaml
from generate_eval_configs import TRAINING_RESULT_DATASETS, _fetch_wandb_run_names
from generate_masking_configs import RUN_CONFIGS_DIR, WANDB_PROJECT

HERE = pathlib.Path(__file__).parent
SST_CONFIG_PREFIX = "ace-inference-sst-config-4deg-"
CHECKPOINT_PATH = "/ckpt.tar"

# Constant SST perturbation amplitudes (Kelvin), keyed by config/job suffix.
SST_PERTURBATIONS = {
    "p0k": 0.0,
    "p2k": 2.0,
    "p4k": 4.0,
}


class DatasetSpec(NamedTuple):
    data_path: str
    file_pattern: str
    n_forward_steps: int


# Varying-CO2 forcing dataset native to the checkpoints. ``data_path`` is the
# zarr directory; ``file_pattern`` is the zarr name within it.
# ``n_forward_steps`` matches the "long"-inference baseline entry.
DATASET = DatasetSpec(
    data_path="/climate-default",
    file_pattern="2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr",
    n_forward_steps=16794,
)

# Free-running inference settings (shared by every config).
FORWARD_STEPS_IN_MEMORY = 40
INITIAL_CONDITION_TIME = "1979-01-01T00:00:00"


def sst_config_filename(level: str) -> str:
    return f"{SST_CONFIG_PREFIX}{level}.yaml"


def _all_runs_finished_in_wandb(level: str) -> bool:
    """True if every training run already has a finished wandb run for this
    SST perturbation level (job name ``{run_name}-{level}``, submitted by
    submit_sst_jobs.py into the training project).
    """
    finished_runs = _fetch_wandb_run_names(WANDB_PROJECT)
    return all(
        f"{run_name}-{level}" in finished_runs for run_name in TRAINING_RESULT_DATASETS
    )


def _build_inference_config(
    data_path: str,
    file_pattern: str,
    n_forward_steps: int,
    amplitude: float,
) -> dict:
    return {
        "checkpoint_path": CHECKPOINT_PATH,
        "allow_incompatible_dataset": True,
        "experiment_dir": "/results",
        "n_forward_steps": n_forward_steps,
        "forward_steps_in_memory": FORWARD_STEPS_IN_MEMORY,
        "data_writer": {
            "save_monthly_files": False,
            "save_prediction_files": False,
        },
        "initial_condition": {
            "path": f"{data_path}/{file_pattern}",
            "engine": "zarr",
            "start_indices": {"times": [INITIAL_CONDITION_TIME]},
        },
        "forcing_loader": {
            "dataset": {
                "data_path": data_path,
                "file_pattern": file_pattern,
                "engine": "zarr",
            },
            "num_data_workers": 4,
            "perturbations": {
                "sst": [
                    {
                        "type": "constant",
                        "config": {"amplitude": amplitude},
                    }
                ]
            },
        },
        "logging": {"project": WANDB_PROJECT},
    }


def generate_configs(
    existing_only: bool = False, delete_if_in_wandb: bool = False
) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for level, amplitude in SST_PERTURBATIONS.items():
        out_path = RUN_CONFIGS_DIR / sst_config_filename(level)
        if delete_if_in_wandb and _all_runs_finished_in_wandb(level):
            if out_path.exists():
                out_path.unlink()
                print(f"Deleted {out_path.name} (all runs exist in wandb)")
            else:
                print(f"Skipped {out_path.name} (all runs exist in wandb, no file)")
            continue
        if existing_only and not out_path.exists():
            print(f"Skipped {out_path.name}")
            continue
        cfg = _build_inference_config(
            DATASET.data_path,
            DATASET.file_pattern,
            DATASET.n_forward_steps,
            amplitude,
        )
        out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        print(f"Wrote {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite SST configs that already exist.",
    )
    parser.add_argument(
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            "Delete perturbation-level configs whose training runs all already "
            "have a finished SST-perturbation run in wandb."
        ),
    )
    args = parser.parse_args()
    generate_configs(
        existing_only=args.existing_only, delete_if_in_wandb=args.delete_if_in_wandb
    )


if __name__ == "__main__":
    main()
