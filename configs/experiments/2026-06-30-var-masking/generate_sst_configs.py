"""Generate SST-perturbation inference configs for the VarMaskingC96 checkpoints.

Three free-running inference configs are written to ``run_configs/``, one per
constant SST perturbation level (p0k / p2k / p4k). Each config mounts its
checkpoint at ``/ckpt.tar`` (supplied per-run by submit_sst_jobs.py) and runs a
prognostic forecast with the SST forcing shifted by a constant amplitude.

The configs are run-agnostic: the per-run checkpoint dataset is provided at
submit time, so the same three configs are reused across every training run.
They are consumed by ``python -m fme.ace.inference`` (via run-ace-inference.sh),
not by the evaluator suite.
"""

import argparse
import pathlib

import yaml
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

# Free-running inference settings (shared by every perturbation level).
N_FORWARD_STEPS = 1460
FORWARD_STEPS_IN_MEMORY = 40
DATA_PATH = "/climate-default/"
DATASET_FILENAME = "2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr"
INITIAL_CONDITION_TIME = "1979-01-01T00:00:00"


def sst_config_filename(level: str) -> str:
    return f"{SST_CONFIG_PREFIX}{level}.yaml"


def _build_inference_config(amplitude: float) -> dict:
    return {
        "checkpoint_path": CHECKPOINT_PATH,
        "allow_incompatible_dataset": True,
        "experiment_dir": "/results",
        "n_forward_steps": N_FORWARD_STEPS,
        "forward_steps_in_memory": FORWARD_STEPS_IN_MEMORY,
        "data_writer": {
            "save_monthly_files": False,
            "save_prediction_files": False,
        },
        "initial_condition": {
            "path": f"{DATA_PATH}{DATASET_FILENAME}",
            "engine": "zarr",
            "start_indices": {"times": [INITIAL_CONDITION_TIME]},
        },
        "forcing_loader": {
            "dataset": {
                "data_path": DATA_PATH,
                "file_pattern": DATASET_FILENAME,
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


def generate_configs(existing_only: bool = False) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for level, amplitude in SST_PERTURBATIONS.items():
        out_path = RUN_CONFIGS_DIR / sst_config_filename(level)
        if existing_only and not out_path.exists():
            print(f"Skipped {out_path.name}")
            continue
        cfg = _build_inference_config(amplitude)
        out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        print(f"Wrote {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite SST configs that already exist.",
    )
    args = parser.parse_args()
    generate_configs(existing_only=args.existing_only)


if __name__ == "__main__":
    main()
