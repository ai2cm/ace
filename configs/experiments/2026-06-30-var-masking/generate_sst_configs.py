"""Generate SST-perturbation inference configs for the VarMaskingC96 checkpoints.

Free-running inference configs are written to ``run_configs/``, one per
(forcing dataset x constant SST perturbation level) combination. The two forcing
datasets are the C96 SHIELD-AMIP datasets used in training (constant-CO2 and the
varying-CO2 ensemble member ic_0001); the perturbation levels are p0k / p2k /
p4k. Each config mounts its checkpoint at ``/ckpt.tar`` (supplied per-run by
submit_sst_jobs.py) and runs a prognostic forecast with the SST forcing shifted
by a constant amplitude.

The configs are run-agnostic: the per-run checkpoint dataset is provided at
submit time, so the same configs are reused across every training run.
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

# Forcing datasets (C96 SHIELD-AMIP, native to the checkpoints), keyed by
# config/job suffix. ``data_path`` is the zarr directory; ``file_pattern`` is
# the zarr name within it.
DATASETS = {
    "constant-co2": {
        "data_path": "/climate-default/2026-07-01-vertically-resolved-c96-4deg-daily-shield-amip-constant-co2-dataset",  # noqa: E501
        "file_pattern": "AMIP-constant-CO2.zarr",
    },
    "varying-co2": {
        "data_path": "/climate-default/2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset",  # noqa: E501
        "file_pattern": "ic_0001.zarr",
    },
}

# Free-running inference settings (shared by every config).
N_FORWARD_STEPS = 15683
FORWARD_STEPS_IN_MEMORY = 40
INITIAL_CONDITION_TIME = "1979-01-01T00:00:00"


def sst_config_filename(dataset: str, level: str) -> str:
    return f"{SST_CONFIG_PREFIX}{dataset}-{level}.yaml"


def _build_inference_config(
    data_path: str, file_pattern: str, amplitude: float
) -> dict:
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


def generate_configs(existing_only: bool = False) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for dataset, spec in DATASETS.items():
        for level, amplitude in SST_PERTURBATIONS.items():
            out_path = RUN_CONFIGS_DIR / sst_config_filename(dataset, level)
            if existing_only and not out_path.exists():
                print(f"Skipped {out_path.name}")
                continue
            cfg = _build_inference_config(
                spec["data_path"], spec["file_pattern"], amplitude
            )
            out_path.write_text(
                yaml.dump(cfg, default_flow_style=False, sort_keys=False)
            )
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
