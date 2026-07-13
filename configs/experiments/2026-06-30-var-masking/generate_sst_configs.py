"""Generate SST-perturbation inference configs for the VarMaskingC96/ERA5
checkpoints.

Free-running inference configs are written to ``run_configs/``, one per
(model x constant SST perturbation level) combination. Each model uses the
single varying-CO2 forcing dataset native to its checkpoints (C96 SHIELD-AMIP
ensemble member ic_0001, or the ERA5 reanalysis zarr); the perturbation levels
are p0k / p2k / p4k. Each config mounts its checkpoint at ``/ckpt.tar``
(supplied per-run by submit_sst_jobs.py) and runs a prognostic forecast with
the SST forcing shifted by a constant amplitude.

The configs are run-agnostic: the per-run checkpoint dataset is provided at
submit time, so the same configs are reused across every training run of a
given model. They are consumed by ``python -m fme.ace.inference`` (via
run-ace-inference.sh), not by the evaluator suite.
"""

import argparse
import pathlib
from typing import NamedTuple

import yaml
from generate_eval_configs import TRAINING_RESULT_DATASETS, _fetch_wandb_run_names
from generate_masking_configs import (
    BASE_MODELS,
    CONFIG_PREFIX,
    RUN_CONFIGS_DIR,
    WANDB_PREFIX,
)

HERE = pathlib.Path(__file__).parent
SST_CONFIG_PREFIX = "ace-inference-sst-config-4deg-"
CHECKPOINT_PATH = "/ckpt.tar"

# Constant SST perturbation amplitudes (Kelvin), keyed by config/job suffix.
SST_PERTURBATIONS = {
    "p0k": 0.0,
    "p2k": 2.0,
    "p4k": 4.0,
}

# BASE_MODELS entries, keyed by the model key used in config/job names below.
MODELS = {
    "c96": BASE_MODELS[0],
    "era5": BASE_MODELS[1],
}


class DatasetSpec(NamedTuple):
    data_path: str
    file_pattern: str
    n_forward_steps: int


# Varying-CO2 forcing dataset native to each model's checkpoints, keyed by
# config/job suffix. ``data_path`` is the zarr directory; ``file_pattern`` is
# the zarr name within it. ``n_forward_steps`` matches each model's own
# "long"-inference baseline entry (era5 and c96 datasets don't span the same
# number of days from the shared initial condition).
DATASETS = {
    "c96": DatasetSpec(
        data_path="/climate-default/2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset",  # noqa: E501
        file_pattern="ic_0001.zarr",
        n_forward_steps=15683,
    ),
    "era5": DatasetSpec(
        data_path="/climate-default",
        file_pattern="2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr",
        n_forward_steps=16794,
    ),
}

# Free-running inference settings (shared by every config).
FORWARD_STEPS_IN_MEMORY = 40
INITIAL_CONDITION_TIME = "1979-01-01T00:00:00"


def sst_config_filename(model: str, level: str) -> str:
    return f"{SST_CONFIG_PREFIX}{model}-{level}.yaml"


def _model_for_run(run_name: str) -> str:
    suffix = run_name.removeprefix(WANDB_PREFIX)
    for model_key, base_model in MODELS.items():
        stem_suffix = base_model.stem.removeprefix(CONFIG_PREFIX)
        if suffix == stem_suffix or suffix.startswith(f"{stem_suffix}-"):
            return model_key
    raise ValueError(f"cannot determine model for run {run_name!r}")


def _all_runs_finished_in_wandb(model: str, level: str) -> bool:
    """True if every training run for ``model`` already has a finished
    wandb run for this SST perturbation level (job name ``{run_name}-{level}``,
    submitted by submit_sst_jobs.py into the run's own training project).
    """
    project = MODELS[model].project
    finished_runs = _fetch_wandb_run_names(project)
    runs_for_model = [
        run_name
        for run_name in TRAINING_RESULT_DATASETS
        if _model_for_run(run_name) == model
    ]
    return all(f"{run_name}-{level}" in finished_runs for run_name in runs_for_model)


def _build_inference_config(
    data_path: str,
    file_pattern: str,
    n_forward_steps: int,
    amplitude: float,
    project: str,
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
        "logging": {"project": project},
    }


def generate_configs(
    existing_only: bool = False, skip_if_in_wandb: bool = False
) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for model, spec in DATASETS.items():
        project = MODELS[model].project
        for level, amplitude in SST_PERTURBATIONS.items():
            out_path = RUN_CONFIGS_DIR / sst_config_filename(model, level)
            if skip_if_in_wandb and _all_runs_finished_in_wandb(model, level):
                print(f"Skipped {out_path.name} (all runs exist in wandb)")
                continue
            if existing_only and not out_path.exists():
                print(f"Skipped {out_path.name}")
                continue
            cfg = _build_inference_config(
                spec.data_path,
                spec.file_pattern,
                spec.n_forward_steps,
                amplitude,
                project,
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
    parser.add_argument(
        "--skip-if-in-wandb",
        action="store_true",
        help=(
            "Skip (model, level) configs whose training runs all already have "
            "a finished SST-perturbation run in wandb."
        ),
    )
    args = parser.parse_args()
    generate_configs(
        existing_only=args.existing_only, skip_if_in_wandb=args.skip_if_in_wandb
    )


if __name__ == "__main__":
    main()
