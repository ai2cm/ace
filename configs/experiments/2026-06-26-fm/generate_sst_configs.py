"""Generate SST-perturbation inference configs for the FM training runs.

Free-running inference configs are written to ``run_configs/``, one per
(forcing grid, constant SST perturbation level) pair. The forcing grids are
the two native training datasets (``era5`` and ``c96``) and the perturbation
levels are p0k / p2k / p4k. Each config mounts its checkpoint at ``/ckpt.tar``
(supplied per-run by submit_sst_jobs.py) and runs a prognostic forecast with
the SST forcing shifted by a constant amplitude.

The configs are run-agnostic: the per-run checkpoint dataset is provided at
submit time, so the same configs are reused across every training run. Which
grids a given run is submitted against is decided by ``run_grids`` from the
training regime in the run name: C96-trained runs (``*-c96-*``) only run on
C96, ERA5-trained runs (``*-era5-*`` and the hand-written ``nc-sfno-vN``) only
on ERA5, and FM runs (``*-fm-*``) on both. Source configs are the hand-written
runs in base_configs and the generated norm-ablation cells in run_configs.

Each config carries the label of its forcing dataset (``era5`` / ``amip``) as
``labels`` on the InferenceConfig, which sets it on both the initial condition
and the forcing windows. The norm-ablation A2/A3 checkpoints resolve their
per-group normalization from these labels; without them, the unconditional
cells would silently normalize against ``default_group`` and the ``-cond``
cells would refuse to run. A1 and the hand-written runs have no grouped
normalization and ignore the labels. The configs are consumed by
``python -m fme.ace.inference`` (via run-ace-inference.sh), not by the
evaluator suite.
"""

import argparse
import pathlib
from typing import NamedTuple

import yaml
from generate_eval_configs import (
    ARCHITECTURES,
    BASE_CONFIGS_DIR,
    RUN_CONFIGS_DIR,
    TRAINING_RESULT_DATASETS,
    WANDB_ENTITY,
    WANDB_PREFIX,
    WANDB_PROJECT,
    _fetch_wandb_run_names,
    discover_source_configs,
    source_config_to_run_name,
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


class DatasetSpec(NamedTuple):
    data_path: str
    file_pattern: str
    n_forward_steps: int
    label: str


# The two native forcing datasets of the FM training runs. ``n_forward_steps``
# matches the corresponding "long" inline-inference entry in the training
# configs (long_46year for ERA5, long_43year for the C96 AMIP ensemble).
# ``label`` is the dataset label the norm-ablation training configs give the
# same dataset (generate_norm_ablation_configs.py), so a grouped-normalization
# checkpoint resolves the group it was trained with.
DATASETS = {
    "era5": DatasetSpec(
        data_path="/climate-default",
        file_pattern="2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr",
        n_forward_steps=16794,
        label="era5",
    ),
    "c96": DatasetSpec(
        data_path=(
            "/climate-default/"
            "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-"
            "ensemble-dataset"
        ),
        file_pattern="ic_0001.zarr",
        n_forward_steps=15683,
        label="amip",
    ),
}

# Free-running inference settings (shared by every config).
FORWARD_STEPS_IN_MEMORY = 73
INITIAL_CONDITION_TIME = "1979-01-01T00:00:00"


def sst_config_filename(grid: str, level: str) -> str:
    return f"{SST_CONFIG_PREFIX}{grid}-{level}.yaml"


def sst_job_name(run_name: str, grid: str, level: str) -> str:
    return f"{run_name}-sst-{grid}-{level}"


def run_grids(run_name: str) -> tuple[str, ...]:
    """Forcing grids a training run should produce SST-perturbation results
    on: C96-trained runs only C96, ERA5-trained runs only ERA5, FM runs
    (trained on both) both.

    The regime is the segment after the architecture tag in the run name
    (``nc-sfno-c96-a1`` -> ``c96``, ``nc-swin-v2-fm-a2-cond`` -> ``fm``), so
    the rule is the same for every architecture. The hand-written ERA5 runs
    (``nc-sfno-v2``) have no regime segment and fall through to ERA5.
    """
    suffix = run_name.removeprefix(WANDB_PREFIX)
    for arch in ARCHITECTURES:
        if suffix.startswith(f"{arch}-"):
            regime = suffix.removeprefix(f"{arch}-").split("-", 1)[0]
            break
    else:
        regime = ""
    if regime == "c96":
        return ("c96",)
    if regime == "fm":
        return ("era5", "c96")
    return ("era5",)


def sst_runs(version: str | None = None) -> dict[str, tuple[str, ...]]:
    """Primary training run name -> forcing grids to perturb on, for every
    base or norm-ablation training config with a recorded training result
    dataset.
    """
    runs: dict[str, tuple[str, ...]] = {}
    for source_path in discover_source_configs(
        version,
        architectures=ARCHITECTURES,
        source_dirs=(BASE_CONFIGS_DIR, RUN_CONFIGS_DIR),
    ):
        run_name = source_config_to_run_name(source_path.name)
        if run_name not in TRAINING_RESULT_DATASETS:
            # No training result dataset recorded for this run yet; skip
            # rather than halt. Matches generate_eval_configs.py.
            print(f"Skipped {source_path.name} (no dataset ID for {run_name!r})")
            continue
        runs[run_name] = run_grids(run_name)
    return runs


def _build_inference_config(spec: DatasetSpec, amplitude: float) -> dict:
    return {
        "checkpoint_path": CHECKPOINT_PATH,
        "allow_incompatible_dataset": True,
        "experiment_dir": "/results",
        "n_forward_steps": spec.n_forward_steps,
        "forward_steps_in_memory": FORWARD_STEPS_IN_MEMORY,
        # Sets the label on the initial condition and every forcing window;
        # see the module docstring for why the grouped-normalization arms
        # need it. Putting ``labels`` on the forcing dataset alone would leave
        # the initial condition unlabeled and fail the stepper's IC/forcing
        # label agreement check.
        "labels": [spec.label],
        # Daily and monthly netCDF output, so the response maps can be built
        # from the data rather than decoded from the logged wandb images.
        "data_writer": {
            "save_monthly_files": True,
            "save_prediction_files": True,
        },
        "initial_condition": {
            "path": f"{spec.data_path}/{spec.file_pattern}",
            "engine": "zarr",
            "start_indices": {"times": [INITIAL_CONDITION_TIME]},
        },
        "forcing_loader": {
            "dataset": {
                "data_path": spec.data_path,
                "file_pattern": spec.file_pattern,
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
        "logging": {"project": WANDB_PROJECT, "entity": WANDB_ENTITY},
    }


def _all_runs_finished_in_wandb(
    grid: str,
    level: str,
    runs: dict[str, tuple[str, ...]],
    wandb_run_names: set[str],
) -> bool:
    """True if every applicable training run already has a wandb run for this
    (grid, level) pair (job names submitted by submit_sst_jobs.py).
    """
    expected = [
        sst_job_name(run_name, grid, level)
        for run_name, grids in runs.items()
        if grid in grids
    ]
    return bool(expected) and all(name in wandb_run_names for name in expected)


def generate_configs(
    existing_only: bool = False, delete_if_in_wandb: bool = False
) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    wandb_run_names: set[str] | None = None
    runs: dict[str, tuple[str, ...]] = {}
    if delete_if_in_wandb:
        print("Fetching run names from wandb...")
        wandb_run_names = _fetch_wandb_run_names()
        print(f"Found {len(wandb_run_names)} existing runs.")
        runs = sst_runs()
    for grid in DATASETS:
        for level, amplitude in SST_PERTURBATIONS.items():
            out_path = RUN_CONFIGS_DIR / sst_config_filename(grid, level)
            if wandb_run_names is not None and _all_runs_finished_in_wandb(
                grid, level, runs, wandb_run_names
            ):
                if out_path.exists():
                    out_path.unlink()
                    print(f"Deleted {out_path.name} (all runs exist in wandb)")
                else:
                    print(f"Skipped {out_path.name} (all runs exist in wandb, no file)")
                continue
            if existing_only and not out_path.exists():
                print(f"Skipped {out_path.name}")
                continue
            cfg = _build_inference_config(DATASETS[grid], amplitude)
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
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            "Delete (grid, perturbation-level) configs whose applicable "
            "training runs all already have a finished SST-perturbation run "
            "in wandb."
        ),
    )
    args = parser.parse_args()
    generate_configs(
        existing_only=args.existing_only, delete_if_in_wandb=args.delete_if_in_wandb
    )


if __name__ == "__main__":
    main()
