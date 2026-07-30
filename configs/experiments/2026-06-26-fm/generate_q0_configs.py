"""Generate Q0-swap evaluator suite configs for the FM training runs.

Sibling to generate_orography_configs.py: the same experiment, but swapping
the lowest-index specific total water level (``specific_total_water_0``, the
model top) rather than ``HGTsfc``. Q0 varies in time and is prognostic, so it
cannot be sourced from another grid's store the way orography is. Instead each
inference entry gets:

  - ``loader.dataset.constant_field_override`` pointing at a grid's precomputed
    time-mean store, so the dataset serves that grid's time-mean Q0 unchanged
    at every timestep, and
  - ``stepper_override.prescribed_prognostic_names: [specific_total_water_0]``,
    so the stepper overwrites its own Q0 prediction with that field on every
    step instead of evolving it freely.

Together those hold Q0 fixed at one grid's time mean for the whole rollout.

Generated for the FM (multi-dataset) runs and for runs trained with Q0 input
masking; the remaining single-dataset runs are not part of this experiment.
Both grid variants are generated for every run,
and each run's suite already contains inference entries on both the ERA5 and
C96 datasets, so the two configs cover all four (forcing dataset, Q0 grid)
combinations.
"""

import argparse
import copy
import json
import pathlib

import yaml
from _version_select import add_version_arg
from generate_eval_configs import (
    CONFIG_PREFIX,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_SOURCE_MAP,
    RUN_CONFIGS_DIR,
    WANDB_PREFIX,
    _build_eval_suite_config,
    _fetch_wandb_finished_summaries,
    _fetch_wandb_run_names,
    _write_config,
    discover_source_configs,
    eval_suite_config_to_run_name,
    source_config_to_run_name,
)
from generate_eval_configs import EVAL_SUITE_CONFIG_PREFIX as _EVAL_SUITE_CONFIG_PREFIX

Q0_EVAL_SUITE_CONFIG_PREFIX = f"{_EVAL_SUITE_CONFIG_PREFIX}q0-"

# The prognostic variable held constant. Present in both in_names and
# out_names of every FM stepper, which prescribed_prognostic_names requires.
Q0_NAME = "specific_total_water_0"

# Q0-swap evals only ever run the best-inference checkpoint (not
# besttrain/lastepoch) -- submit_q0_jobs.py sources it directly from the
# corresponding non-swapped run's own result dataset, so there is no separate
# "Q0 training run" or dataset to record.
Q0_CHECKPOINT_SUFFIXES = ("-bestinf",)

# Precomputed time-mean stores for each grid, on the same weka mount the
# training configs read their normalization statistics from. These are the
# per-source subdirectories (not `combined/`), matching the subdirectory each
# grid's training config already uses for centering/scaling: the ERA5 store
# for ERA5, and the ic_0001 member for the C96 AMIP ensemble. Both are on the
# same 4-degree target grid, so either grid's Q0 field is spatially
# compatible with any inference entry's dataset.
Q0_TIME_MEAN_PATHS = {
    "era5": (
        "/climate-default/2026-04-17-era5-4deg-8layer-daily-stats-1990-2019/"
        "2026-03-19-era5-4deg-8layer-1940-2025/time-mean.nc"
    ),
    "c96": (
        "/climate-default/"
        "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-"
        "dataset-stats/ic_0001/time-mean.nc"
    ),
}


def _is_fm_run(run_name: str) -> bool:
    """True for the multi-dataset (foundation model) training runs."""
    return run_name.removeprefix(WANDB_PREFIX).startswith("nc-sfno-fm")


def _masks_q0_in_training(train_cfg: dict) -> bool:
    """True if the run's input dropout masks Q0 during training.

    A single-dataset run that was trained with Q0 masked is also a valid
    subject for the Q0 swap: it has learned to run with Q0 supplied rather
    than freely evolved, which is exactly what the swap prescribes.
    """
    dropout = train_cfg["stepper"]["step"]["config"].get("input_dropout")
    if dropout is None:
        return False
    return any(
        Q0_NAME in group.get("variables", [])
        for group in dropout.get("override_groups", [])
    )


def _is_q0_eval_run(run_name: str, train_cfg: dict) -> bool:
    return _is_fm_run(run_name) or _masks_q0_in_training(train_cfg)


def _apply_q0_override(dataset: dict, grid: str) -> dict:
    """Return `dataset` with Q0 sourced from `grid`'s time mean.

    Unlike the orography swap there is no "already on this grid" shortcut: the
    point is to hold Q0 constant in time, so the override is applied even when
    the dataset and the time mean come from the same grid.
    """
    if "data_path" not in dataset:
        raise ValueError(
            f"Dataset {dataset!r} is not a plain data_path-keyed dict; "
            "concat/merge inference datasets are not supported by this script."
        )
    return {
        **copy.deepcopy(dataset),
        "constant_field_override": {
            "path": Q0_TIME_MEAN_PATHS[grid],
            "names": [Q0_NAME],
        },
    }


def source_config_to_q0_eval_suite_config(config_filename: str, grid: str) -> str:
    suffix = pathlib.Path(config_filename).stem.removeprefix(CONFIG_PREFIX)
    return f"{Q0_EVAL_SUITE_CONFIG_PREFIX}{grid}-{suffix}.yaml"


def generate_q0_eval_config(
    source_path: pathlib.Path,
    source_map: dict[str, str],
    inference_names: list[str] | None,
    checkpoint_path: str,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
    wandb_finished_summaries: dict[str, list[set[str]]] | None = None,
) -> None:
    source_run_name = source_config_to_run_name(source_path.name)
    source_dataset_id = source_map.get(source_run_name)
    if source_dataset_id is None:
        # No training result dataset recorded for this run yet (e.g. a config
        # not present in the source map). Skip rather than halt the whole run.
        print(f"Skipped {source_path.name} (no dataset ID for {source_run_name!r})")
        return

    with source_path.open() as f:
        train_cfg = yaml.safe_load(f)

    if not _is_q0_eval_run(source_run_name, train_cfg):
        print(f"Skipped {source_path.name} (not a multi-dataset FM or Q0-masked run)")
        return

    for grid in Q0_TIME_MEAN_PATHS:
        cfg = _build_eval_suite_config(
            train_cfg=train_cfg,
            inference_names=inference_names,
            checkpoint_path=checkpoint_path,
        )
        for entry in cfg["inferences"]:
            entry_cfg = entry["config"]
            loader = entry_cfg["loader"]
            loader["dataset"] = _apply_q0_override(loader["dataset"], grid)
            entry_cfg["stepper_override"] = {"prescribed_prognostic_names": [Q0_NAME]}
        out_path = RUN_CONFIGS_DIR / source_config_to_q0_eval_suite_config(
            source_path.name, grid
        )
        _write_config(
            cfg,
            out_path,
            source_run_name,
            source_dataset_id,
            existing_only,
            wandb_run_names,
            eval_run_name_base=eval_suite_config_to_run_name(out_path.name),
            checkpoint_suffixes=Q0_CHECKPOINT_SUFFIXES,
            wandb_finished_summaries=wandb_finished_summaries,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
    parser.add_argument(
        "--inference-name",
        nargs="+",
        default=None,
        help="Inline inference entry name(s) to export (default: all entries).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to the mounted checkpoint (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    parser.add_argument(
        "--source-map",
        metavar="PATH",
        default=DEFAULT_SOURCE_MAP,
        help=(
            "JSON file mapping training run name → Beaker dataset ID"
            f" (default: {DEFAULT_SOURCE_MAP})."
        ),
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite evaluator configs that already exist.",
    )
    parser.add_argument(
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            "Delete/skip eval suites whose checkpoint runs all already exist "
            "in wandb."
        ),
    )
    parser.add_argument(
        "--skip-if-in-wandb",
        action="store_true",
        help=(
            "Delete/skip eval suites whose checkpoint runs all finished in "
            "wandb with every inference entry logged. Stricter than "
            "--delete-if-in-wandb, which only checks that runs with the "
            "expected names exist."
        ),
    )
    args = parser.parse_args()

    with open(args.source_map) as f:
        source_map: dict[str, str] = json.load(f)

    wandb_run_names: set[str] | None = None
    if args.delete_if_in_wandb:
        print("Fetching run names from wandb...")
        wandb_run_names = _fetch_wandb_run_names()
        print(f"Found {len(wandb_run_names)} existing runs.")

    wandb_finished_summaries: dict[str, list[set[str]]] | None = None
    if args.skip_if_in_wandb:
        print("Fetching finished runs from wandb...")
        wandb_finished_summaries = _fetch_wandb_finished_summaries()
        print(f"Found {len(wandb_finished_summaries)} finished run names.")

    source_configs = discover_source_configs(args.version)

    for source_path in source_configs:
        generate_q0_eval_config(
            source_path=source_path,
            source_map=source_map,
            inference_names=args.inference_name,
            checkpoint_path=args.checkpoint_path,
            existing_only=args.existing_only,
            wandb_run_names=wandb_run_names,
            wandb_finished_summaries=wandb_finished_summaries,
        )


if __name__ == "__main__":
    main()
