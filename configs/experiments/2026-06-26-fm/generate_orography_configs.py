"""Generate orography-swap evaluator suite configs for the FM training runs.

Sibling to generate_eval_configs.py: produces the same eval suites (one per
inline `inference` entry, three checkpoints per training run), but forces
every inference entry's `loader.dataset` to source `HGTsfc` from a specific
grid (`era5` or `c96`) via `XarrayDataConfig.orography_override`, regardless
of which grid the run actually trained on. Both grid variants are generated
unconditionally for every run.
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
    _build_eval_suite_config,
    _fetch_wandb_run_names,
    _write_config,
    discover_source_configs,
    eval_suite_config_to_run_name,
    source_config_to_run_name,
)
from generate_eval_configs import EVAL_SUITE_CONFIG_PREFIX as _EVAL_SUITE_CONFIG_PREFIX

OROGRAPHY_EVAL_SUITE_CONFIG_PREFIX = f"{_EVAL_SUITE_CONFIG_PREFIX}orog-"

# Orography-swap evals only ever run the best-inference checkpoint (not
# besttrain/lastepoch) -- submit_orography_jobs.py sources it directly from
# the corresponding non-orography run's own result dataset, so there is no
# separate "orography training run" or dataset to record.
OROGRAPHY_CHECKPOINT_SUFFIXES = ("-bestinf",)

# Plain data_path/file_pattern/engine dicts identifying each grid's native
# store, confirmed against the real GCS zarr stores. Both are the same 4°
# target grid, so `HGTsfc` sourced from either is spatially compatible with
# any inference entry's other requested variables.
OROGRAPHY_SOURCES = {
    "era5": {
        "data_path": "/climate-default",
        "file_pattern": "2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr",
        "engine": "zarr",
    },
    "c96": {
        "data_path": (
            "/climate-default/"
            "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-"
            "ensemble-dataset"
        ),
        "file_pattern": "ic_0001.zarr",
        "engine": "zarr",
    },
}


def _dataset_grid(dataset: dict) -> str | None:
    for grid, source in OROGRAPHY_SOURCES.items():
        if (
            dataset.get("data_path") == source["data_path"]
            and dataset.get("file_pattern") == source["file_pattern"]
        ):
            return grid
    return None


def _swap_hgtsfc_grid(dataset: dict, grid: str) -> dict:
    if "data_path" not in dataset:
        raise ValueError(
            f"Dataset {dataset!r} is not a plain data_path-keyed dict; "
            "concat/merge inference datasets are not supported by this script."
        )
    if _dataset_grid(dataset) == grid:
        return copy.deepcopy(dataset)
    return {
        **copy.deepcopy(dataset),
        "orography_override": copy.deepcopy(OROGRAPHY_SOURCES[grid]),
    }


def source_config_to_orography_eval_suite_config(
    config_filename: str, grid: str
) -> str:
    suffix = pathlib.Path(config_filename).stem.removeprefix(CONFIG_PREFIX)
    return f"{OROGRAPHY_EVAL_SUITE_CONFIG_PREFIX}{grid}-{suffix}.yaml"


def generate_orography_eval_config(
    source_path: pathlib.Path,
    source_map: dict[str, str],
    inference_names: list[str] | None,
    checkpoint_path: str,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
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

    for grid in ("era5", "c96"):
        cfg = _build_eval_suite_config(
            train_cfg=train_cfg,
            inference_names=inference_names,
            checkpoint_path=checkpoint_path,
        )
        for entry in cfg["inferences"]:
            loader = entry["config"]["loader"]
            loader["dataset"] = _swap_hgtsfc_grid(loader["dataset"], grid)
        out_path = RUN_CONFIGS_DIR / source_config_to_orography_eval_suite_config(
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
            checkpoint_suffixes=OROGRAPHY_CHECKPOINT_SUFFIXES,
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
    args = parser.parse_args()

    with open(args.source_map) as f:
        source_map: dict[str, str] = json.load(f)

    wandb_run_names: set[str] | None = None
    if args.delete_if_in_wandb:
        print("Fetching run names from wandb...")
        wandb_run_names = _fetch_wandb_run_names()
        print(f"Found {len(wandb_run_names)} existing runs.")

    source_configs = discover_source_configs(args.version)

    for source_path in source_configs:
        generate_orography_eval_config(
            source_path=source_path,
            source_map=source_map,
            inference_names=args.inference_name,
            checkpoint_path=args.checkpoint_path,
            existing_only=args.existing_only,
            wandb_run_names=wandb_run_names,
        )


if __name__ == "__main__":
    main()
