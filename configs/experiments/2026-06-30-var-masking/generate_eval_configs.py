"""Generate evaluator suite configs for the var-masking training runs.

Each suite config contains all inline inference entries from the corresponding
training config in ``run_configs/``.  submit_eval_jobs.py submits one job per
checkpoint, and that job runs all entries in the suite under one WandB run.

The generated suite configs are written into ``run_configs/`` (alongside the
training and cooldown configs), where run-ace-eval.sh reads them.
"""

import argparse
import copy
import functools
import json
import pathlib

import yaml
from generate_masking_configs import (
    CONFIG_PREFIX,
    RUN_CONFIGS_DIR,
    WANDB_ENTITY,
    WANDB_PREFIX,
    WANDB_PROJECT,
    WANDB_SUFFIX,
)

HERE = pathlib.Path(__file__).parent
EVAL_SUITE_CONFIG_PREFIX = "ace-eval-suite-config-4deg-"
# Each eval suite config produces one wandb run per checkpoint variant; the run
# name is the base name plus one of these suffixes (see submit_eval_jobs.py).
CHECKPOINT_RUN_SUFFIXES = ("-besttrain", "-bestinf", "-lastepoch")
DEFAULT_CHECKPOINT_PATH = "/ckpt.tar"
DEFAULT_SOURCE_MAP = str(HERE / "wandb_to_beaker_map.json")

# Mapping of training run name -> Beaker result dataset ID, loaded from the
# source map. Consumed by submit_eval_jobs.py to locate each run's checkpoints.
# The map is populated by update_beaker_map.py as training runs finish, so it
# may not exist yet (e.g. right after a base-config migration with no
# finished runs).
if pathlib.Path(DEFAULT_SOURCE_MAP).exists():
    with open(DEFAULT_SOURCE_MAP) as _f:
        TRAINING_RESULT_DATASETS: dict[str, str] = json.load(_f)
else:
    TRAINING_RESULT_DATASETS = {}


def source_config_to_run_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


def eval_suite_config_to_run_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(EVAL_SUITE_CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


def source_config_to_eval_suite_config(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{EVAL_SUITE_CONFIG_PREFIX}{suffix}.yaml"


def _inference_entries(train_cfg: dict) -> list[dict]:
    entries = train_cfg.get("inference", [])
    if isinstance(entries, list):
        return entries
    return [entries]


def _resolve_inference_entries(
    train_cfg: dict,
    inference_names: list[str] | None,
) -> list[tuple[str, dict]]:
    entries = _inference_entries(train_cfg)
    resolved_entries = [
        (entry.get("name", f"inference_{i}"), entry) for i, entry in enumerate(entries)
    ]
    if inference_names is None:
        return resolved_entries

    entry_by_name = {name: entry for name, entry in resolved_entries}
    missing_names = sorted(set(inference_names) - set(entry_by_name))
    if missing_names:
        raise ValueError(
            f"Inference entries {missing_names!r} not found; "
            f"available entries: {list(entry_by_name)}"
        )
    return [(name, entry_by_name[name]) for name in inference_names]


def _build_eval_config(
    train_cfg: dict,
    inference_cfg: dict,
    inference_name: str,
    checkpoint_path: str,
) -> dict:
    return {
        "experiment_dir": f"/results/{inference_name}",
        "n_forward_steps": inference_cfg["n_forward_steps"],
        "forward_steps_in_memory": inference_cfg["forward_steps_in_memory"],
        "checkpoint_path": checkpoint_path,
        "logging": copy.deepcopy(train_cfg["logging"]),
        "loader": inference_cfg["loader"],
        "aggregator": inference_cfg.get("aggregator", {}),
        "data_writer": {
            "save_prediction_files": False,
            "save_monthly_files": False,
        },
        "n_ensemble_per_ic": inference_cfg.get("n_ensemble_per_ic", 1),
    }


def _build_eval_suite_config(
    train_cfg: dict,
    inference_names: list[str] | None,
    checkpoint_path: str,
) -> dict:
    inference_entries = _resolve_inference_entries(train_cfg, inference_names)
    return {
        "experiment_dir": "/results",
        "logging": copy.deepcopy(train_cfg["logging"]),
        "inferences": [
            {
                "name": inference_name,
                "config": _build_eval_config(
                    train_cfg=train_cfg,
                    inference_cfg=copy.deepcopy(inference_cfg),
                    inference_name=inference_name,
                    checkpoint_path=checkpoint_path,
                ),
            }
            for inference_name, inference_cfg in inference_entries
        ],
    }


@functools.cache
def _fetch_wandb_run_states(project: str) -> dict[str, str]:
    import wandb  # lazy import: only needed when wandb lookups are required

    print(f"Fetching run names from {WANDB_ENTITY}/{project}...")
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{project}")
    states = {run.name: run.state for run in runs}
    print(f"Found {len(states)} existing runs.")
    return states


def _fetch_wandb_run_names(project: str) -> set[str]:
    return {
        name
        for name, state in _fetch_wandb_run_states(project).items()
        if state == "finished"
    }


def _write_config(
    cfg: dict,
    out_path: pathlib.Path,
    source_run_name: str,
    source_dataset_id: str,
    existing_only: bool,
    project: str,
    delete_if_in_wandb: bool = False,
) -> None:
    if delete_if_in_wandb:
        wandb_run_names = _fetch_wandb_run_names(project)
        base_run_name = eval_suite_config_to_run_name(out_path.name)
        expected_runs = {
            f"{base_run_name}{suffix}" for suffix in CHECKPOINT_RUN_SUFFIXES
        }
        missing_runs = expected_runs - wandb_run_names
        if not missing_runs:
            if out_path.exists():
                out_path.unlink()
                print(f"Deleted {out_path.name} (all runs exist in wandb)")
            else:
                print(f"Skipped {out_path.name} (all runs exist in wandb, no file)")
            return
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return
    header = (
        f"# source_run: {source_run_name}\n" f"# source_dataset: {source_dataset_id}\n"
    )
    with out_path.open("w") as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}")


def generate_eval_config(
    source_path: pathlib.Path,
    source_map: dict[str, str],
    inference_names: list[str] | None,
    checkpoint_path: str,
    existing_only: bool,
    delete_if_in_wandb: bool = False,
) -> None:
    source_run_name = source_config_to_run_name(source_path.name)
    source_dataset_id = source_map.get(source_run_name)
    if source_dataset_id is None:
        run_state = _fetch_wandb_run_states(WANDB_PROJECT).get(source_run_name)
        if run_state == "finished":
            raise KeyError(
                f"Run {source_run_name!r} is finished in wandb but has no "
                "training result dataset ID configured; refresh "
                "wandb_to_beaker_map.json"
            )
        print(
            f"Skipped {source_path.name} (run {source_run_name!r} not "
            f"finished in wandb: state={run_state!r})"
        )
        return

    with source_path.open() as f:
        train_cfg = yaml.safe_load(f)

    cfg = _build_eval_suite_config(
        train_cfg=train_cfg,
        inference_names=inference_names,
        checkpoint_path=checkpoint_path,
    )
    out_path = RUN_CONFIGS_DIR / source_config_to_eval_suite_config(source_path.name)
    _write_config(
        cfg,
        out_path,
        source_run_name,
        source_dataset_id,
        existing_only,
        WANDB_PROJECT,
        delete_if_in_wandb,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
            f"Delete evaluator configs whose run name already exists in "
            f"the run's wandb project (entity {WANDB_ENTITY})."
        ),
    )
    args = parser.parse_args()

    with open(args.source_map) as f:
        source_map: dict[str, str] = json.load(f)

    source_configs = sorted(
        p
        for p in RUN_CONFIGS_DIR.glob("*-mask*.yaml")
        if p.name.startswith(CONFIG_PREFIX)
        and not p.name.endswith("-finetune.yaml")
        and not p.name.endswith("-cooldown.yaml")
        and not p.name.endswith("-bestinfcooldown.yaml")
    )

    for source_path in source_configs:
        generate_eval_config(
            source_path=source_path,
            source_map=source_map,
            inference_names=args.inference_name,
            checkpoint_path=args.checkpoint_path,
            existing_only=args.existing_only,
            delete_if_in_wandb=args.delete_if_in_wandb,
        )


if __name__ == "__main__":
    main()
