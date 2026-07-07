"""Generate evaluator suite configs for the FM training runs.

Each suite config contains all inline inference entries from the corresponding
training config.  submit_eval_jobs.py submits one job per checkpoint, and that
job runs all entries in the suite under one WandB run.
"""

import argparse
import copy
import json
import pathlib

import yaml
from _version_select import add_version_arg, stem_matches_version

# WandB project and run-name convention shared across the FM submit scripts.
# A training config stem "{CONFIG_PREFIX}{suffix}" maps to run name
# "{WANDB_PREFIX}{suffix}" (matches submit_fm_jobs.py); the version tag
# -v1 / -v2 is part of {suffix}.
WANDB_PROJECT = "FM"
WANDB_ENTITY = "ai2cm"
WANDB_PREFIX = "ace2-fm-"
CONFIG_PREFIX = "ace-train-config-4deg-AIMIP-"

# Eval run-name suffixes, one per evaluated checkpoint. A training run's eval
# is complete in wandb when a run exists for every suffix. Source of truth for
# submit_eval_jobs.py's CHECKPOINTS and for --delete-if-in-wandb below.
EVAL_CHECKPOINT_NAME_SUFFIXES = ("-besttrain", "-bestinf", "-lastepoch")

HERE = pathlib.Path(__file__).parent
EVAL_SUITE_CONFIG_PREFIX = "ace-eval-suite-config-4deg-AIMIP-"
DEFAULT_CHECKPOINT_PATH = "/ckpt.tar"
DEFAULT_SOURCE_MAP = str(HERE / "wandb_to_beaker_map.json")

# Mapping of training run name -> Beaker result dataset ID, loaded from the
# source map. Consumed by submit_eval_jobs.py to locate each run's checkpoints.
with open(DEFAULT_SOURCE_MAP) as _f:
    TRAINING_RESULT_DATASETS: dict[str, str] = json.load(_f)


def source_config_to_run_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}"


def eval_suite_config_to_run_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(EVAL_SUITE_CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}"


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


def _fetch_wandb_run_names() -> set[str]:
    import wandb  # lazy import: only needed with --delete-if-in-wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    return {run.name for run in runs}


def _write_config(
    cfg: dict,
    out_path: pathlib.Path,
    source_run_name: str,
    source_dataset_id: str,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
) -> None:
    if wandb_run_names is not None:
        eval_run_names = [
            f"{source_run_name}{suffix}" for suffix in EVAL_CHECKPOINT_NAME_SUFFIXES
        ]
        if all(name in wandb_run_names for name in eval_run_names):
            if out_path.exists():
                out_path.unlink()
                print(f"Deleted {out_path.name} (all eval runs exist in wandb)")
            else:
                print(f"Skipped {out_path.name} (all eval runs exist in wandb)")
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

    cfg = _build_eval_suite_config(
        train_cfg=train_cfg,
        inference_names=inference_names,
        checkpoint_path=checkpoint_path,
    )
    out_path = HERE / source_config_to_eval_suite_config(source_path.name)
    _write_config(
        cfg,
        out_path,
        source_run_name,
        source_dataset_id,
        existing_only,
        wandb_run_names,
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
            f"Delete/skip eval suites whose checkpoint runs all already exist "
            f"in {WANDB_ENTITY}/{WANDB_PROJECT}."
        ),
    )
    args = parser.parse_args()

    with open(args.source_map) as f:
        source_map: dict[str, str] = json.load(f)

    wandb_run_names: set[str] | None = None
    if args.delete_if_in_wandb:
        print(f"Fetching run names from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_run_names = _fetch_wandb_run_names()
        print(f"Found {len(wandb_run_names)} existing runs.")

    source_configs = sorted(
        p
        for p in HERE.glob("*.yaml")
        if p.name.startswith(CONFIG_PREFIX)
        and "nc-sfno" in p.name
        and stem_matches_version(p.stem, args.version)
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
            wandb_run_names=wandb_run_names,
        )


if __name__ == "__main__":
    main()
