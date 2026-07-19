"""Generate evaluator suite configs for the var-masking training runs.

Each suite config contains all inline inference entries from the corresponding
training config.  submit_eval_jobs.py submits one job per checkpoint, and that
job runs all entries in the suite under one WandB run.

Training runs are enumerated in memory for every baseline version (default:
both -v1 and -v2) across both the masking family (generate_masking_configs.py)
and the seed-replicate family (generate_seed_configs.py), so eval configs are
produced for all of them in one pass.  This does not depend on the source
training configs sitting in ``run_configs/``: the generators wipe ``*.yaml`` on
each run, so v1/v2 and mask/seed never coexist on disk.

An eval config is written for every training run that has finished in wandb
(i.e. has a Beaker result dataset in the source map).  With ``--delete-if-in
-wandb``, eval configs whose evaluator runs have themselves already finished are
deleted instead, leaving only the eval runs not yet finished.

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
    BASE_CONFIG_FILENAMES,
    CONFIG_PREFIX,
    RUN_CONFIGS_DIR,
    WANDB_ENTITY,
    WANDB_PREFIX,
    WANDB_PROJECT,
    config_name_to_run_name,
)
from generate_masking_configs import iter_train_configs as iter_masking_train_configs
from generate_seed_configs import DEFAULT_N_SEEDS
from generate_seed_configs import iter_train_configs as iter_seed_train_configs

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


def eval_suite_config_to_run_name(config_filename: str) -> str:
    """Wandb run name for an eval suite config filename (stem ends in -v1/-v2)."""
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


def delete_eval_configs_in_wandb(project: str) -> None:
    """Delete on-disk eval suite configs whose checkpoint runs all exist in wandb.

    Driven off the eval suite config files present in ``RUN_CONFIGS_DIR`` rather
    than the source training configs, so it covers every version's eval configs
    (e.g. both -v1 and -v2) regardless of which training configs currently sit in
    the directory or whether they appear in the beaker map.
    """
    wandb_run_names = _fetch_wandb_run_names(project)
    eval_configs = sorted(RUN_CONFIGS_DIR.glob(f"{EVAL_SUITE_CONFIG_PREFIX}*.yaml"))
    for out_path in eval_configs:
        base_run_name = eval_suite_config_to_run_name(out_path.name)
        expected_runs = {
            f"{base_run_name}{suffix}" for suffix in CHECKPOINT_RUN_SUFFIXES
        }
        if expected_runs - wandb_run_names:
            continue
        out_path.unlink()
        print(f"Deleted {out_path.name} (all runs exist in wandb)")


def generate_eval_config(
    config_name: str,
    train_cfg: dict,
    source_map: dict[str, str],
    inference_names: list[str] | None,
    checkpoint_path: str,
    existing_only: bool,
    delete_if_in_wandb: bool = False,
) -> None:
    source_run_name = config_name_to_run_name(config_name)
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
            f"Skipped {config_name} (run {source_run_name!r} not "
            f"finished in wandb: state={run_state!r})"
        )
        return

    cfg = _build_eval_suite_config(
        train_cfg=train_cfg,
        inference_names=inference_names,
        checkpoint_path=checkpoint_path,
    )
    out_path = RUN_CONFIGS_DIR / source_config_to_eval_suite_config(
        f"{config_name}.yaml"
    )
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
        "--version",
        "-v",
        choices=sorted(BASE_CONFIG_FILENAMES),
        default=None,
        help="Restrict to source configs of this baseline version (default: all).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help=f"Number of seeds per seed-config group (default: {DEFAULT_N_SEEDS}).",
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

    if args.delete_if_in_wandb:
        # Standalone pass over on-disk eval configs so cleanup covers every
        # version present (both -v1 and -v2), not just the version whose training
        # configs currently sit in run_configs. Generation below still (re)writes
        # eval configs for runs not yet finished in wandb.
        delete_eval_configs_in_wandb(WANDB_PROJECT)

    # Enumerate every training run in memory (masking + seed families) for the
    # requested version(s), so eval configs are produced for all of them without
    # the source training configs needing to sit in run_configs at once (the
    # generators wipe *.yaml, so v1/v2 and mask/seed never coexist on disk).
    versions = [args.version] if args.version else sorted(BASE_CONFIG_FILENAMES)
    for version in versions:
        train_configs = iter_masking_train_configs(version) + iter_seed_train_configs(
            version, args.n_seeds
        )
        for config_name, train_cfg in train_configs:
            generate_eval_config(
                config_name=config_name,
                train_cfg=train_cfg,
                source_map=source_map,
                inference_names=args.inference_name,
                checkpoint_path=args.checkpoint_path,
                existing_only=args.existing_only,
                delete_if_in_wandb=args.delete_if_in_wandb,
            )


if __name__ == "__main__":
    main()
