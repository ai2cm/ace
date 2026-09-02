"""Generate evaluator suite configs for the FM training runs.

Each suite config contains all inline inference entries from the corresponding
training config.  submit_eval_jobs.py submits one job per checkpoint, and that
job runs all entries in the suite under one WandB run.

Source training configs are read from both base_configs (the hand-written FM
runs) and run_configs (the generated norm-ablation cells), and can be narrowed
to one architecture with --arch.
"""

import argparse
import copy
import json
import pathlib
from collections.abc import Sequence

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

# Summary metric logged per inference entry at the end of its evaluation (see
# fme/ace/inference/evaluator.py); used by --skip-if-in-wandb to tell a suite
# that ran to completion from one that died partway through.
SUCCESS_METRIC = "total_steps_per_second"

HERE = pathlib.Path(__file__).parent
BASE_CONFIGS_DIR = HERE / "base_configs"
RUN_CONFIGS_DIR = HERE / "run_configs"
EVAL_SUITE_CONFIG_PREFIX = "ace-eval-suite-config-4deg-AIMIP-"

# Architecture tags appearing in the training config filenames, and the
# vocabulary of --arch here and in submit_eval_jobs.py. A config belongs to an
# architecture when its filename contains that tag; every config which has an
# eval suite carries exactly one of them.
ARCHITECTURES = ("nc-sfno", "nc-swin-v2")
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
        # Training mixes datasets on different vertical grids via strict:false and
        # pins the first concat member's coordinate into the checkpoint; inline
        # inference never checks compatibility. The standalone evaluator does, so
        # allow the same mismatch here to reproduce training's inference exactly.
        "allow_incompatible_dataset": True,
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


def fetch_wandb_finished_summaries() -> dict[str, list[set[str]]]:
    """Map run name -> summary key sets of that name's finished wandb runs.

    One key set per run, so a suite is only considered done if a *single* run
    logged every inference entry (a name reused across partial runs must not
    add up to a complete suite).
    """
    import wandb  # lazy import: only needed with --skip-if-in-wandb

    api = wandb.Api()
    summaries: dict[str, list[set[str]]] = {}
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"):
        if run.state != "finished":
            continue
        summaries.setdefault(run.name, []).append(set(run.summary.keys()))
    return summaries


def all_inferences_succeeded(
    cfg: dict,
    eval_run_names: list[str],
    wandb_finished_summaries: dict[str, list[set[str]]],
) -> bool:
    """True if every eval run logged SUCCESS_METRIC for every inference entry.

    run_eval_suite.py labels each entry's logs with the entry name, and the
    summary metric is only written once that entry's inference completes, so
    its presence means the entry ran through.
    """
    required_keys = {f"{entry['name']}/{SUCCESS_METRIC}" for entry in cfg["inferences"]}
    return all(
        any(
            required_keys <= summary_keys
            for summary_keys in wandb_finished_summaries.get(run_name, [])
        )
        for run_name in eval_run_names
    )


def _delete_or_skip(out_path: pathlib.Path, reason: str) -> None:
    if out_path.exists():
        out_path.unlink()
        print(f"Deleted {out_path.name} ({reason})")
    else:
        print(f"Skipped {out_path.name} ({reason})")


def _write_config(
    cfg: dict,
    out_path: pathlib.Path,
    source_run_name: str,
    source_dataset_id: str,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
    eval_run_name_base: str | None = None,
    checkpoint_suffixes: tuple[str, ...] = EVAL_CHECKPOINT_NAME_SUFFIXES,
    wandb_finished_summaries: dict[str, list[set[str]]] | None = None,
) -> None:
    eval_run_names = [
        f"{eval_run_name_base or source_run_name}{suffix}"
        for suffix in checkpoint_suffixes
    ]
    if wandb_run_names is not None:
        if all(name in wandb_run_names for name in eval_run_names):
            _delete_or_skip(out_path, "all eval runs exist in wandb")
            return
    if wandb_finished_summaries is not None:
        if all_inferences_succeeded(cfg, eval_run_names, wandb_finished_summaries):
            _delete_or_skip(out_path, "all inferences succeeded in wandb")
            return
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return
    header = (
        f"# source_run: {source_run_name}\n" f"# source_dataset: {source_dataset_id}\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        wandb_run_names,
        wandb_finished_summaries=wandb_finished_summaries,
    )


def discover_source_configs(
    version: str | None,
    architectures: Sequence[str] = ("nc-sfno",),
    source_dirs: Sequence[pathlib.Path] = (BASE_CONFIGS_DIR,),
) -> list[pathlib.Path]:
    """Training configs to build eval suites from, at most one per filename.

    The defaults are the scope the orography, fixed-variable and SST
    generators want: the nc-sfno configs in base_configs. main() below widens
    both, because the norm-ablation training configs are generated into
    run_configs and cover nc-swin-v2 as well. An earlier source dir wins a
    filename clash.
    """
    by_name: dict[str, pathlib.Path] = {}
    for source_dir in source_dirs:
        for path in sorted(source_dir.glob("*.yaml")):
            if not path.name.startswith(CONFIG_PREFIX):
                continue
            if not any(arch in path.name for arch in architectures):
                continue
            if not stem_matches_version(path.stem, version):
                continue
            if path.name.endswith(
                ("-finetune.yaml", "-cooldown.yaml", "-bestinfcooldown.yaml")
            ):
                continue
            by_name.setdefault(path.name, path)
    return [by_name[name] for name in sorted(by_name)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
    parser.add_argument(
        "--arch",
        nargs="+",
        choices=ARCHITECTURES,
        default=None,
        help="Only generate suites for these architectures (default: all).",
    )
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
    parser.add_argument(
        "--skip-if-in-wandb",
        action="store_true",
        help=(
            "Delete/skip eval suites whose checkpoint runs all finished in "
            f"{WANDB_ENTITY}/{WANDB_PROJECT} with every inference entry logged. "
            "Stricter than --delete-if-in-wandb, which only checks that runs "
            "with the expected names exist."
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

    wandb_finished_summaries: dict[str, list[set[str]]] | None = None
    if args.skip_if_in_wandb:
        print(f"Fetching finished runs from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_finished_summaries = fetch_wandb_finished_summaries()
        print(f"Found {len(wandb_finished_summaries)} finished run names.")

    source_configs = discover_source_configs(
        args.version,
        architectures=args.arch or ARCHITECTURES,
        source_dirs=(BASE_CONFIGS_DIR, RUN_CONFIGS_DIR),
    )

    for source_path in source_configs:
        generate_eval_config(
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
