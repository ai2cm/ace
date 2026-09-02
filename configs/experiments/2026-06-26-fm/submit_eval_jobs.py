"""Submit evaluator-suite jobs for the FM checkpoints.

Each generated evaluator suite (generate_eval_configs.py output) is run
against three checkpoints from the corresponding training result dataset:

  - training_checkpoints/best_ckpt.tar -> -besttrain
  - training_checkpoints/best_inference_ckpt.tar -> -bestinf
  - training_checkpoints/ckpt.tar -> -lastepoch

Orography-swap and fixed-variable eval suites (the output of
generate_orography_configs.py and generate_fixed_var_configs.py) are submitted
separately by submit_orography_jobs.py and submit_fixed_var_jobs.py.

Usage:
    python submit_eval_jobs.py [--version {v1,v2,v3}] [--arch ARCH [ARCH ...]]
                               [--skip-if-in-wandb] [--dry-run]
                               [--beaker-workspace WORKSPACE]
                               [--beaker-cluster CLUSTER [CLUSTER ...]]
                               [--beaker-priority PRIORITY]
                               [--cm-priority PRIORITY]
"""

import argparse
import pathlib
from collections.abc import Sequence

import yaml
from _submit_common import add_beaker_args, check_configs_at_head, submit_job
from _version_select import add_version_arg, stem_matches_version
from generate_eval_configs import (
    ARCHITECTURES,
    EVAL_CHECKPOINT_NAME_SUFFIXES,
    EVAL_SUITE_CONFIG_PREFIX,
    TRAINING_RESULT_DATASETS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    all_inferences_succeeded,
    eval_suite_config_to_run_name,
    fetch_wandb_finished_summaries,
)
from generate_fixed_var_configs import FIXED_VAR_EVAL_SUITE_CONFIG_PREFIX
from generate_orography_configs import OROGRAPHY_EVAL_SUITE_CONFIG_PREFIX
from run_eval_suite import run_eval_suite

from fme.core.distributed import Distributed

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-eval.sh"
WANDB_GROUP = "ace2-fm-eval-2026-06-26"

# Checkpoint file paths paired with the eval run-name suffixes (source of truth
# in generate_eval_configs.py, kept in the same order as the checkpoints here).
CHECKPOINT_PATHS = [
    "training_checkpoints/best_ckpt.tar",
    "training_checkpoints/best_inference_ckpt.tar",
    "training_checkpoints/ckpt.tar",
]
CHECKPOINTS = list(zip(CHECKPOINT_PATHS, EVAL_CHECKPOINT_NAME_SUFFIXES))


def configs_for_version(
    version: str | None, architectures: Sequence[str]
) -> tuple[list[str], int]:
    """Suites to submit, and how many matched the filters before the map lookup.

    The count lets main() tell "no suite matches --version/--arch" from "the
    matching suites have no training result dataset recorded yet", which are
    fixed by different commands.
    """
    matched = []
    for path in sorted(RUN_CONFIGS_DIR.glob("*.yaml")):
        if not path.name.startswith(EVAL_SUITE_CONFIG_PREFIX):
            continue
        if path.name.startswith(OROGRAPHY_EVAL_SUITE_CONFIG_PREFIX):
            continue
        if path.name.startswith(FIXED_VAR_EVAL_SUITE_CONFIG_PREFIX):
            continue
        if not any(arch in path.name for arch in architectures):
            continue
        if not stem_matches_version(path.stem, version):
            continue
        matched.append(path.name)

    configs = []
    for config_filename in matched:
        run_name = eval_suite_config_to_run_name(config_filename)
        if run_name not in TRAINING_RESULT_DATASETS:
            # No training result dataset recorded for this run; skip rather
            # than fail in config_to_jobs. Matches generate_eval_configs.py.
            print(f"Skipped {config_filename} (no dataset ID for {run_name!r})")
            continue
        configs.append(config_filename)
    return configs, len(matched)


def validate_configs(config_filenames: list[str]) -> None:
    with Distributed.context():
        for config_filename in config_filenames:
            run_eval_suite(str(RUN_CONFIGS_DIR / config_filename), validate_only=True)


def config_to_jobs(config_filename: str) -> list[tuple[str, str, str]]:
    run_name = eval_suite_config_to_run_name(config_filename)
    source_dataset_id = TRAINING_RESULT_DATASETS[run_name]
    return [
        (f"{run_name}{name_suffix}", source_dataset_id, checkpoint_path)
        for checkpoint_path, name_suffix in CHECKPOINTS
    ]


def _pending_jobs(
    config_filenames: list[str],
    wandb_finished_summaries: dict[str, list[set[str]]] | None,
) -> tuple[list[tuple[str, list[tuple[str, str, str]]]], int]:
    """The jobs still to run, per suite, and how many were already done.

    A checkpoint's job is dropped when a single finished wandb run of its name
    logged every inference entry in the suite, so a suite whose three
    checkpoints are all done contributes nothing.
    """
    pending = []
    n_skipped = 0
    for config_filename in config_filenames:
        config_path = RUN_CONFIGS_DIR / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found - run generate_eval_configs.py first"
            )
        jobs = config_to_jobs(config_filename)
        if wandb_finished_summaries is not None:
            with config_path.open() as f:
                cfg = yaml.safe_load(f)
            remaining = []
            for job in jobs:
                job_name = job[0]
                if all_inferences_succeeded(cfg, [job_name], wandb_finished_summaries):
                    print(f"Skipping (already finished in wandb): {job_name}")
                    n_skipped += 1
                else:
                    remaining.append(job)
            jobs = remaining
        if jobs:
            pending.append((config_filename, jobs))
    return pending, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
    parser.add_argument(
        "--arch",
        nargs="+",
        choices=ARCHITECTURES,
        default=None,
        help="Only submit suites for these architectures (default: all).",
    )
    parser.add_argument(
        "--skip-if-in-wandb",
        action="store_true",
        help=(
            "Skip each checkpoint whose eval run already finished in "
            f"{WANDB_ENTITY}/{WANDB_PROJECT} with every inference entry in the "
            "suite logged; a suite with all three done submits nothing."
        ),
    )
    add_beaker_args(
        parser,
        default_workspace="ai2/ace",
        default_cluster=["ai2/titan", "ai2/jupiter"],
        default_priority="high",
    )
    args = parser.parse_args()

    configs, n_matched = configs_for_version(args.version, args.arch or ARCHITECTURES)
    if n_matched == 0:
        raise SystemExit(
            "No eval suite config matches these filters - run "
            "generate_eval_configs.py with the same --version/--arch first."
        )
    if not configs:
        raise SystemExit(
            f"None of the {n_matched} matching eval suite configs has a training "
            "result dataset recorded - run update_beaker_map.py (a run only "
            "enters the map once its Beaker job has exited 0)."
        )

    wandb_finished_summaries = None
    if args.skip_if_in_wandb:
        print(f"Fetching finished runs from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_finished_summaries = fetch_wandb_finished_summaries()
        print(f"Found {len(wandb_finished_summaries)} finished run names.")

    pending, n_skipped = _pending_jobs(configs, wandb_finished_summaries)
    if not pending:
        print("Nothing to submit.")
        return

    check_configs_at_head(
        [RUN_CONFIGS_DIR / config_filename for config_filename, _ in pending]
    )

    if not args.dry_run:
        validate_configs([config_filename for config_filename, _ in pending])

    n_submitted = 0
    for config_filename, jobs in pending:
        for job_name, source_dataset_id, checkpoint_path in jobs:
            # submit_job raises on the first gantry failure, so this total is
            # only ever printed once every job went out.
            submit_job(
                RUN_SCRIPT,
                [
                    f"{RUN_CONFIGS_DIRNAME}/{config_filename}",
                    job_name,
                    WANDB_GROUP,
                    source_dataset_id,
                    checkpoint_path,
                ],
                wandb_project=WANDB_PROJECT,
                args=args,
                cwd=HERE,
                extra_env={"SKIP_VALIDATE": "1"},
            )
            n_submitted += 1

    noun = "job" if n_submitted == 1 else "jobs"
    verb = "would be submitted" if args.dry_run else "submitted"
    skipped = f" ({n_skipped} skipped, already finished in wandb)" if n_skipped else ""
    print(f"{n_submitted} {noun} {verb}{skipped}")


if __name__ == "__main__":
    main()
