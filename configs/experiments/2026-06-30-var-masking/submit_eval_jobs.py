"""Submit evaluator-suite jobs for the var-masking checkpoints.

Each generated evaluator suite is run against three checkpoints from the
corresponding training result dataset:

  - training_checkpoints/best_ckpt.tar -> -besttrain
  - training_checkpoints/best_inference_ckpt.tar -> -bestinf
  - training_checkpoints/ckpt.tar -> -lastepoch

``--skip-evaluated`` drops the individual checkpoint jobs whose evaluator run
already exists in wandb, so a config whose evaluation was only partly
completed (e.g. one checkpoint preempted) resubmits just the missing
checkpoints rather than all three.
"""

import argparse
import os
import pathlib
import subprocess

from generate_eval_configs import (
    EVAL_SUITE_CONFIG_PREFIX,
    TRAINING_RESULT_DATASETS,
    eval_suite_config_to_run_name,
    fetch_wandb_run_states,
)
from generate_masking_configs import (
    BASE_CONFIG_FILENAMES,
    RUN_CONFIGS_DIR,
    WANDB_PROJECT,
    stem_has_version,
)
from run_eval_suite import run_eval_suite

from fme.core.distributed import Distributed

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-eval.sh"
WANDB_GROUP = "ace2-var-masking-eval-2026-06-30"

CHECKPOINTS = [
    ("training_checkpoints/best_ckpt.tar", "-besttrain"),
    ("training_checkpoints/best_inference_ckpt.tar", "-bestinf"),
    ("training_checkpoints/ckpt.tar", "-lastepoch"),
]

# Wandb states meaning "this evaluator job is done or on its way", i.e. no
# resubmission needed. Anything else (crashed, failed, killed, ...) is a dead
# run worth retrying. Listing the live states rather than the dead ones keeps
# an unrecognised state from silently suppressing a submission.
IN_FLIGHT_STATES = frozenset({"finished", "running", "pending", "preempting"})


def validate_configs(config_filenames: list[str]) -> None:
    with Distributed.context():
        for config_filename in config_filenames:
            run_eval_suite(str(RUN_CONFIGS_DIR / config_filename), validate_only=True)


def config_to_jobs(
    config_filename: str,
    evaluated_states: dict[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """Jobs to submit for one eval suite config, one per checkpoint.

    ``evaluated_states`` maps wandb run name -> state; checkpoints whose run is
    in an in-flight state are dropped. Filtering happens before the beaker map
    lookup so that a config with no work left never raises for a missing map
    entry.
    """
    run_name = eval_suite_config_to_run_name(config_filename)
    checkpoints = []
    for checkpoint_path, name_suffix in CHECKPOINTS:
        job_name = f"{run_name}{name_suffix}"
        state = (evaluated_states or {}).get(job_name)
        if state in IN_FLIGHT_STATES:
            print(f"Skipping {job_name} (state={state})")
            continue
        checkpoints.append((job_name, checkpoint_path))
    if not checkpoints:
        return []
    source_dataset_id = TRAINING_RESULT_DATASETS[run_name]
    return [
        (job_name, source_dataset_id, checkpoint_path)
        for job_name, checkpoint_path in checkpoints
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--beaker-workspace",
        default="ai2/climate-titan",
        help="Beaker workspace to submit jobs to (default: ai2/climate-titan).",
    )
    parser.add_argument(
        "--beaker-cluster",
        nargs="+",
        default=["ai2/titan"],
        metavar="CLUSTER",
        help="Beaker cluster(s) to target (default: ai2/titan).",
    )
    parser.add_argument(
        "--beaker-priority",
        default="urgent",
        help="Beaker job priority (default: urgent).",
    )
    parser.add_argument(
        "--version",
        "-v",
        choices=sorted(BASE_CONFIG_FILENAMES),
        default=None,
        help="Restrict to configs of this baseline version (default: all).",
    )
    parser.add_argument(
        "--skip-evaluated",
        action="store_true",
        help=(
            "Skip checkpoint jobs whose evaluator run already exists in wandb "
            "and is finished, running or queued. Runs are matched by name "
            "only, so an existing run is skipped even if it evaluated a "
            "different set of inference entries; omit this flag to resubmit "
            "regardless."
        ),
    )
    args = parser.parse_args()

    configs = sorted(
        path.name
        for path in RUN_CONFIGS_DIR.glob("*.yaml")
        if path.name.startswith(EVAL_SUITE_CONFIG_PREFIX)
        and (args.version is None or stem_has_version(path.stem, args.version))
    )
    if not configs:
        raise FileNotFoundError(
            f"no eval suite configs in {RUN_CONFIGS_DIR}"
            " — run generate_eval_configs.py first"
        )

    for config_filename in configs:
        config_path = RUN_CONFIGS_DIR / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found - run generate_eval_configs.py first"
            )

    evaluated_states = (
        fetch_wandb_run_states(WANDB_PROJECT) if args.skip_evaluated else None
    )
    # Resolve jobs up front so configs with nothing left to submit are dropped
    # before the (comparatively slow) validation pass.
    jobs_by_config = {
        config_filename: config_to_jobs(config_filename, evaluated_states)
        for config_filename in configs
    }
    jobs_by_config = {
        config_filename: jobs
        for config_filename, jobs in jobs_by_config.items()
        if jobs
    }
    if not jobs_by_config:
        print("Nothing to submit: all eval jobs already exist in wandb.")
        return

    if not args.dry_run:
        validate_configs(list(jobs_by_config))

    for config_filename, jobs in jobs_by_config.items():
        for job_name, source_dataset_id, checkpoint_path in jobs:
            cmd = [
                str(RUN_SCRIPT),
                config_filename,
                job_name,
                WANDB_GROUP,
                source_dataset_id,
                checkpoint_path,
            ]
            print("Submitting:", " ".join(cmd))
            if not args.dry_run:
                env = {
                    **os.environ,
                    "WANDB_PROJECT": WANDB_PROJECT,
                    "BEAKER_WORKSPACE": args.beaker_workspace,
                    "BEAKER_CLUSTER": " ".join(args.beaker_cluster),
                    "BEAKER_PRIORITY": args.beaker_priority,
                    "SKIP_VALIDATE": "1",
                }
                subprocess.run(cmd, check=True, cwd=HERE, env=env)


if __name__ == "__main__":
    main()
