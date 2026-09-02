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

``--checkpoint`` restricts submission to a subset of the three variants (e.g.
``--checkpoint bestinf``), for sweeps where only the best-inference checkpoint
is of interest. Pass the same flag to generate_eval_configs.py's
``--delete-if-in-wandb`` so config cleanup does not wait on variants that were
never submitted.

``--match`` restricts submission to configs whose filename contains a given
substring. ``run_configs/`` accumulates eval configs for arms that were already
evaluated (only ``--delete-if-in-wandb`` prunes them, and configs are kept
while their eval jobs are in flight), and nothing downstream notices a
duplicate submission, so a sweep of one arm needs a narrower filter than
``--version``. Repeat the flag to submit several arms.
"""

import argparse
import os
import pathlib
import subprocess

from generate_eval_configs import (
    EVAL_SUITE_CONFIG_PREFIX,
    TRAINING_RESULT_DATASETS,
    add_checkpoint_argument,
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
from submit_preflight import check_submit_preconditions

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


def checkpoints_for_names(
    checkpoint_names: list[str],
) -> list[tuple[str, str]]:
    """``CHECKPOINTS`` restricted to the given ``--checkpoint`` labels."""
    wanted = {f"-{name}" for name in checkpoint_names}
    return [
        (path, name_suffix)
        for path, name_suffix in CHECKPOINTS
        if name_suffix in wanted
    ]


def config_to_jobs(
    config_filename: str,
    evaluated_states: dict[str, str] | None = None,
    checkpoints_to_run: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    """Jobs to submit for one eval suite config, one per checkpoint.

    ``evaluated_states`` maps wandb run name -> state; checkpoints whose run is
    in an in-flight state are dropped. Filtering happens before the beaker map
    lookup so that a config with no work left never raises for a missing map
    entry. ``checkpoints_to_run`` defaults to every entry in ``CHECKPOINTS``.
    """
    run_name = eval_suite_config_to_run_name(config_filename)
    checkpoints = []
    for checkpoint_path, name_suffix in (
        CHECKPOINTS if checkpoints_to_run is None else checkpoints_to_run
    ):
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
        default="ai2/ace",
        help="Beaker workspace to submit jobs to (default: ai2/ace).",
    )
    parser.add_argument(
        "--beaker-cluster",
        nargs="+",
        default=["ai2/titan", "ai2/jupiter"],
        metavar="CLUSTER",
        help="Beaker cluster(s) to target (default: ai2/titan ai2/jupiter).",
    )
    parser.add_argument(
        "--beaker-priority",
        default="high",
        help="Beaker job priority (default: high).",
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
    parser.add_argument(
        "--match",
        nargs="+",
        default=None,
        metavar="SUBSTRING",
        help=(
            "Restrict to configs whose filename contains any of these "
            "substrings (ex: --match maskbatch masksample). Default: all."
        ),
    )
    add_checkpoint_argument(parser)
    args = parser.parse_args()
    checkpoints_to_run = checkpoints_for_names(args.checkpoint)

    configs = sorted(
        path.name
        for path in RUN_CONFIGS_DIR.glob("*.yaml")
        if path.name.startswith(EVAL_SUITE_CONFIG_PREFIX)
        and (args.version is None or stem_has_version(path.stem, args.version))
        and (args.match is None or any(m in path.name for m in args.match))
    )
    if not configs:
        raise FileNotFoundError(
            f"no eval suite configs in {RUN_CONFIGS_DIR} matching the given "
            "filters — run generate_eval_configs.py first, or widen "
            "--version/--match"
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
        config_filename: config_to_jobs(
            config_filename, evaluated_states, checkpoints_to_run
        )
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

    # The evaluator entrypoint is run by repository path, so it is declared
    # alongside the configs; both must be in the commit gantry sends to Beaker.
    check_submit_preconditions(
        [RUN_CONFIGS_DIR / config_filename for config_filename in jobs_by_config]
        + [HERE / "run_eval_suite.py"],
        args.dry_run,
    )

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
