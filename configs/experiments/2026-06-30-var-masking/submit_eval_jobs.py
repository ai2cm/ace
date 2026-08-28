"""Submit evaluator-suite jobs for the var-masking checkpoints.

Each generated evaluator suite is run against one or more checkpoints from the
corresponding training result dataset, selected with ``--checkpoint`` (default:
all of them). The checkpoints, their paths within the result dataset and the
evaluator run name suffix each one gets are defined in eval_checkpoints.py.

The multi-step fine-tunes want ``--checkpoint lastepochema`` alone: they run a
fixed ``max_epochs: 20`` continuation of an already-converged checkpoint, so the
fine-tuned model *is* the final epoch, and the EMA-averaged weights are what the
training run's own inline inference charts report. ``lastepoch`` is the same
epoch of the same run evaluated without EMA, which is a different and much worse
model; the two selection-based checkpoints only re-evaluate a partially
fine-tuned one.

``--skip-evaluated`` drops the individual checkpoint jobs whose evaluator run
already exists in wandb, so a config whose evaluation was only partly
completed (e.g. one checkpoint preempted) resubmits just the missing
checkpoints rather than every one.

Naming configs positionally submits exactly those and ignores ``--version`` --
use it to evaluate one run (or the multi-step fine-tune family, whose eval
suites a bare ``-v v5`` would sweep in alongside the pre-training runs)
without submitting a whole version's worth of jobs.

Usage:
    python submit_eval_jobs.py [CONFIG ...]
                               [--dry-run]
                               [--version {v1,...}]
                               [--checkpoint NAME]
                               [--skip-evaluated]
                               [--beaker-workspace WORKSPACE]
                               [--beaker-cluster CLUSTER [CLUSTER ...]]
                               [--beaker-priority PRIORITY]
                               [--cm-priority PRIORITY]
"""

import argparse
import pathlib
import subprocess
from collections.abc import Sequence

import beaker_submit
from eval_checkpoints import EvalCheckpoint, by_names, names
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

# Evaluations are titan-only, like the fine-tunes they score. A queued urgent
# job is charged pessimistically against every cluster it could land on (see
# beaker_submit and scripts/beaker_balancer/README.md), so making these
# single-GPU jobs eligible for jupiter too holds headroom there that the
# multi-GPU training jobs need.
EVAL_CLUSTERS = ["ai2/titan"]

# Wandb states meaning "this evaluator job is done or on its way", i.e. no
# resubmission needed. Anything else (crashed, failed, killed, ...) is a dead
# run worth retrying. Listing the live states rather than the dead ones keeps
# an unrecognised state from silently suppressing a submission.
IN_FLIGHT_STATES = frozenset({"finished", "running", "pending", "preempting"})


def available_configs() -> list[str]:
    """Every evaluator suite config in run_configs/, whatever the version."""
    return sorted(
        path.name for path in RUN_CONFIGS_DIR.glob(f"{EVAL_SUITE_CONFIG_PREFIX}*.yaml")
    )


def select_configs(named: list[str], version: str | None) -> list[str]:
    """Eval suite configs to submit, either named explicitly or by version.

    Explicitly named configs bypass the version filter -- naming a config is
    already a deliberate choice, so it should not also have to match
    ``--version``.
    """
    present = available_configs()
    if named:
        wanted = [pathlib.Path(name).name for name in named]
        missing = [name for name in wanted if name not in present]
        if missing:
            raise FileNotFoundError(
                f"not in {RUN_CONFIGS_DIR}: {', '.join(missing)}\n"
                "available:\n" + "\n".join(f"  {name}" for name in present)
            )
        return sorted(set(wanted))
    return [
        name
        for name in present
        if version is None or stem_has_version(pathlib.Path(name).stem, version)
    ]


def validate_configs(config_filenames: list[str]) -> None:
    with Distributed.context():
        for config_filename in config_filenames:
            run_eval_suite(str(RUN_CONFIGS_DIR / config_filename), validate_only=True)


def config_to_jobs(
    config_filename: str,
    selected_checkpoints: Sequence[EvalCheckpoint],
    evaluated_states: dict[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """Jobs to submit for one eval suite config, one per selected checkpoint.

    ``evaluated_states`` maps wandb run name -> state; checkpoints whose run is
    in an in-flight state are dropped. Filtering happens before the beaker map
    lookup so that a config with no work left never raises for a missing map
    entry.
    """
    run_name = eval_suite_config_to_run_name(config_filename)
    checkpoints = []
    for checkpoint in selected_checkpoints:
        job_name = f"{run_name}{checkpoint.suffix}"
        state = (evaluated_states or {}).get(job_name)
        if state in IN_FLIGHT_STATES:
            print(f"Skipping {job_name} (state={state})")
            continue
        checkpoints.append((job_name, checkpoint))
    if not checkpoints:
        return []
    source_dataset_id = TRAINING_RESULT_DATASETS[run_name]
    jobs = []
    for job_name, checkpoint in checkpoints:
        # Resolved after the map lookup, since resolving needs the dataset ID.
        checkpoint_path = checkpoint.resolve(source_dataset_id)
        if checkpoint_path is None:
            # The run never reached an epoch this checkpoint is written at, so
            # there is nothing to evaluate. Skip the one job rather than failing
            # the submission: one short run should not block a sweep, and a
            # missing final-epoch EMA checkpoint is worth printing because it
            # means the run trained less far than the caller thinks.
            print(
                f"Skipping {job_name}: no {checkpoint.name} checkpoint in "
                f"{source_dataset_id}"
            )
            continue
        jobs.append((job_name, source_dataset_id, checkpoint_path))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="*",
        metavar="CONFIG",
        help=(
            "Evaluator suite config filename(s) in run_configs/ to submit. "
            "Bypasses --version; default is every config of the selected "
            "version(s)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    beaker_submit.add_arguments(parser, default_clusters=EVAL_CLUSTERS)
    parser.add_argument(
        "--version",
        "-v",
        choices=sorted(BASE_CONFIG_FILENAMES),
        default=None,
        help="Restrict to configs of this baseline version (default: all).",
    )
    parser.add_argument(
        # One value per flag, repeated, rather than nargs="+": a variadic option
        # swallows the positional CONFIG names that follow it, and naming
        # configs positionally is this script's main path.
        "--checkpoint",
        action="append",
        choices=list(names()),
        default=None,
        metavar="NAME",
        help=(
            "Checkpoint from each training result dataset to evaluate; repeat "
            f"to select several (default: all of {', '.join(names())}). "
            "Pass --checkpoint lastepochema for the multi-step fine-tunes, "
            "whose final epoch is the fine-tuned model and whose charts report "
            "EMA weights (see module docstring)."
        ),
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

    configs = select_configs(args.configs, args.version)
    if not configs:
        raise FileNotFoundError(
            f"no eval suite configs in {RUN_CONFIGS_DIR}"
            " — run generate_eval_configs.py first"
        )

    evaluated_states = (
        fetch_wandb_run_states(WANDB_PROJECT) if args.skip_evaluated else None
    )
    # Resolve jobs up front so configs with nothing left to submit are dropped
    # before the (comparatively slow) validation pass.
    selected_checkpoints = by_names(args.checkpoint or names())

    jobs_by_config = {
        config_filename: config_to_jobs(
            config_filename, selected_checkpoints, evaluated_states
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
                env = beaker_submit.env(
                    args, WANDB_PROJECT=WANDB_PROJECT, SKIP_VALIDATE="1"
                )
                subprocess.run(cmd, check=True, cwd=HERE, env=env)


if __name__ == "__main__":
    main()
