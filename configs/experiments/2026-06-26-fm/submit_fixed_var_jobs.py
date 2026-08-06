"""Submit fixed-variable evaluator-suite jobs for the FM checkpoints.

Each generated fixed-variable eval suite (generate_fixed_var_configs.py output)
has no training run or result dataset of its own -- it sources its checkpoint
from the corresponding unmodified run's own result dataset, and only runs that
run's best-inference checkpoint:

  - training_checkpoints/best_inference_ckpt.tar -> -bestinf

Pass --skip-if-in-wandb to skip jobs whose run name already exists in wandb,
so an interrupted sweep can be re-run without resubmitting what already went
out. Note this skips on the name existing at all, not on the run having
finished: to retry a suite that crashed partway, delete its wandb run first.
"""

import argparse
import pathlib

from _submit_common import add_beaker_args, submit_job
from _version_select import add_version_arg, stem_matches_version
from generate_eval_configs import (
    CONFIG_PREFIX,
    EVAL_CHECKPOINT_NAME_SUFFIXES,
    TRAINING_RESULT_DATASETS,
    WANDB_ENTITY,
    WANDB_PREFIX,
    WANDB_PROJECT,
    _fetch_wandb_run_names,
    eval_suite_config_to_run_name,
)
from generate_fixed_var_configs import (
    FIXED_VAR_CHECKPOINT_SUFFIXES,
    FIXED_VAR_EVAL_SUITE_CONFIG_PREFIX,
    select_source_configs,
)
from run_eval_suite import run_eval_suite

from fme.core.distributed import Distributed

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-eval.sh"
WANDB_GROUP = "ace2-fm-eval-2026-06-26"

# Prefix every fixed-variable eval run name carries, ahead of the held variable.
FIXED_VAR_RUN_NAME_PREFIX = f"{WANDB_PREFIX}fixed-"

# Checkpoint file paths paired with the eval run-name suffixes (source of truth
# in generate_eval_configs.py), restricted to the suffixes fixed-variable evals
# actually run.
CHECKPOINT_PATHS = [
    "training_checkpoints/best_ckpt.tar",
    "training_checkpoints/best_inference_ckpt.tar",
    "training_checkpoints/ckpt.tar",
]
CHECKPOINTS = [
    (path, suffix)
    for path, suffix in zip(CHECKPOINT_PATHS, EVAL_CHECKPOINT_NAME_SUFFIXES)
    if suffix in FIXED_VAR_CHECKPOINT_SUFFIXES
]


def _fixed_var_base_run_name(run_name: str) -> str:
    """Return the plain (unmodified) run name this fixed-variable eval run
    should source its checkpoint/dataset from.

    A fixed-variable run name is "{WANDB_PREFIX}fixed-{variable}-{suffix}" and
    the run it comes from is "{WANDB_PREFIX}{suffix}", so the base run is found
    by matching the longest recorded run name that the eval name ends with.
    Matching the longest keeps a suffix that is itself the tail of a longer one
    from claiming the name.
    """
    if not run_name.startswith(FIXED_VAR_RUN_NAME_PREFIX):
        raise ValueError(f"{run_name!r} is not a fixed-variable eval run name")
    matches = [
        base_run_name
        for base_run_name in TRAINING_RESULT_DATASETS
        if run_name.endswith(f"-{base_run_name.removeprefix(WANDB_PREFIX)}")
    ]
    if not matches:
        raise ValueError(
            f"No recorded training run name matches the tail of {run_name!r}"
        )
    return max(matches, key=len)


def configs_for_version(
    version: str | None, base_configs: list[str] | None = None
) -> list[str]:
    """Generated fixed-variable suite filenames to submit.

    `base_configs` narrows the selection to the suites generated from the named
    training configs, resolved the same way generate_fixed_var_configs.py
    resolves its own --base-config. A suite filename ends with the training
    config's stem, so matching that tail selects one run's whole variable sweep
    without also catching runs whose stem merely starts the same way.
    """
    suffixes: list[str] | None = None
    if base_configs is not None:
        suffixes = [
            path.stem.removeprefix(CONFIG_PREFIX)
            for path in select_source_configs(version, base_configs)
        ]
    configs = []
    for path in sorted(RUN_CONFIGS_DIR.glob("*.yaml")):
        if not path.name.startswith(FIXED_VAR_EVAL_SUITE_CONFIG_PREFIX):
            continue
        if not stem_matches_version(path.stem, version):
            continue
        if suffixes is not None and not any(
            path.stem.endswith(f"-{suffix}") for suffix in suffixes
        ):
            continue
        run_name = eval_suite_config_to_run_name(path.name)
        try:
            _fixed_var_base_run_name(run_name)
        except ValueError:
            # No training result dataset recorded for the unmodified run;
            # skip rather than fail in config_to_jobs.
            print(f"Skipped {path.name} (no dataset ID for the run it comes from)")
            continue
        configs.append(path.name)
    return configs


def validate_configs(config_filenames: list[str]) -> None:
    with Distributed.context():
        for config_filename in config_filenames:
            run_eval_suite(str(RUN_CONFIGS_DIR / config_filename), validate_only=True)


def config_to_jobs(config_filename: str) -> list[tuple[str, str, str]]:
    run_name = eval_suite_config_to_run_name(config_filename)
    source_dataset_id = TRAINING_RESULT_DATASETS[_fixed_var_base_run_name(run_name)]
    return [
        (f"{run_name}{name_suffix}", source_dataset_id, checkpoint_path)
        for checkpoint_path, name_suffix in CHECKPOINTS
    ]


def _jobs_to_submit(
    config_filenames: list[str], existing_runs: set[str]
) -> dict[str, list[tuple[str, str, str]]]:
    """Jobs per config, dropping those whose run name is already in wandb.

    Configs left with no jobs are dropped entirely, so a fully-submitted suite
    is not validated -- validation loads every inference entry and is the slow
    part of a submit run.
    """
    jobs = {}
    for config_filename in config_filenames:
        remaining = []
        for job in config_to_jobs(config_filename):
            if job[0] in existing_runs:
                print(f"Skipping (already in wandb): {job[0]}")
            else:
                remaining.append(job)
        if remaining:
            jobs[config_filename] = remaining
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
    parser.add_argument(
        "--base-config",
        nargs="+",
        default=None,
        help=(
            "Only submit suites generated from these training config(s), named "
            "by filename, stem, or run-name suffix (default: all suites)."
        ),
    )
    add_beaker_args(
        parser,
        default_workspace="ai2/climate-titan",
        default_cluster=["ai2/titan"],
        default_priority="urgent",
    )
    parser.add_argument(
        "--skip-if-in-wandb",
        action="store_true",
        help=(
            "Skip jobs whose run name already exists as a run in the "
            f"{WANDB_ENTITY}/{WANDB_PROJECT} wandb project."
        ),
    )
    args = parser.parse_args()

    existing_runs = _fetch_wandb_run_names() if args.skip_if_in_wandb else set()

    configs = configs_for_version(args.version, args.base_config)
    for config_filename in configs:
        config_path = RUN_CONFIGS_DIR / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found - run "
                "generate_fixed_var_configs.py first"
            )

    jobs = _jobs_to_submit(configs, existing_runs)
    if not args.dry_run:
        validate_configs(list(jobs))

    for config_filename, config_jobs in jobs.items():
        for job_name, source_dataset_id, checkpoint_path in config_jobs:
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


if __name__ == "__main__":
    main()
