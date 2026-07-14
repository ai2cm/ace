"""Submit orography-swap evaluator-suite jobs for the FM checkpoints.

Each generated orography-swap eval suite (generate_orography_configs.py
output) has no training run or result dataset of its own -- it sources its
checkpoint from the corresponding native-grid run's own result dataset, and
only runs that run's best-inference checkpoint:

  - training_checkpoints/best_inference_ckpt.tar -> -bestinf
"""

import argparse
import pathlib

from _submit_common import add_beaker_args, submit_job
from _version_select import add_version_arg, stem_matches_version
from generate_eval_configs import (
    EVAL_CHECKPOINT_NAME_SUFFIXES,
    TRAINING_RESULT_DATASETS,
    WANDB_PREFIX,
    WANDB_PROJECT,
    eval_suite_config_to_run_name,
)
from generate_orography_configs import (
    OROGRAPHY_CHECKPOINT_SUFFIXES,
    OROGRAPHY_EVAL_SUITE_CONFIG_PREFIX,
    OROGRAPHY_SOURCES,
)
from run_eval_suite import run_eval_suite

from fme.core.distributed import Distributed

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-eval.sh"
WANDB_GROUP = "ace2-fm-eval-2026-06-26"

# Checkpoint file paths paired with the eval run-name suffixes (source of truth
# in generate_eval_configs.py), restricted to the suffixes orography-swap
# evals actually run.
CHECKPOINT_PATHS = [
    "training_checkpoints/best_ckpt.tar",
    "training_checkpoints/best_inference_ckpt.tar",
    "training_checkpoints/ckpt.tar",
]
CHECKPOINTS = [
    (path, suffix)
    for path, suffix in zip(CHECKPOINT_PATHS, EVAL_CHECKPOINT_NAME_SUFFIXES)
    if suffix in OROGRAPHY_CHECKPOINT_SUFFIXES
]


def _orography_base_run_name(run_name: str) -> str:
    """Return the plain (native-grid) run name this orography-swap eval run
    should source its checkpoint/dataset from.
    """
    for grid in OROGRAPHY_SOURCES:
        prefix = f"{WANDB_PREFIX}orog-{grid}-"
        if run_name.startswith(prefix):
            return f"{WANDB_PREFIX}{run_name.removeprefix(prefix)}"
    raise ValueError(f"{run_name!r} is not an orography-swap eval run name")


def configs_for_version(version: str | None) -> list[str]:
    configs = []
    for path in sorted(RUN_CONFIGS_DIR.glob("*.yaml")):
        if not path.name.startswith(OROGRAPHY_EVAL_SUITE_CONFIG_PREFIX):
            continue
        if not stem_matches_version(path.stem, version):
            continue
        run_name = eval_suite_config_to_run_name(path.name)
        base_run_name = _orography_base_run_name(run_name)
        if base_run_name not in TRAINING_RESULT_DATASETS:
            # No training result dataset recorded for the native-grid run;
            # skip rather than fail in config_to_jobs.
            print(f"Skipped {path.name} (no dataset ID for {base_run_name!r})")
            continue
        configs.append(path.name)
    return configs


def validate_configs(config_filenames: list[str]) -> None:
    with Distributed.context():
        for config_filename in config_filenames:
            run_eval_suite(str(RUN_CONFIGS_DIR / config_filename), validate_only=True)


def config_to_jobs(config_filename: str) -> list[tuple[str, str, str]]:
    run_name = eval_suite_config_to_run_name(config_filename)
    source_dataset_id = TRAINING_RESULT_DATASETS[_orography_base_run_name(run_name)]
    return [
        (f"{run_name}{name_suffix}", source_dataset_id, checkpoint_path)
        for checkpoint_path, name_suffix in CHECKPOINTS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
    add_beaker_args(
        parser,
        default_workspace="ai2/climate-titan",
        default_cluster=["ai2/titan"],
        default_priority="urgent",
    )
    args = parser.parse_args()

    configs = configs_for_version(args.version)
    if not args.dry_run:
        validate_configs(configs)

    for config_filename in configs:
        config_path = RUN_CONFIGS_DIR / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found"
                " - run generate_orography_configs.py first"
            )
        for job_name, source_dataset_id, checkpoint_path in config_to_jobs(
            config_filename
        ):
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
