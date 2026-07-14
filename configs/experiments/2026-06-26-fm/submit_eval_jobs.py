"""Submit evaluator-suite jobs for the FM checkpoints.

Each generated evaluator suite is run against three checkpoints from the
corresponding training result dataset:

  - training_checkpoints/best_ckpt.tar -> -besttrain
  - training_checkpoints/best_inference_ckpt.tar -> -bestinf
  - training_checkpoints/ckpt.tar -> -lastepoch

Orography-swap eval suites (generate_orography_configs.py output) are the
exception: they only run the -bestinf checkpoint, sourced from the
corresponding native-grid run's own result dataset.
"""

import argparse
import os
import pathlib
import subprocess

from _version_select import add_version_arg, stem_matches_version
from generate_eval_configs import (
    EVAL_CHECKPOINT_NAME_SUFFIXES,
    EVAL_SUITE_CONFIG_PREFIX,
    TRAINING_RESULT_DATASETS,
    WANDB_PREFIX,
    WANDB_PROJECT,
    eval_suite_config_to_run_name,
)
from generate_orography_configs import OROGRAPHY_CHECKPOINT_SUFFIXES, OROGRAPHY_SOURCES
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


def _orography_base_run_name(run_name: str) -> str | None:
    """If `run_name` is an orography-swap eval run, return the plain (native
    grid) run name it should source its checkpoint/dataset from. Orography
    evals have no training run or result dataset of their own -- they reuse
    the corresponding native run's best-inference checkpoint.
    """
    for grid in OROGRAPHY_SOURCES:
        prefix = f"{WANDB_PREFIX}orog-{grid}-"
        if run_name.startswith(prefix):
            return f"{WANDB_PREFIX}{run_name.removeprefix(prefix)}"
    return None


def configs_for_version(version: str | None) -> list[str]:
    configs = []
    for path in sorted(RUN_CONFIGS_DIR.glob("*.yaml")):
        if not path.name.startswith(EVAL_SUITE_CONFIG_PREFIX):
            continue
        if not stem_matches_version(path.stem, version):
            continue
        run_name = eval_suite_config_to_run_name(path.name)
        dataset_lookup_name = _orography_base_run_name(run_name) or run_name
        if dataset_lookup_name not in TRAINING_RESULT_DATASETS:
            # No training result dataset recorded for this run; skip rather
            # than fail in config_to_jobs. Matches generate_eval_configs.py.
            print(f"Skipped {path.name} (no dataset ID for {dataset_lookup_name!r})")
            continue
        configs.append(path.name)
    return configs


def validate_configs(config_filenames: list[str]) -> None:
    with Distributed.context():
        for config_filename in config_filenames:
            run_eval_suite(str(RUN_CONFIGS_DIR / config_filename), validate_only=True)


def config_to_jobs(config_filename: str) -> list[tuple[str, str, str]]:
    run_name = eval_suite_config_to_run_name(config_filename)
    base_run_name = _orography_base_run_name(run_name)
    if base_run_name is not None:
        source_dataset_id = TRAINING_RESULT_DATASETS[base_run_name]
        checkpoints = [
            (path, suffix)
            for path, suffix in CHECKPOINTS
            if suffix in OROGRAPHY_CHECKPOINT_SUFFIXES
        ]
    else:
        source_dataset_id = TRAINING_RESULT_DATASETS[run_name]
        checkpoints = CHECKPOINTS
    return [
        (f"{run_name}{name_suffix}", source_dataset_id, checkpoint_path)
        for checkpoint_path, name_suffix in checkpoints
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
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
    args = parser.parse_args()

    configs = configs_for_version(args.version)
    if not args.dry_run:
        validate_configs(configs)

    for config_filename in configs:
        config_path = RUN_CONFIGS_DIR / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found - run generate_eval_configs.py first"
            )
        for job_name, source_dataset_id, checkpoint_path in config_to_jobs(
            config_filename
        ):
            cmd = [
                str(RUN_SCRIPT),
                f"{RUN_CONFIGS_DIRNAME}/{config_filename}",
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
