"""Submit evaluator-suite jobs for the VarMasking4 checkpoints.

Each generated evaluator suite is run against three checkpoints from the
corresponding training result dataset:

  - training_checkpoints/best_ckpt.tar -> -besttrain
  - training_checkpoints/best_inference_ckpt.tar -> -bestinf
  - training_checkpoints/ckpt.tar -> -lastepoch
"""

import argparse
import os
import pathlib
import subprocess

from generate_eval_configs import (
    EVAL_SUITE_CONFIG_PREFIX,
    TRAINING_RESULT_DATASETS,
    eval_suite_config_to_run_name,
)
from generate_masking_configs import WANDB_PROJECT
from run_eval_suite import run_eval_suite

from fme.core.distributed import Distributed

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-eval.sh"
WANDB_GROUP = "ace2-var-masking-eval-2026-06-17"

CHECKPOINTS = [
    ("training_checkpoints/best_ckpt.tar", "-besttrain"),
    ("training_checkpoints/best_inference_ckpt.tar", "-bestinf"),
    ("training_checkpoints/ckpt.tar", "-lastepoch"),
]

CONFIGS = sorted(
    path.name
    for path in HERE.glob("*.yaml")
    if path.name.startswith(EVAL_SUITE_CONFIG_PREFIX)
)


def validate_configs(config_filenames: list[str]) -> None:
    with Distributed.context():
        for config_filename in config_filenames:
            run_eval_suite(str(HERE / config_filename), validate_only=True)


def config_to_jobs(config_filename: str) -> list[tuple[str, str, str]]:
    run_name = eval_suite_config_to_run_name(config_filename)
    source_dataset_id = TRAINING_RESULT_DATASETS[run_name]
    return [
        (f"{run_name}{name_suffix}", source_dataset_id, checkpoint_path)
        for checkpoint_path, name_suffix in CHECKPOINTS
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
    args = parser.parse_args()

    if not args.dry_run:
        validate_configs(CONFIGS)

    for config_filename in CONFIGS:
        config_path = HERE / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found - run generate_eval_configs.py first"
            )
        for job_name, source_dataset_id, checkpoint_path in config_to_jobs(
            config_filename
        ):
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
