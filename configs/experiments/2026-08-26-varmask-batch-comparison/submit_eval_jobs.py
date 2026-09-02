"""Submit evaluator-suite jobs for the arm x seed checkpoints.

Each generated evaluator suite is run against the best-inference checkpoint of
the corresponding training result dataset:

  - training_checkpoints/best_inference_ckpt.tar -> -bestinf

A config whose training run has no Beaker result dataset in
wandb_to_beaker_map.json is skipped, so the sweep can be evaluated in waves as
its training jobs finish: run update_beaker_map.py, then this script, and the
runs that finished since get their evaluators submitted.
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
from generate_masking_configs import RUN_CONFIGS_DIR, WANDB_PROJECT
from run_eval_suite import run_eval_suite

from fme.core.distributed import Distributed

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-eval.sh"
WANDB_GROUP = "ace2-varmask-batch-comparison-eval-2026-08-26"

CHECKPOINTS = [
    ("training_checkpoints/best_inference_ckpt.tar", "-bestinf"),
]

CONFIGS = sorted(
    path.name for path in RUN_CONFIGS_DIR.glob(f"{EVAL_SUITE_CONFIG_PREFIX}*.yaml")
)


def validate_configs(config_filenames: list[str]) -> None:
    with Distributed.context():
        for config_filename in config_filenames:
            run_eval_suite(str(RUN_CONFIGS_DIR / config_filename), validate_only=True)


def submittable_configs(config_filenames: list[str]) -> list[str]:
    """Configs whose training run already has a Beaker result dataset."""
    submittable = []
    for config_filename in config_filenames:
        run_name = eval_suite_config_to_run_name(config_filename)
        if run_name not in TRAINING_RESULT_DATASETS:
            print(
                f"Skipping {config_filename}: no Beaker result dataset for "
                f"{run_name!r} in wandb_to_beaker_map.json - run "
                "update_beaker_map.py once the training job has finished."
            )
            continue
        submittable.append(config_filename)
    return submittable


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
        default="ai2/ace",
        help=(
            "Beaker workspace to submit jobs to (default: ai2/ace, the "
            "workspace scripts/beaker_balancer manages, so CM_PRIORITY is "
            "honoured)."
        ),
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
        default="urgent",
        help="Beaker job priority (default: urgent).",
    )
    parser.add_argument(
        "--cm-priority",
        default="high",
        choices=["low", "normal", "high", "urgent"],
        help=(
            "CM_PRIORITY env var set on the job, opting it in to "
            "scripts/beaker_balancer (default: high)."
        ),
    )
    args = parser.parse_args()

    configs = submittable_configs(CONFIGS)
    if not configs:
        print("No config has a Beaker result dataset yet; nothing to submit.")
        return

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
                    "CM_PRIORITY": args.cm_priority,
                    "SKIP_VALIDATE": "1",
                }
                subprocess.run(cmd, check=True, cwd=HERE, env=env)


if __name__ == "__main__":
    main()
