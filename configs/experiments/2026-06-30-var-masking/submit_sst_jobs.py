"""Submit SST-perturbation inference jobs for the VarMaskingC96 checkpoints.

For each training run in wandb_to_beaker_map.json, submits one free-running
inference job per SST perturbation level (p0k / p2k / p4k). Each job mounts that
run's best_inference_ckpt.tar at /ckpt.tar and runs the matching config from
run_configs/ (produced by generate_sst_configs.py) via run-ace-inference.sh.

Usage:
    python submit_sst_jobs.py [--dry-run] [--run RUN [RUN ...]]
                              [--perturbation {p0k,p2k,p4k} ...]
                              [--beaker-workspace WORKSPACE]
                              [--beaker-cluster CLUSTER [CLUSTER ...]]
                              [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess
import sys

from generate_eval_configs import TRAINING_RESULT_DATASETS
from generate_masking_configs import WANDB_PROJECT
from generate_sst_configs import RUN_CONFIGS_DIR, SST_PERTURBATIONS, sst_config_filename

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-inference.sh"
WANDB_GROUP = "ace2-var-masking-sst-perts-2026-07-08"
# best_inference_ckpt.tar is always written by training; mounted at /ckpt.tar.
CHECKPOINT_PATH = "training_checkpoints/best_inference_ckpt.tar"


def validate_configs(levels: list[str]) -> None:
    for level in levels:
        config_path = RUN_CONFIGS_DIR / sst_config_filename(level)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fme.ace.validate_config",
                "--config_type",
                "inference",
                str(config_path),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        metavar="RUN",
        help="Restrict to these training run names (default: all runs in the map).",
    )
    parser.add_argument(
        "--perturbation",
        nargs="+",
        default=None,
        choices=list(SST_PERTURBATIONS),
        help="Restrict to these SST perturbation levels (default: all).",
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

    levels = args.perturbation or list(SST_PERTURBATIONS)
    run_names = args.run or sorted(TRAINING_RESULT_DATASETS)

    unknown_runs = sorted(set(run_names) - set(TRAINING_RESULT_DATASETS))
    if unknown_runs:
        raise KeyError(
            f"unknown training run(s) {unknown_runs} — "
            f"available: {sorted(TRAINING_RESULT_DATASETS)}"
        )

    for level in levels:
        config_path = RUN_CONFIGS_DIR / sst_config_filename(level)
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path.name} not found — run generate_sst_configs.py first"
            )

    if not args.dry_run:
        validate_configs(levels)

    for run_name in run_names:
        source_dataset_id = TRAINING_RESULT_DATASETS[run_name]
        for level in levels:
            config_filename = sst_config_filename(level)
            job_name = f"{run_name}-{level}"
            cmd = [
                str(RUN_SCRIPT),
                config_filename,
                job_name,
                WANDB_GROUP,
                source_dataset_id,
                CHECKPOINT_PATH,
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
