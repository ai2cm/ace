"""Submit a gantry training job for each generated arm x seed config.

Each config produced by generate_masking_configs.py is submitted via
run-ace-train.sh, which validates the config and calls gantry.

Usage:
    python submit_mask_jobs.py [--dry-run] [--beaker-workspace WORKSPACE]
                               [--beaker-cluster CLUSTER [CLUSTER ...]]
                               [--beaker-priority PRIORITY]
                               [--cm-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess

from generate_masking_configs import (
    CONFIG_PREFIX,
    RUN_CONFIGS_DIR,
    WANDB_PROJECT,
    config_to_run_name,
)

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_GROUP = "ace2-varmask-batch-comparison-2026-08-26"

CONFIGS = sorted(
    path.name for path in RUN_CONFIGS_DIR.glob(f"{CONFIG_PREFIX}*-seed*.yaml")
)


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
        help=("Beaker cluster(s) to target (default: ai2/titan)."),
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

    for config_filename in CONFIGS:
        config_path = RUN_CONFIGS_DIR / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found — run generate_masking_configs.py first"
            )
        job_name = config_to_run_name(config_filename)
        cmd = [str(RUN_SCRIPT), config_filename, job_name, WANDB_GROUP]
        print("Submitting:", " ".join(cmd))
        if not args.dry_run:
            env = {
                **os.environ,
                "WANDB_PROJECT": WANDB_PROJECT,
                "BEAKER_WORKSPACE": args.beaker_workspace,
                "BEAKER_CLUSTER": " ".join(args.beaker_cluster),
                "BEAKER_PRIORITY": args.beaker_priority,
                "CM_PRIORITY": args.cm_priority,
            }
            subprocess.run(cmd, check=True, cwd=HERE, env=env)


if __name__ == "__main__":
    main()
