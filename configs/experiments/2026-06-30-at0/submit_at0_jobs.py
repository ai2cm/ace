"""Submit a gantry training job for each generated at0 config.

Each config produced by generate_at0_configs.py is submitted via
run-ace-train.sh, which validates the config and calls gantry.

Usage:
    python submit_at0_jobs.py [--dry-run] [--beaker-workspace WORKSPACE]
                               [--beaker-cluster CLUSTER [CLUSTER ...]]
                               [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess

from generate_at0_configs import (
    CONFIG_PREFIX,
    WANDB_PREFIX,
    WANDB_PROJECT,
    WANDB_SUFFIX,
)

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_GROUP = "ace2-at0-2026-06-30"

CONFIGS = sorted(
    path.name
    for path in HERE.glob("*.yaml")
    if path.name.startswith(CONFIG_PREFIX) and "-1940" in path.name
)


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-AIMIP-nc-sfno-econ-gmron-1940off.yaml
    # → ace2-at0-nc-sfno-econ-gmron-1940off
    stem = pathlib.Path(config_filename).stem  # strip .yaml
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


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
        help=("Beaker cluster(s) to target (ex: ai2/titan" "ai2/jupiter ai2/ceres)."),
    )
    parser.add_argument(
        "--beaker-priority",
        default="urgent",
        help="Beaker job priority (ex: high or urgent).",
    )
    args = parser.parse_args()

    for config_filename in CONFIGS:
        config_path = HERE / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found — run generate_at0_configs.py first"
            )
        job_name = config_to_job_name(config_filename)
        cmd = [str(RUN_SCRIPT), config_filename, job_name, WANDB_GROUP]
        print("Submitting:", " ".join(cmd))
        if not args.dry_run:
            env = {
                **os.environ,
                "WANDB_PROJECT": WANDB_PROJECT,
                "BEAKER_WORKSPACE": args.beaker_workspace,
                "BEAKER_CLUSTER": " ".join(args.beaker_cluster),
                "BEAKER_PRIORITY": args.beaker_priority,
            }
            subprocess.run(cmd, check=True, cwd=HERE, env=env)


if __name__ == "__main__":
    main()
