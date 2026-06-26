"""Submit a gantry training job for each nc-sfno foundation model (fm) config.

Each nc-sfno config in this directory is submitted via run-ace-train.sh, which
validates the config and calls gantry.

Usage:
    python submit_fm_jobs.py [--dry-run] [--beaker-workspace WORKSPACE]
                             [--beaker-cluster CLUSTER [CLUSTER ...]]
                             [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_PROJECT = "FM"
WANDB_PREFIX = "ace2-fm-"
WANDB_SUFFIX = "-v1"
WANDB_GROUP = "ace2-fm-2026-06-26"
CONFIG_PREFIX = "ace-train-config-4deg-AIMIP-"

CONFIGS = sorted(
    path.name
    for path in HERE.glob("*.yaml")
    if path.name.startswith(CONFIG_PREFIX) and "nc-sfno" in path.name
)


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-AIMIP-nc-sfno-fm.yaml
    # → ace2-fm-nc-sfno-fm-v1
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
        help=("Beaker cluster(s) to target (default: ai2/titan)."),
    )
    parser.add_argument(
        "--beaker-priority",
        default="urgent",
        help="Beaker job priority (default: urgent).",
    )
    args = parser.parse_args()

    for config_filename in CONFIGS:
        config_path = HERE / config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"{config_filename} not found")
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
