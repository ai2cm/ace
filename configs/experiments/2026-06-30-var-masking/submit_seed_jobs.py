"""Submit a gantry training job for each generated seed-replicate config.

Each ``*-seed*.yaml`` config in run_configs/ (from generate_seed_configs.py) is
submitted via run-ace-train.sh, which validates it and calls gantry.

Usage:
    python submit_seed_jobs.py [--dry-run] [--beaker-workspace WORKSPACE]
                               [--beaker-cluster CLUSTER [CLUSTER ...]]
                               [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess

from generate_masking_configs import (
    CONFIG_PREFIX,
    RUN_CONFIGS_DIR,
    WANDB_PREFIX,
    WANDB_PROJECT,
    WANDB_SUFFIX,
)

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"
WANDB_GROUP = "ace2-var-masking-seeds-2026-06-30"


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-nc-sfno-c96-mask10-co2default-seed0.yaml
    # -> ace2-var-mask-nc-sfno-c96-mask10-co2default-seed0-v1
    suffix = pathlib.Path(config_filename).stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them."
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
        help="Beaker cluster(s) to target (ex: ai2/titan ai2/jupiter ai2/ceres).",
    )
    parser.add_argument(
        "--beaker-priority",
        default="urgent",
        help="Beaker job priority (ex: high or urgent).",
    )
    args = parser.parse_args()

    configs = sorted(path.name for path in RUN_CONFIGS_DIR.glob("*-seed*.yaml"))
    if not configs:
        raise FileNotFoundError(
            f"no seed configs in {RUN_CONFIGS_DIR} — run generate_seed_configs.py first"
        )

    env = {
        **os.environ,
        "WANDB_PROJECT": WANDB_PROJECT,
        "BEAKER_WORKSPACE": args.beaker_workspace,
        "BEAKER_CLUSTER": " ".join(args.beaker_cluster),
        "BEAKER_PRIORITY": args.beaker_priority,
    }
    for config_filename in configs:
        job_name = config_to_job_name(config_filename)
        cmd = [str(RUN_SCRIPT), config_filename, job_name, WANDB_GROUP]
        print("Submitting:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=HERE, env=env)


if __name__ == "__main__":
    main()
