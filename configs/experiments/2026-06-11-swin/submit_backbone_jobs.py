"""Submit a gantry training job for each generated backbone config.

Each config produced by generate_backbone_configs.py is submitted via
run-ace-train.sh, which validates the config and calls gantry.

Usage:
    python submit_backbone_jobs.py [--dry-run] [--beaker-workspace WORKSPACE]
                               [--beaker-cluster CLUSTER [CLUSTER ...]]
                               [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_PROJECT = "Backbones2"
WANDB_GROUP = "ace2-backbones-2026-06-11"

CONFIGS = sorted(
    path.name
    for path in HERE.glob("*.yaml")
    if path.name.startswith("ace-train-config-4deg-AIMIP-")
    and "hybrid-swin" in path.name
)


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-AIMIP-swin.yaml
    # → ace2-backbone
    stem = pathlib.Path(config_filename).stem  # strip .yaml
    suffix = stem.removeprefix("ace-train-config-4deg-AIMIP-")
    return f"ace2-backbone-{suffix}"


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
        help="Beaker workspace to submit jobs to (default: ai2/ace).",
    )
    parser.add_argument(
        "--beaker-cluster",
        nargs="+",
        default=["ai2/titan"],
        metavar="CLUSTER",
        help=(
            "Beaker cluster(s) to target (default: ai2/titan" "ai2/jupiter ai2/ceres)."
        ),
    )
    parser.add_argument(
        "--beaker-priority",
        default="urgent",
        help="Beaker job priority (default: high).",
    )
    args = parser.parse_args()

    for config_filename in CONFIGS:
        config_path = HERE / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found — run generate_backbone_configs.py first"
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
