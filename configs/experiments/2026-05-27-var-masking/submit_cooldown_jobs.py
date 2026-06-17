"""Submit a gantry training job for each generated var-masking cooldown config.

Each config produced by generate_cooldown_configs.py is submitted via
run-ace-train.sh, which validates the config and calls gantry.

Usage:
    python submit_cooldown_jobs.py [--dry-run] [--beaker-workspace WORKSPACE]
                                   [--beaker-cluster CLUSTER [CLUSTER ...]]
                                   [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_PROJECT = "VarMasking4"
WANDB_GROUP = "ace2-var-masking-cooldown-2026-06-17"

CONFIGS = sorted(
    path.name
    for path in HERE.glob("*-cooldown.yaml")
    if path.name.startswith("ace-train-config-4deg-AIMIP-")
)


def config_to_job_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix("ace-train-config-4deg-AIMIP-")
    suffix = suffix.removesuffix("-cooldown")
    return f"ace2-var-mask-{suffix}-v4-cooldown"


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
        help="Beaker workspace to submit jobs to (ex: ai2/ace or ai2/climate-titan).",
    )
    parser.add_argument(
        "--beaker-cluster",
        nargs="+",
        default=["ai2/titan"],
        metavar="CLUSTER",
        help=(
            "Beaker cluster(s) to target (ex: ai2/titan ai2/saturn "
            "ai2/jupiter ai2/ceres)."
        ),
    )
    parser.add_argument(
        "--beaker-priority",
        default="urgent",
        help="Beaker job priority (ex: urgent or high).",
    )
    args = parser.parse_args()

    for config_filename in CONFIGS:
        config_path = HERE / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found"
                " — run generate_cooldown_configs.py first"
            )
        config_text = config_path.read_text()
        if "REPLACE_WITH_BEAKER_DATASET_ID" in config_text:
            raise ValueError(
                f"{config_filename} still contains a placeholder dataset ID — "
                "run generate_cooldown_configs.py --source-map <sources.json> "
                "with real Beaker dataset IDs first."
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
