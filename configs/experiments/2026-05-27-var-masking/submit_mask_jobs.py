"""Submit a gantry training job for each generated var-masking config.

Each config produced by generate_masking_configs.py is submitted via
run-ace-train.sh, which validates the config and calls gantry.

Usage:
    python submit_mask_jobs.py [--dry-run]
"""

import argparse
import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_PROJECT = "VarMasking"
WANDB_GROUP = "ace2-var-masking-2026-05-27"

CONFIGS = [
    "ace-train-config-4deg-AIMIP-sfno-mask0.00-uniform-gmron-rpon.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.00-uniform-gmron-rpoff.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.00-uniform-gmroff-rpon.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.00-uniform-gmroff-rpoff.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-uniform-gmron-rpon.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-uniform-gmron-rpoff.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-uniform-gmroff-rpon.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-uniform-gmroff-rpoff.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-forcing-gmron-rpon.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-forcing-gmron-rpoff.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-forcing-gmroff-rpon.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-forcing-gmroff-rpoff.yaml",
]
# "ace-train-config-4deg-AIMIP-sfno-mask0.80-uniform-gmron-rpon.yaml",
# "ace-train-config-4deg-AIMIP-sfno-mask0.80-uniform-gmron-rpoff.yaml",
# "ace-train-config-4deg-AIMIP-sfno-mask0.80-uniform-gmroff-rpon.yaml",
# "ace-train-config-4deg-AIMIP-sfno-mask0.80-uniform-gmroff-rpoff.yaml",
# "ace-train-config-4deg-AIMIP-sfno-mask0.80-forcing-gmron-rpon.yaml",
# "ace-train-config-4deg-AIMIP-sfno-mask0.80-forcing-gmron-rpoff.yaml",
# "ace-train-config-4deg-AIMIP-sfno-mask0.80-forcing-gmroff-rpon.yaml",
# "ace-train-config-4deg-AIMIP-sfno-mask0.80-forcing-gmroff-rpoff.yaml",
# ]


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-AIMIP-sfno-mask0.20-forcing-gmron-rpon.yaml
    # → ace2-var-mask-sfno-mask0.20-forcing-gmron-rpon
    stem = pathlib.Path(config_filename).stem  # strip .yaml
    suffix = stem.removeprefix("ace-train-config-4deg-AIMIP-")
    return f"ace2-var-mask-{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    for config_filename in CONFIGS:
        config_path = HERE / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found — run generate_masking_configs.py first"
            )
        job_name = config_to_job_name(config_filename)
        cmd = [str(RUN_SCRIPT), config_filename, job_name, WANDB_GROUP]
        print("Submitting:", " ".join(cmd))
        if not args.dry_run:
            env = {**os.environ, "WANDB_PROJECT": WANDB_PROJECT}
            subprocess.run(cmd, check=True, cwd=HERE, env=env)


if __name__ == "__main__":
    main()
