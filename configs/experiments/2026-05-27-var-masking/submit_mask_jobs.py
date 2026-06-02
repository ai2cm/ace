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
    # bernoulli configs
    "ace-train-config-4deg-AIMIP-sfno-mask0.00-all-gmron-rpoff-bernoulli.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-all-gmron-rpoff-bernoulli.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.20-noforcing-gmron-rpoff-bernoulli.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.40-all-gmron-rpoff-bernoulli.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask0.40-noforcing-gmron-rpoff-bernoulli.yaml",
    "ace-train-config-4deg-AIMIP-nc-sfno-mask0.00-all-gmron-rpoff-bernoulli.yaml",
    "ace-train-config-4deg-AIMIP-nc-sfno-mask0.20-all-gmron-rpoff-bernoulli.yaml",
    "ace-train-config-4deg-AIMIP-nc-sfno-mask0.20-noforcing-gmron-rpoff-bernoulli.yaml",
    "ace-train-config-4deg-AIMIP-nc-sfno-mask0.40-all-gmron-rpoff-bernoulli.yaml",
    "ace-train-config-4deg-AIMIP-nc-sfno-mask0.40-noforcing-gmron-rpoff-bernoulli.yaml",
    # uniform configs
    "ace-train-config-4deg-AIMIP-sfno-maskall-all-gmron-rpoff-uniform.yaml",
    "ace-train-config-4deg-AIMIP-sfno-maskall-noforcing-gmron-rpoff-uniform.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask17-all-gmron-rpoff-uniform.yaml",
    "ace-train-config-4deg-AIMIP-sfno-mask15-noforcing-gmron-rpoff-uniform.yaml",
    "ace-train-config-4deg-AIMIP-nc-sfno-maskall-all-gmron-rpoff-uniform.yaml",
    "ace-train-config-4deg-AIMIP-nc-sfno-maskall-noforcing-gmron-rpoff-uniform.yaml",
    "ace-train-config-4deg-AIMIP-nc-sfno-mask17-all-gmron-rpoff-uniform.yaml",
    "ace-train-config-4deg-AIMIP-nc-sfno-mask15-noforcing-gmron-rpoff-uniform.yaml",
]


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
