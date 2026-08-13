"""Submit a gantry training job for each multi-step fine-tuning config.

Each ``*-mstepft.yaml`` config in run_configs/ (from
generate_finetune_configs.py) is submitted via run-ace-train.sh, which validates
it and calls gantry. All fine-tunes are v5 (1-degree), so they take the 8-GPU /
400GiB footprint like the v5 seed replicates.

Usage:
    python submit_finetune_jobs.py [--dry-run]
                                   [--beaker-workspace WORKSPACE]
                                   [--beaker-cluster CLUSTER [CLUSTER ...]]
                                   [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess

from generate_masking_configs import CONFIG_PREFIX, RUN_CONFIGS_DIR, WANDB_PREFIX

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"
WANDB_PROJECT = "VarMasking8"
WANDB_GROUP = "ace2-var-masking-mstepft-2026-06-30"

# All fine-tunes are v5 (1-degree); match the v5 seed-run footprint (the
# run-ace-train.sh defaults of N_GPUS=2 / 100GiB are for the 4-degree runs).
V5_N_GPUS = "8"
V5_SHARED_MEMORY = "400GiB"


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-nc-sfno-era5-gmroff-mask0-seed1-v5-mstepft.yaml
    # -> ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5-mstepft
    suffix = pathlib.Path(config_filename).stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}"


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
        default="high",
        help="Beaker job priority (ex: high or urgent).",
    )
    args = parser.parse_args()

    configs = sorted(
        path.name
        for path in RUN_CONFIGS_DIR.glob("*-mstepft.yaml")
        if path.name.startswith(CONFIG_PREFIX)
    )
    if not configs:
        raise FileNotFoundError(
            f"no fine-tune configs in {RUN_CONFIGS_DIR} — run "
            "generate_finetune_configs.py first"
        )

    base_env = {
        **os.environ,
        "WANDB_PROJECT": WANDB_PROJECT,
        "BEAKER_WORKSPACE": args.beaker_workspace,
        "BEAKER_CLUSTER": " ".join(args.beaker_cluster),
        "BEAKER_PRIORITY": args.beaker_priority,
        "N_GPUS": V5_N_GPUS,
        "BEAKER_SHARED_MEMORY": V5_SHARED_MEMORY,
    }
    for config_filename in configs:
        config_text = (RUN_CONFIGS_DIR / config_filename).read_text()
        if "REPLACE_WITH_BEAKER_DATASET_ID" in config_text:
            raise ValueError(
                f"{config_filename} still contains a placeholder dataset ID — "
                "run generate_finetune_configs.py with a real source map first."
            )
        job_name = config_to_job_name(config_filename)
        cmd = [str(RUN_SCRIPT), config_filename, job_name, WANDB_GROUP]
        print(f"Submitting ({WANDB_PROJECT}):", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=HERE, env=base_env)


if __name__ == "__main__":
    main()
