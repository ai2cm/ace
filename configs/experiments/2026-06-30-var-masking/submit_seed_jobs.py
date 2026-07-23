"""Submit a gantry training job for each generated seed-replicate config.

Each ``*-seed*.yaml`` config in run_configs/ (from generate_seed_configs.py) is
submitted via run-ace-train.sh, which validates it and calls gantry.

Usage:
    python submit_seed_jobs.py [--dry-run] [--version {v1,v2,v3,v4}]
                               [--beaker-workspace WORKSPACE]
                               [--beaker-cluster CLUSTER [CLUSTER ...]]
                               [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess

from generate_masking_configs import (
    BASE_CONFIG_FILENAMES,
    CONFIG_PREFIX,
    RUN_CONFIGS_DIR,
    WANDB_PREFIX,
    WANDB_PROJECT,
    stem_has_version,
)

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"
WANDB_GROUP = "ace2-var-masking-seeds-2026-06-30"

# v5 sources the 1-degree, native 6-hourly baseline (see
# baseline_configs/versions.md); its larger grid, batch, and heavier model need
# more GPUs and shared memory than the 4-degree v1-v4 defaults in
# run-ace-train.sh (N_GPUS=2, 100GiB).
V5_N_GPUS = "8"
V5_SHARED_MEMORY = "400GiB"


def resource_env_for_config(config_filename: str) -> dict[str, str]:
    """Per-config N_GPUS / shared-memory overrides for run-ace-train.sh.

    v5 is 1-degree and needs the 8-GPU, 400GiB baseline footprint; v1-v4 keep
    the script defaults.
    """
    if stem_has_version(pathlib.Path(config_filename).stem, "v5"):
        return {"N_GPUS": V5_N_GPUS, "BEAKER_SHARED_MEMORY": V5_SHARED_MEMORY}
    return {}


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-nc-sfno-era5-mask10-co2default-seed0-v1.yaml
    # -> ace2-var-mask-nc-sfno-era5-mask10-co2default-seed0-v1
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
        default="urgent",
        help="Beaker job priority (ex: high or urgent).",
    )
    parser.add_argument(
        "--version",
        "-v",
        choices=sorted(BASE_CONFIG_FILENAMES),
        default=None,
        help="Restrict to configs of this baseline version (default: all).",
    )
    args = parser.parse_args()

    configs = sorted(
        path.name
        for path in RUN_CONFIGS_DIR.glob("*-seed*.yaml")
        if args.version is None or stem_has_version(path.stem, args.version)
    )
    if not configs:
        raise FileNotFoundError(
            f"no seed configs in {RUN_CONFIGS_DIR} — run generate_seed_configs.py first"
        )

    base_env = {
        **os.environ,
        "BEAKER_WORKSPACE": args.beaker_workspace,
        "BEAKER_CLUSTER": " ".join(args.beaker_cluster),
        "BEAKER_PRIORITY": args.beaker_priority,
    }
    for config_filename in configs:
        job_name = config_to_job_name(config_filename)
        env = {
            **base_env,
            "WANDB_PROJECT": WANDB_PROJECT,
            **resource_env_for_config(config_filename),
        }
        cmd = [str(RUN_SCRIPT), config_filename, job_name, WANDB_GROUP]
        print(f"Submitting ({WANDB_PROJECT}):", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=HERE, env=env)


if __name__ == "__main__":
    main()
