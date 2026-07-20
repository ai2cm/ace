"""Submit a gantry training job for each generated var-masking cooldown config.

Each config produced by generate_cooldown_configs.py (in run_configs/) is
submitted via run-ace-train.sh, which validates the config and calls gantry.

Usage:
    python submit_cooldown_jobs.py [--dry-run] [--version {v1,v2,v3}]
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
WANDB_GROUP = "ace2-var-masking-cooldown-2026-06-30"


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-nc-sfno-era5-mask10-co2default-v1-cooldown.yaml
    # -> ace2-var-mask-nc-sfno-era5-mask10-co2default-v1-cooldown
    suffix = pathlib.Path(config_filename).stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}"


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
        for path in RUN_CONFIGS_DIR.glob("*cooldown.yaml")
        if path.name.startswith(CONFIG_PREFIX)
        and (args.version is None or stem_has_version(path.stem, args.version))
    )
    if not configs:
        raise FileNotFoundError(
            f"no cooldown configs in {RUN_CONFIGS_DIR}"
            " — run generate_cooldown_configs.py first"
        )

    for config_filename in configs:
        config_path = RUN_CONFIGS_DIR / config_filename
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
