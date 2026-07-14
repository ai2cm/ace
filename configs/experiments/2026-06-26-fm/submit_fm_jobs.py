"""Submit a gantry training job for each nc-sfno foundation model (fm) config.

Each nc-sfno config in this directory is submitted via run-ace-train.sh, which
validates the config and calls gantry.

Usage:
    python submit_fm_jobs.py [--version {v1,v2}] [--dry-run]
                             [--beaker-workspace WORKSPACE]
                             [--beaker-cluster CLUSTER [CLUSTER ...]]
                             [--beaker-priority PRIORITY]
                             [--skip-if-in-wandb]
"""

import argparse
import os
import pathlib
import subprocess

from _version_select import add_version_arg, stem_matches_version

HERE = pathlib.Path(__file__).parent
BASE_CONFIGS_DIR = HERE / "base_configs"
BASE_CONFIGS_DIRNAME = BASE_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_ENTITY = "ai2cm"
WANDB_PROJECT = "FM"
WANDB_PREFIX = "ace2-fm-"
WANDB_GROUP = "ace2-fm-2026-06-26"
CONFIG_PREFIX = "ace-train-config-4deg-"
# Dataset tag stripped from job names so that
# ace-train-config-4deg-AIMIP-nc-sfno-v3.yaml -> ace2-fm-nc-sfno-v3. Configs
# without the tag keep their full suffix.
DATASET_TAG = "AIMIP-"


def configs_for_version(version: str | None) -> list[str]:
    # Training configs only: cooldown configs are submitted by
    # submit_cooldown_jobs.py, and eval suites use a different prefix.
    return sorted(
        path.name
        for path in BASE_CONFIGS_DIR.glob("*.yaml")
        if path.name.startswith(CONFIG_PREFIX)
        and "nc-sfno" in path.name
        and stem_matches_version(path.stem, version)
        and not path.name.endswith("-finetune.yaml")
        and not path.name.endswith("-cooldown.yaml")
        and not path.name.endswith("-bestinfcooldown.yaml")
    )


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-AIMIP-nc-sfno-fm-0.1-v1.yaml
    # → ace2-fm-nc-sfno-fm-0.1-v1
    # ace-train-config-4deg-AIMIP-nc-sfno-c96-v1.yaml → ace2-fm-nc-sfno-c96-v1
    stem = pathlib.Path(config_filename).stem  # strip .yaml
    suffix = stem.removeprefix(CONFIG_PREFIX).removeprefix(DATASET_TAG)
    return f"{WANDB_PREFIX}{suffix}"


def wandb_run_names() -> set[str]:
    import wandb  # lazy import: keeps the module importable without wandb

    api = wandb.Api()
    return {run.name for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
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
    parser.add_argument(
        "--skip-if-in-wandb",
        action="store_true",
        help=(
            "Skip configs whose job name already exists as a run in the "
            f"{WANDB_ENTITY}/{WANDB_PROJECT} wandb project."
        ),
    )
    args = parser.parse_args()

    existing_runs = wandb_run_names() if args.skip_if_in_wandb else set()

    configs = configs_for_version(args.version)
    for config_filename in configs:
        config_path = BASE_CONFIGS_DIR / config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"{config_filename} not found")
        job_name = config_to_job_name(config_filename)
        if job_name in existing_runs:
            print(f"Skipping (already in wandb): {job_name}")
            continue
        cmd = [
            str(RUN_SCRIPT),
            f"{BASE_CONFIGS_DIRNAME}/{config_filename}",
            job_name,
            WANDB_GROUP,
        ]
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
