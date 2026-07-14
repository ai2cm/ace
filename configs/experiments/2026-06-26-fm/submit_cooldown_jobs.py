"""Submit a gantry training job for each generated FM cooldown config.

Each config produced by generate_cooldown_configs.py is submitted via
run-ace-train.sh, which validates the config and calls gantry.

Usage:
    python submit_cooldown_jobs.py [--version {v1,v2}] [--dry-run]
                                   [--beaker-workspace WORKSPACE]
                                   [--beaker-cluster CLUSTER [CLUSTER ...]]
                                   [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import subprocess

from _version_select import add_version_arg, stem_matches_version

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_PROJECT = "FM"
WANDB_GROUP = "ace2-fm-cooldown-2026-06-26"
CONFIG_PREFIX = "ace-train-config-4deg-AIMIP-"


def configs_for_version(version: str | None) -> list[str]:
    return sorted(
        path.name
        for path in RUN_CONFIGS_DIR.glob("*cooldown.yaml")
        if path.name.startswith(CONFIG_PREFIX)
        and stem_matches_version(path.stem, version)
    )


def config_to_job_name(config_filename: str) -> str:
    # The version tag (-v1 / -v2) already sits before the cooldown suffix in
    # the filename, e.g. ...-nc-sfno-fm-0.1-v1-cooldown.yaml → run name
    # ace2-fm-nc-sfno-fm-0.1-v1-cooldown.
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"ace2-fm-{suffix}"


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
        default="ai2/ace",
        help="Beaker workspace to submit jobs to (ex: ai2/ace or ai2/climate-titan).",
    )
    parser.add_argument(
        "--beaker-cluster",
        nargs="+",
        default=["ai2/titan", "ai2/jupiter", "ai2/ceres"],
        metavar="CLUSTER",
        help=(
            "Beaker cluster(s) to target (ex: ai2/titan ai2/saturn "
            "ai2/jupiter ai2/ceres)."
        ),
    )
    parser.add_argument(
        "--beaker-priority",
        default="high",
        help="Beaker job priority (ex: urgent or high).",
    )
    args = parser.parse_args()

    configs = configs_for_version(args.version)
    for config_filename in configs:
        config_path = RUN_CONFIGS_DIR / config_filename
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
        cmd = [
            str(RUN_SCRIPT),
            f"{RUN_CONFIGS_DIRNAME}/{config_filename}",
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
