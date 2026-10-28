"""Submit a gantry training job for each foundation model (fm) run config.

Each config in run_configs/ (written by create_fm_jobs.py) is submitted via
run-ace-train.sh, which validates the config and calls gantry.

Usage:
    python submit_fm_jobs.py [--version {v1,v2,v3}]
                             [--arch ARCH [ARCH ...]] [--dry-run]
                             [--beaker-workspace WORKSPACE]
                             [--beaker-cluster CLUSTER [CLUSTER ...]]
                             [--beaker-priority PRIORITY]
                             [--skip-if-in-beaker] [--exclude-job NAME ...]
                             [--skip-if-in-wandb]
"""

import argparse
import pathlib
from collections.abc import Sequence

from _submit_common import (
    add_beaker_args,
    check_configs_at_head,
    drop_jobs_in_beaker,
    submit_job,
)
from _version_select import add_version_arg, stem_matches_version

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_ENTITY = "ai2cm"
WANDB_PROJECT = "FM"
WANDB_PREFIX = "ace2-fm-"
WANDB_GROUP = "ace2-fm-2026-10-19"
CONFIG_PREFIX = "ace-train-config-4deg-"
# Architecture tags appearing in config filenames, directly after
# CONFIG_PREFIX. Matching is exact-tag, so `nc-swin-v2` never selects a
# `nc-swin-v2.2` config (or vice versa).
ARCH_CHOICES = ("nc-sfno", "nc-swin-v2", "nc-swin-v2.1", "nc-swin-v2.2")
# Beaker defaults for add_beaker_args, shared with submit_inference_jobs.py so
# inference jobs land where training jobs do. Keep in step with the
# BEAKER_* defaults in run-ace-train.sh / run-ace-inference-job.sh.
BEAKER_WORKSPACE: str = "ai2/ace"
BEAKER_CLUSTERS: list[str] = ["ai2/jupiter", "ai2/titan", "ai2/saturn"]
BEAKER_PRIORITY: str = "normal"


def add_default_beaker_args(parser: argparse.ArgumentParser) -> None:
    """add_beaker_args with the training submitter's defaults."""
    add_beaker_args(
        parser,
        default_workspace=BEAKER_WORKSPACE,
        default_cluster=list(BEAKER_CLUSTERS),
        default_priority=BEAKER_PRIORITY,
    )


def select_configs(version: str | None, archs: Sequence[str]) -> list[str]:
    # Training configs only: cooldown/finetune configs are not training runs
    # from scratch, and eval suites use a different prefix. `-diag` configs
    # are one-off diagnostics submitted by hand via run-ace-train.sh; they
    # are short runs that would waste a slot here.
    arch_prefixes = tuple(f"{CONFIG_PREFIX}{arch}-" for arch in archs)
    return sorted(
        path.name
        for path in RUN_CONFIGS_DIR.glob("*.yaml")
        if path.name.startswith(arch_prefixes)
        and stem_matches_version(path.stem, version)
        and not path.name.endswith("-finetune.yaml")
        and not path.name.endswith("-cooldown.yaml")
        and not path.name.endswith("-bestinfcooldown.yaml")
        and not path.name.endswith("-diag.yaml")
    )


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-nc-sfno-fm-a2-cond-era5-c96-s0-maskon.yaml
    # → ace2-fm-nc-sfno-fm-a2-cond-era5-c96-s0-maskon
    stem = pathlib.Path(config_filename).stem  # strip .yaml
    return f"{WANDB_PREFIX}{stem.removeprefix(CONFIG_PREFIX)}"


def wandb_run_names() -> set[str]:
    import wandb  # lazy import: keeps the module importable without wandb

    api = wandb.Api()
    return {run.name for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
    parser.add_argument(
        "--arch",
        nargs="+",
        choices=ARCH_CHOICES,
        default=list(ARCH_CHOICES),
        help=(
            "Restrict submission to configs whose filename carries one of these "
            "architecture tags, e.g. 'nc-swin-v2.2' selects "
            f"{CONFIG_PREFIX}nc-swin-v2.2-*.yaml only. Defaults to "
            "all architectures."
        ),
    )
    add_default_beaker_args(parser)
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

    configs = drop_jobs_in_beaker(
        select_configs(args.version, args.arch), config_to_job_name, args
    )
    if not args.dry_run:
        check_configs_at_head([RUN_CONFIGS_DIR / name for name in configs])
    for config_filename in configs:
        job_name = config_to_job_name(config_filename)
        if job_name in existing_runs:
            print(f"Skipping (already in wandb): {job_name}")
            continue
        submit_job(
            RUN_SCRIPT,
            [f"{RUN_CONFIGS_DIRNAME}/{config_filename}", job_name, WANDB_GROUP],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=HERE,
        )


if __name__ == "__main__":
    main()
