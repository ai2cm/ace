"""Submit the 2026-11-09 FM v2 training jobs (create_fm_v2_jobs.py configs).

Each config is submitted via run-ace-train.sh with N_GPUS=4 into the wandb
group ace2-fm-2026-11-09. Job names follow submit_fm_jobs.config_to_job_name:
ace-train-config-4deg-<stem>.yaml -> ace2-fm-<stem>.

Beaker priority defaults to `high` for seed 0 and `normal` for every other
seed; --beaker-priority overrides it for all seeds.

Usage:
    python submit_fm_v2_jobs.py [--arch {nc-sfno-v2,nc-swin-v2.2-v2} ...]
                                [--arm {fm-era5,fm-balanced,fm-e3sm-balanced} ...]
                                [--seed N ...] [--dry-run]
                                [--skip-if-in-beaker] [--exclude-job NAME ...]
                                [--beaker-workspace ...] [--beaker-cluster ...]
                                [--beaker-priority ...]
"""

import argparse
import copy
import re

from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from create_fm_v2_jobs import ARMS, SEEDS
from submit_fm_jobs import (
    EXP_DIR,
    RUN_CONFIGS_DIR,
    RUN_SCRIPT,
    V2_ARCH_CHOICES,
    WANDB_PROJECT,
    add_default_beaker_args,
    arch_tag,
    config_to_job_name,
    select_configs,
)

WANDB_GROUP = "ace2-fm-2026-11-09"
N_GPUS = "4"
FIRST_SEED_PRIORITY = "high"
OTHER_SEED_PRIORITY = "normal"
NAME_RE = re.compile(r"-(?P<arm>fm-[a-z0-9-]+)-s(?P<seed>\d+)-maskoff\.yaml$")


def parse(config_filename: str) -> tuple[str, str, int]:
    match = NAME_RE.search(config_filename)
    arch = arch_tag(config_filename)
    if match is None or arch is None:
        raise ValueError(f"unexpected v2 config name {config_filename}")
    return arch, match["arm"], int(match["seed"])


def default_priority(seed: int) -> str:
    return FIRST_SEED_PRIORITY if seed == SEEDS[0] else OTHER_SEED_PRIORITY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", nargs="+", choices=V2_ARCH_CHOICES, default=list(V2_ARCH_CHOICES)
    )
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seed", nargs="+", type=int, default=list(SEEDS))
    add_default_beaker_args(parser)
    # Per-seed default instead of the shared one; an explicit value wins.
    parser.set_defaults(beaker_priority=None)
    args = parser.parse_args()

    names = []
    for name in select_configs(None, args.arch):
        _, arm, seed = parse(name)
        if arm in args.arm and seed in args.seed:
            names.append(name)
    configs = drop_jobs_in_beaker(names, config_to_job_name, args)
    if not args.dry_run:
        check_configs_at_head([RUN_CONFIGS_DIR / name for name in configs])
    rel_dir = RUN_CONFIGS_DIR.relative_to(EXP_DIR)
    for config_filename in configs:
        _, _, seed = parse(config_filename)
        job_args = copy.copy(args)
        if args.beaker_priority is None:
            job_args.beaker_priority = default_priority(seed)
        submit_job(
            RUN_SCRIPT,
            [
                f"{rel_dir}/{config_filename}",
                config_to_job_name(config_filename),
                WANDB_GROUP,
            ],
            wandb_project=WANDB_PROJECT,
            args=job_args,
            cwd=EXP_DIR,
            extra_env={"N_GPUS": N_GPUS},
        )


if __name__ == "__main__":
    main()
