"""Submit the 2026-11-05 FM v2 training jobs (create_fm_v2_jobs.py configs).

Each config is submitted via run-ace-train.sh with N_GPUS=4 into the wandb
group ace2-fm-2026-11-05. Job names follow submit_fm_jobs.config_to_job_name:
ace-train-config-4deg-<stem>.yaml -> ace2-fm-<stem>.

Usage:
    python submit_fm_v2_jobs.py [--arch {nc-sfno-v2,nc-swin-v2.2-v2} ...]
                                [--arm {fm-balanced,fm-e3sm-balanced} ...]
                                [--seed N ...] [--smoke] [--dry-run]
                                [--skip-if-in-beaker] [--exclude-job NAME ...]
                                [--beaker-workspace ...] [--beaker-cluster ...]
                                [--beaker-priority ...]

--smoke submits the run_configs/v2-smoke/ configs (job names end in -smoke)
instead of the real runs; --arch still applies.
"""

import argparse
import re

from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from create_fm_v2_jobs import ARMS, SEEDS, SMOKE_CONFIGS_DIR
from submit_fm_jobs import (
    HERE,
    RUN_CONFIGS_DIR,
    RUN_SCRIPT,
    V2_ARCH_CHOICES,
    WANDB_PROJECT,
    add_default_beaker_args,
    arch_tag,
    config_to_job_name,
    select_configs,
)

WANDB_GROUP = "ace2-fm-2026-11-05"
N_GPUS = "4"
NAME_RE = re.compile(r"-(?P<arm>fm-[a-z0-9-]+)-s(?P<seed>\d+)-maskoff(-smoke)?\.yaml$")


def parse(config_filename: str) -> tuple[str, str, int]:
    match = NAME_RE.search(config_filename)
    arch = arch_tag(config_filename)
    if match is None or arch is None:
        raise ValueError(f"unexpected v2 config name {config_filename}")
    return arch, match["arm"], int(match["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", nargs="+", choices=V2_ARCH_CHOICES, default=list(V2_ARCH_CHOICES)
    )
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seed", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    add_default_beaker_args(parser)
    args = parser.parse_args()

    if args.smoke:
        config_dir = SMOKE_CONFIGS_DIR
        names = sorted(
            p.name
            for p in SMOKE_CONFIGS_DIR.glob("*-smoke.yaml")
            if arch_tag(p.name) in args.arch
        )
    else:
        config_dir = RUN_CONFIGS_DIR
        names = []
        for name in select_configs(None, args.arch):
            _, arm, seed = parse(name)
            if arm in args.arm and seed in args.seed:
                names.append(name)
    configs = drop_jobs_in_beaker(names, config_to_job_name, args)
    if not args.dry_run:
        check_configs_at_head([config_dir / name for name in configs])
    rel_dir = config_dir.relative_to(HERE)
    for config_filename in configs:
        submit_job(
            RUN_SCRIPT,
            [
                f"{rel_dir}/{config_filename}",
                config_to_job_name(config_filename),
                WANDB_GROUP,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=HERE,
            extra_env={"N_GPUS": N_GPUS},
        )


if __name__ == "__main__":
    main()
