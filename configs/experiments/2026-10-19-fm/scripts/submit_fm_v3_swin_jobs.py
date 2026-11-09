"""Submit the 2026-11-06 FM swin-v2.2-v3 runs (create_fm_v3_swin_jobs.py configs).

Each config is submitted via run-ace-train.sh with N_GPUS=4 into the wandb
group ace2-fm-2026-11-06 (shared with the sfno-v3 runs). Job names follow
submit_fm_jobs.config_to_job_name: ace-train-config-4deg-<stem>.yaml ->
ace2-fm-<stem>. Real submissions need the configs committed and pushed (gantry
clones HEAD); --dry-run skips that check and prints the commands.

Usage:
    python submit_fm_v3_swin_jobs.py [--arm {fm-balanced,fm-e3sm-balanced} ...]
                                     [--seed N ...] [--smoke] [--dry-run]
                                     [--skip-if-in-beaker] [--exclude-job NAME ...]
                                     [--beaker-workspace ...] [--beaker-cluster ...]
                                     [--beaker-priority ...]

--smoke submits the run_configs/v3-smoke/ swin config (job name ending in
-smoke) instead of the real runs.
"""

import argparse

from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from create_fm_v3_jobs import ARMS, SEEDS, SMOKE_ARM, SMOKE_SEED
from create_fm_v3_swin_jobs import run_config_path
from submit_fm_jobs import (
    EXP_DIR,
    RUN_SCRIPT,
    WANDB_PROJECT,
    add_default_beaker_args,
    config_to_job_name,
)
from submit_fm_v3_jobs import N_GPUS, WANDB_GROUP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seed", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    add_default_beaker_args(parser)
    args = parser.parse_args()

    if args.smoke:
        paths = [run_config_path(SMOKE_ARM, SMOKE_SEED, smoke=True)]
    else:
        paths = [run_config_path(arm, seed) for arm in args.arm for seed in args.seed]
    by_name = {path.name: path for path in paths}
    configs = drop_jobs_in_beaker(list(by_name), config_to_job_name, args)
    if not args.dry_run:
        check_configs_at_head([by_name[name] for name in configs])
    for config_filename in configs:
        submit_job(
            RUN_SCRIPT,
            [
                str(by_name[config_filename].relative_to(EXP_DIR)),
                config_to_job_name(config_filename),
                WANDB_GROUP,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=EXP_DIR,
            extra_env={"N_GPUS": N_GPUS},
        )


if __name__ == "__main__":
    main()
