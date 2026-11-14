"""Submit the 1deg, 6-hourly FM training jobs (create_fm_1deg_jobs.py configs).

Each config is submitted via run-ace-train.sh with N_GPUS=8 and
SHARED_MEMORY=400GiB into the wandb group ace2-fm-1deg-2026-11-13. Job names
follow submit_fm_jobs.config_to_job_name:
ace-train-config-1deg-<stem>.yaml -> ace2-fm-1deg-<stem>.

--smoke submits the smoke-test configs (...-s0-smoke.yaml, one per arm)
instead, into the wandb group ace2-fm-1deg-2026-11-13-smoke.

The default clusters (ai2/jupiter, ai2/titan, ai2/saturn) all have 8-GPU
nodes. Beaker priority defaults to `normal`.

Real submissions need the configs committed and pushed (gantry clones HEAD);
--dry-run skips that check and prints the commands without submitting.

Usage:
    python submit_fm_1deg_jobs.py [--arm {fm-e3sm-balanced,fm-era5} ...]
                                  [--seed N ...] [--smoke] [--dry-run]
                                  [--skip-if-in-beaker] [--exclude-job NAME ...]
                                  [--beaker-workspace ...] [--beaker-cluster ...]
                                  [--beaker-priority ...]
"""

import argparse

from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from create_fm_1deg_jobs import ARMS, SEEDS, SMOKE_SEED, run_config_name
from submit_fm_jobs import (
    EXP_DIR,
    RUN_CONFIGS_DIR,
    RUN_SCRIPT,
    WANDB_PREFIX,
    WANDB_PROJECT,
    add_default_beaker_args,
    config_to_job_name,
)

WANDB_GROUP = "ace2-fm-1deg-2026-11-13"
SMOKE_WANDB_GROUP = f"{WANDB_GROUP}-smoke"
N_GPUS = "8"
SHARED_MEMORY = "400GiB"  # as configs/baselines/era5/run-ace-train.sh
# Narrow Beaker --text filter matching every 1deg job name.
LIST_TEXT = f"{WANDB_PREFIX}1deg-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=None,
        help=(
            f"Seeds to submit (default: all, {' '.join(map(str, SEEDS))}). "
            f"Not allowed with --smoke, which is seed {SMOKE_SEED} only."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Submit the smoke-test configs into {SMOKE_WANDB_GROUP} instead.",
    )
    add_default_beaker_args(parser)
    args = parser.parse_args()
    if args.smoke and args.seed is not None:
        parser.error("--seed is not allowed with --smoke")

    seeds = [SMOKE_SEED] if args.smoke else args.seed or list(SEEDS)
    names = [
        run_config_name(arm, seed, args.smoke) for arm in args.arm for seed in seeds
    ]
    missing = [name for name in names if not (RUN_CONFIGS_DIR / name).exists()]
    if missing:
        raise SystemExit(
            "Configs not generated (create_fm_1deg_jobs.py):\n  " + "\n  ".join(missing)
        )
    configs = drop_jobs_in_beaker(names, config_to_job_name, args, [LIST_TEXT])
    if configs and not args.dry_run:
        check_configs_at_head([RUN_CONFIGS_DIR / name for name in configs])
    rel_dir = RUN_CONFIGS_DIR.relative_to(EXP_DIR)
    env = {"N_GPUS": N_GPUS, "SHARED_MEMORY": SHARED_MEMORY}
    group = SMOKE_WANDB_GROUP if args.smoke else WANDB_GROUP
    if args.dry_run:
        print("env:", " ".join(f"{key}={value}" for key, value in env.items()))
        print(f"clusters: {' '.join(args.beaker_cluster)}")
    for config_filename in configs:
        submit_job(
            RUN_SCRIPT,
            [
                f"{rel_dir}/{config_filename}",
                config_to_job_name(config_filename),
                group,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=EXP_DIR,
            extra_env=env,
        )


if __name__ == "__main__":
    main()
