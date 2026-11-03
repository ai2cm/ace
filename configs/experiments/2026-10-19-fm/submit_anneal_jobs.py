"""Submit the anneal fine-tuning runs written by create_anneal_jobs.py.

Each config in run_configs/anneal/ is submitted via run-ace-anneal.sh with the
training submitter's Beaker defaults (submit_fm_jobs.py: workspace, clusters,
priority) and wandb project, in wandb group WANDB_GROUP. Before submitting,
the parent result dataset named in each config's header is checked against the
Beaker listing of the parent's succeeded experiment, and its
ema_ckpt_0150.tar is checked to exist.

Usage:
    python submit_anneal_jobs.py [--arms a0 a1 ...] [--seeds 0 1 ...]
                                 [--only JOB_NAME ...] [--smoke] [--dry-run]
                                 [--beaker-workspace WORKSPACE]
                                 [--beaker-cluster CLUSTER [CLUSTER ...]]
                                 [--beaker-priority PRIORITY]
                                 [--skip-if-in-beaker] [--exclude-job NAME ...]
"""

import argparse
import pathlib
import subprocess

from _beaker_listing import OK, fetch_experiments_by_name
from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from create_anneal_jobs import (
    ARMS,
    PARENT_CHECKPOINT,
    SEEDS,
    SMOKE_ARMS,
    SMOKE_SEED,
    anneal_config_path,
    read_header,
)
from submit_fm_jobs import WANDB_PROJECT, add_default_beaker_args

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-anneal.sh"
WANDB_GROUP = "ace2-fm-anneal-2026-10-19"


def check_parent(header: dict[str, str], experiments) -> None:
    parent, dataset = header["PARENT_JOB"], header["PARENT_DATASET"]
    named = experiments.get(parent)
    if named is None or named.status != OK:
        raise SystemExit(f"parent {parent} has not succeeded in Beaker")
    if named.result_dataset != dataset:
        raise SystemExit(
            f"parent {parent}: config header names dataset {dataset}, Beaker "
            f"has {named.result_dataset}; rerun create_anneal_jobs.py"
        )
    listing = subprocess.run(
        ["beaker", "dataset", "ls", dataset],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if PARENT_CHECKPOINT not in listing:
        raise SystemExit(f"{PARENT_CHECKPOINT} not found in {dataset} ({parent})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="JOB_NAME",
        help="Submit only these job names (after --arms/--seeds/--smoke).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Submit the 2-epoch smoke configs (arms {SMOKE_ARMS}, seed "
            f"{SMOKE_SEED}) instead of the sweep."
        ),
    )
    add_default_beaker_args(parser)
    args = parser.parse_args()

    if args.smoke:
        pairs = [(a, SMOKE_SEED) for a in SMOKE_ARMS if a in args.arms]
    else:
        pairs = [(a, s) for a in args.arms for s in args.seeds]
    paths = [anneal_config_path(a, s, args.smoke) for a, s in pairs]
    jobs = [(path, read_header(path)) for path in paths]
    if args.only is not None:
        unknown = set(args.only) - {h["JOB_NAME"] for _, h in jobs}
        if unknown:
            raise SystemExit(f"--only names not selected: {sorted(unknown)}")
        jobs = [(p, h) for p, h in jobs if h["JOB_NAME"] in args.only]
    jobs = drop_jobs_in_beaker(jobs, lambda job: job[1]["JOB_NAME"], args)

    print("Checking parent checkpoints in Beaker...")
    experiments = fetch_experiments_by_name(args.beaker_workspace)
    for _, header in jobs:
        check_parent(header, experiments)
    if not args.dry_run:
        check_configs_at_head([path for path, _ in jobs])
    for path, header in jobs:
        submit_job(
            RUN_SCRIPT,
            [
                str(path.relative_to(HERE)),
                header["JOB_NAME"],
                WANDB_GROUP,
                header["PARENT_DATASET"],
                header["DESCRIPTION"],
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=HERE,
        )


if __name__ == "__main__":
    main()
