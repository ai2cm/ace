"""Submit the 2026-11-12 fine-tuning sweep of the ext80 runs
(create_fm_v2_finetune_jobs.py configs in run_configs/ft/), with N_GPUS=4.

Each job's name, wandb group, parent job and description come from its config
header. Jobs go through run-ace-anneal.sh, as the ext80 forks do
(submit_fm_v2_followup_jobs.py), with the ext80 parent's
training_checkpoints/ema_ckpt_0080.tar mounted at /parent_ckpt.tar. The
parent's result dataset is resolved at submit time, not from
records/wandb_to_beaker_map.json: the parent's Beaker experiment(s) are found
by canonical job name and the result datasets of their jobs searched for the
checkpoint (submit_fm_v2_followup_jobs.find_checkpoint; a preempted parent
resumes in a new job with a new result dataset, and only the job that
finished holds ema_ckpt_0080.tar). One lookup per parent, shared by its 16
children. A child whose parent has no such checkpoint in a finished job's
dataset is skipped with a message (a running job's dataset is uncommitted and
cannot be mounted).

Beaker priority defaults to `normal`. Real submissions need the configs
committed and pushed (gantry clones the commit from GitHub); --dry-run skips
that check, still queries Beaker (read-only) for the parent datasets, and
prints the commands without submitting. --list prints the selected job names
and configs without touching Beaker.

Usage:
    python submit_fm_v2_finetune_jobs.py [--arch nc-sfno-v2 nc-swin-v2.2-v2]
        [--seed N ...] [--mix {25,50,70,100} ...] [--lr {1e-4,3e-5,1e-5,3e-6} ...]
        [--only JOB_NAME ...] [--list] [--dry-run] [--skip-if-in-beaker]
        [--exclude-job NAME ...] [--beaker-workspace ...] [--beaker-cluster ...]
        [--beaker-priority ...]
"""

import argparse

from _beaker_listing import list_experiments
from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from create_anneal_jobs import read_header
from create_fm_v2_finetune_jobs import (
    LRS,
    MIXES,
    N_GPUS,
    PARENT_ARM,
    PARENT_CHECKPOINT,
    FtRun,
    ft_runs,
)
from create_fm_v2_followup_jobs import EXT80_ARCHS
from create_fm_v2_jobs import SEEDS
from submit_fm_jobs import EXP_DIR, WANDB_PREFIX, WANDB_PROJECT, add_default_beaker_args
from submit_fm_v2_followup_jobs import FORK_SCRIPT, find_checkpoint


def list_texts(runs: list[FtRun]) -> list[str]:
    """Narrow Beaker --text filters, one per arch, each covering the arch's
    parents and their ft children.
    """
    return sorted({f"{WANDB_PREFIX}{run.parent.tag}-{PARENT_ARM}-" for run in runs})


def parent_datasets(runs: list[FtRun], workspace: str) -> dict[str, str]:
    """Parent job name -> committed result dataset holding PARENT_CHECKPOINT,
    for the parents that have one.
    """
    experiments: list[dict] = []
    for text in list_texts(runs):
        experiments += list_experiments(workspace, text=text)
    datasets = {}
    for parent in dict.fromkeys(run.parent.job_name for run in runs):
        found = find_checkpoint(parent, experiments, PARENT_CHECKPOINT)
        if found is None:
            print(f"Parent {parent}: no {PARENT_CHECKPOINT} yet")
        elif found[1]:
            print(f"Parent {parent}: {PARENT_CHECKPOINT} only in a running job")
        else:
            print(f"Parent {parent}: {PARENT_CHECKPOINT} in dataset {found[0]}")
            datasets[parent] = found[0]
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arch", nargs="+", choices=EXT80_ARCHS, default=list(EXT80_ARCHS)
    )
    parser.add_argument("--seed", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--mix", nargs="+", choices=list(MIXES), default=list(MIXES))
    parser.add_argument("--lr", nargs="+", choices=LRS, default=list(LRS))
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="JOB_NAME",
        help="Submit only these job names (after the other selections).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected job names and configs, then exit.",
    )
    add_default_beaker_args(parser)
    args = parser.parse_args()

    runs = [
        run
        for run in ft_runs(tuple(args.arch))
        if run.seed in args.seed and run.mix in args.mix and run.lr in args.lr
    ]
    if args.only is not None:
        unknown = set(args.only) - {run.job_name for run in runs}
        if unknown:
            raise SystemExit(f"--only names not selected: {sorted(unknown)}")
        runs = [run for run in runs if run.job_name in args.only]
    missing = [run.path.name for run in runs if not run.path.exists()]
    if missing:
        raise SystemExit(
            "Configs not generated (create_fm_v2_finetune_jobs.py):\n  "
            + "\n  ".join(missing)
        )
    if args.list:
        for run in runs:
            print(f"{run.job_name}  {run.path.relative_to(EXP_DIR)}")
        print(f"{len(runs)} jobs.")
        return

    runs = drop_jobs_in_beaker(runs, lambda run: run.job_name, args, list_texts(runs))
    if not runs:
        return
    print("Looking for parent checkpoints in Beaker...")
    datasets = parent_datasets(runs, args.beaker_workspace)
    ready = []
    for run in runs:
        header = read_header(run.path)
        assert header["JOB_NAME"] == run.job_name
        assert header["PARENT_JOB"] == run.parent.job_name
        assert header["WANDB_GROUP"] == run.group
        assert header["PARENT_CHECKPOINT"].startswith(f"{PARENT_CHECKPOINT} ")
        if run.parent.job_name not in datasets:
            print(f"Skipping (parent checkpoint not ready): {run.job_name}")
            continue
        ready.append((run, header, datasets[run.parent.job_name]))
    print(f"{len(ready)} to submit.")
    if ready and not args.dry_run:
        check_configs_at_head([run.path for run, _, _ in ready])
    env = {"N_GPUS": str(N_GPUS), "PARENT_CHECKPOINT": PARENT_CHECKPOINT}
    for run, header, dataset in ready:
        submit_job(
            FORK_SCRIPT,
            [
                str(run.path.relative_to(EXP_DIR)),
                run.job_name,
                header["WANDB_GROUP"],
                dataset,
                header["DESCRIPTION"],
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=EXP_DIR,
            extra_env=env,
        )
    if args.dry_run and ready:
        print("env: " + " ".join(f"{k}={v}" for k, v in env.items()))


if __name__ == "__main__":
    main()
