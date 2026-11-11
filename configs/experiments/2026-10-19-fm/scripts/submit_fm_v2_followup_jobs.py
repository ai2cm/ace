"""Submit the 2026-11-11 follow-ups of the FM v2 runs
(create_fm_v2_followup_jobs.py configs), with N_GPUS=4 as the v2 runs.

Each job's name and wandb group come from its config header (JOB_NAME,
WANDB_GROUP). Beaker priority defaults to `normal`.

- `lrsweep` (phase A, 3 jobs) and `e120 --lr X` (phase C, 12 jobs): from
  scratch, via run-ace-train.sh, as submit_fm_v2_jobs.py.
- `ext80` (phase B, 12 jobs per --arch): forks, via run-ace-anneal.sh with the
  parent's training_checkpoints/ema_ckpt_0040.tar mounted at /parent_ckpt.tar,
  as the sfno-v3 forks (backup/exp-alexeyfm2-eff922d97, submit_fm_v3_jobs.py).
  The parent's Beaker experiment(s) are found by canonical job name and the
  result datasets of all their jobs searched, finished jobs first, each newest
  first, for the checkpoint (a preempted parent resumes in a new job whose
  result dataset starts as a copy of the previous one). A child whose parent
  has no such checkpoint yet is skipped with a message. A running job's
  dataset is uncommitted and cannot be mounted, so when it is the only holder
  the checkpoint is copied into a new committed dataset
  `<child>-parent-ema_ckpt_0040` (reused if present).

Real submissions need the configs committed and pushed (gantry clones HEAD);
--dry-run skips that check and prints the commands without submitting.

Usage:
    python submit_fm_v2_followup_jobs.py lrsweep [--lr 1e-4 ...] [--dry-run] ...
    python submit_fm_v2_followup_jobs.py ext80 [--arch nc-sfno-v2 ...]
                                         [--arm ARM ...] [--seed N ...] ...
    python submit_fm_v2_followup_jobs.py e120 --lr X [--date YYYY-MM-DD] ...
  common: [--dry-run] [--skip-if-in-beaker] [--exclude-job NAME ...]
          [--beaker-workspace ...] [--beaker-cluster ...] [--beaker-priority ...]
"""

import argparse
import json
import pathlib
import re
import subprocess
import tempfile
from collections.abc import Iterable

from _beaker_listing import RUNNING, canonical_name, job_status, list_experiments
from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from create_anneal_jobs import read_header
from create_fm_v2_followup_jobs import (
    EXT80_ARCHS,
    LRSWEEP_LRS,
    PARENT_CHECKPOINT,
    PHASES,
    Run,
    e120_runs,
    ext80_runs,
    lrsweep_runs,
)
from create_fm_v2_jobs import ARMS, SEEDS
from submit_fm_jobs import (
    EXP_DIR,
    HERE,
    WANDB_PREFIX,
    WANDB_PROJECT,
    add_default_beaker_args,
)

TRAIN_SCRIPT = HERE / "run-ace-train.sh"
FORK_SCRIPT = HERE / "run-ace-anneal.sh"
N_GPUS = "4"
BUDGET = "ai2/atec-climate"  # as run-ace-anneal.sh
EMA_CKPT_RE = re.compile(r"training_checkpoints/ema_ckpt_(\d{4})\.tar")


def parent_jobs(parent: str, experiments: Iterable[dict]) -> list[dict]:
    """Every job of every experiment whose canonical name is `parent`: finished
    jobs first (their result datasets are complete), each group newest first.
    """
    jobs = [
        job
        for experiment in experiments
        if canonical_name(experiment["name"]) == parent
        for job in experiment.get("jobs") or []
    ]
    jobs.sort(key=lambda job: job.get("status", {}).get("created", ""), reverse=True)
    return sorted(jobs, key=lambda job: job_status(job) == RUNNING)


def ema_checkpoints(dataset: str) -> list[str]:
    """The ema_ckpt_XXXX.tar paths in a result dataset, oldest epoch first.

    A job that never started may have no listable dataset; that counts as none.
    """
    proc = subprocess.run(
        ["beaker", "dataset", "ls", dataset], capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(f"  (could not list dataset {dataset}: {proc.stderr.strip()})")
        return []
    return sorted(set(m.group(0) for m in EMA_CKPT_RE.finditer(proc.stdout)))


def find_checkpoint(
    parent: str, experiments: list[dict], checkpoint: str = PARENT_CHECKPOINT
) -> tuple[str, bool] | None:
    """(result dataset, dataset belongs to a running job) holding the parent's
    `checkpoint`, or None.
    """
    for job in parent_jobs(parent, experiments):
        dataset = job.get("result", {}).get("beaker")
        if dataset and checkpoint in ema_checkpoints(dataset):
            return dataset, job_status(job) == RUNNING
    return None


def _committed_dataset_id(full_name: str) -> str | None:
    """The id of dataset `full_name` if it exists and is committed, else None."""
    proc = subprocess.run(
        ["beaker", "dataset", "get", full_name, "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    found = json.loads(proc.stdout)
    found = found[0] if isinstance(found, list) else found
    return found["id"] if found.get("committed") else None


def committed_copy(
    dataset: str, checkpoint: str, child: str, workspace: str, dry_run: bool
) -> str:
    """A committed dataset holding only `checkpoint` (same path) copied from
    the uncommitted `dataset` of a running job, reusing an earlier copy.
    """
    name = f"{child}-parent-{pathlib.PurePath(checkpoint).stem}"
    whoami = subprocess.run(
        ["beaker", "account", "whoami", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    account = json.loads(whoami.stdout)[0]["name"]
    found = _committed_dataset_id(f"{account}/{name}")
    if found is not None:
        print(f"  reusing committed copy {name} ({found})")
        return found
    if dry_run:
        print(f"  would copy {checkpoint} from uncommitted {dataset} to {name}")
        return f"<copy of {dataset}>"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["beaker", "dataset", "fetch", dataset, "--prefix", checkpoint]
            + ["--output", tmp],
            check=True,
        )
        subprocess.run(
            ["beaker", "dataset", "create", tmp, "--name", name]
            + ["--workspace", workspace, "--budget", BUDGET]
            + ["--desc", f"{checkpoint} of {dataset}, for {child}"],
            check=True,
        )
    dataset_id = _committed_dataset_id(f"{account}/{name}")
    if dataset_id is None:
        raise RuntimeError(f"dataset {account}/{name} was not created and committed")
    print(f"  copied {checkpoint} from uncommitted {dataset} to {name} ({dataset_id})")
    return dataset_id


def list_texts(runs: list[Run]) -> list[str]:
    """Narrow Beaker --text filters, one per tag, covering every run."""
    return sorted({f"{WANDB_PREFIX}{run.tag}-fm-" for run in runs})


def submit_scratch(runs: list[Run], args: argparse.Namespace) -> None:
    texts = list_texts(runs)
    runs = drop_jobs_in_beaker(runs, lambda run: run.job_name, args, texts)
    if runs and not args.dry_run:
        check_configs_at_head([run.path for run in runs])
    for run in runs:
        header = read_header(run.path)
        assert header["JOB_NAME"] == run.job_name
        submit_job(
            TRAIN_SCRIPT,
            [
                str(run.path.relative_to(EXP_DIR)),
                run.job_name,
                header["WANDB_GROUP"],
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=EXP_DIR,
            extra_env={"N_GPUS": N_GPUS},
        )


def submit_forks(runs: list[Run], args: argparse.Namespace) -> None:
    texts = list_texts(runs)
    runs = drop_jobs_in_beaker(runs, lambda run: run.job_name, args, texts)
    print("Looking for parent checkpoints in Beaker...")
    experiments: list[dict] = []
    for arch in sorted({run.arch for run in runs}):
        # Parents only: the broad default "ace2-fm-" listing times out.
        experiments += list_experiments(
            args.beaker_workspace, text=f"{WANDB_PREFIX}{arch}-fm-"
        )
    ready = []
    for run in runs:
        header = read_header(run.path)
        assert header["JOB_NAME"] == run.job_name
        assert header["PARENT_JOB"] == run.parent_job
        found = find_checkpoint(run.parent_job, experiments)
        if found is None:
            print(
                f"Skipping (parent has no {PARENT_CHECKPOINT} yet): "
                f"{run.job_name} <- {run.parent_job}"
            )
            continue
        dataset, running = found
        print(f"Parent {run.parent_job}: {PARENT_CHECKPOINT} in dataset {dataset}")
        if running:
            dataset = committed_copy(
                dataset,
                PARENT_CHECKPOINT,
                run.job_name,
                args.beaker_workspace,
                args.dry_run,
            )
        ready.append((run, header, dataset))
    print(f"{len(ready)} to submit.")
    if ready and not args.dry_run:
        check_configs_at_head([run.path for run, _, _ in ready])
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
            extra_env={"N_GPUS": N_GPUS, "PARENT_CHECKPOINT": PARENT_CHECKPOINT},
        )
        if args.dry_run:
            print(f"  env: N_GPUS={N_GPUS} PARENT_CHECKPOINT={PARENT_CHECKPOINT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument(
        "--lr",
        nargs="+",
        help=(
            f"lrsweep: LRs to submit (default: all, {' '.join(LRSWEEP_LRS)}); "
            "e120: the chosen peak LR X (exactly one, required)."
        ),
    )
    parser.add_argument(
        "--date", help="e120 only: date in the wandb group (as the generator)."
    )
    parser.add_argument(
        "--arch", nargs="+", choices=EXT80_ARCHS, default=list(EXT80_ARCHS)
    )
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seed", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    add_default_beaker_args(parser)
    args = parser.parse_args()

    if args.phase == "lrsweep":
        lrs = args.lr or list(LRSWEEP_LRS)
        runs = [run for run in lrsweep_runs() if run.lr in lrs]
        if len(runs) != len(lrs):
            parser.error(f"--lr must be among {LRSWEEP_LRS}")
    elif args.phase == "e120":
        if not args.lr or len(args.lr) != 1:
            parser.error("e120 needs exactly one --lr")
        runs = e120_runs(args.lr[0], args.date)
    else:
        runs = ext80_runs(tuple(args.arch))
    if args.phase != "lrsweep":
        runs = [run for run in runs if run.arm in args.arm and run.seed in args.seed]
    missing = [run.path.name for run in runs if not run.path.exists()]
    if missing:
        raise SystemExit(
            "Configs not generated (create_fm_v2_followup_jobs.py):\n  "
            + "\n  ".join(missing)
        )
    if args.phase == "e120":
        groups = {read_header(run.path)["WANDB_GROUP"] for run in runs}
        if groups != {runs[0].group}:
            raise SystemExit(
                f"config wandb group(s) {sorted(groups)} differ from "
                f"{runs[0].group}; regenerate with the same --date"
            )
    if args.phase == "ext80":
        submit_forks(runs, args)
    else:
        submit_scratch(runs, args)


if __name__ == "__main__":
    main()
