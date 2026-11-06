"""Submit the 2026-11-06 FM sfno-v3 runs (create_fm_v3_jobs.py configs).

Each v3 run forks an nc-sfno-v2 parent run at its epoch-40 EMA checkpoint. For
each selected child this script:

1. skips the child if its job name already has a succeeded or running
   experiment in Beaker (always on, as with --skip-if-in-beaker);
2. finds the parent's Beaker experiment(s) by canonical job name and looks
   through the result datasets of all their jobs, newest first, for
   training_checkpoints/ema_ckpt_0040.tar (a preempted parent resumes in a new
   job whose result dataset starts as a copy of the previous one, so any job
   from epoch 40 on holds it; datasets of finished jobs are preferred over
   that of a running job); a child whose parent has no such dataset yet is
   skipped with a message. A running job's dataset is uncommitted and cannot
   be mounted, so when it is the only holder the checkpoint is copied into a
   new committed dataset `<child>-parent-ema_ckpt_0040` (reused if present);
3. submits run-ace-anneal.sh with N_GPUS=4 (as the parents) and that checkpoint
   mounted at /parent_ckpt.tar (stepper_training.parameter_init.weights_path),
   into wandb group WANDB_GROUP.

--smoke submits the run_configs/v3-smoke/ config instead (job name ending in
-smoke). The smoke test checks the mechanics (parameter init, sampled 1/2/4
steps under CUDA graphs, LR schedule, EMA checkpoints, inline inference), not
the fork point, so when its parent has no ema_ckpt_0040.tar yet it mounts the
parent's latest EMA checkpoint instead.

Real submissions need the configs committed and pushed (gantry clones HEAD);
--dry-run skips that check and prints the commands without submitting.

Usage:
    python submit_fm_v3_jobs.py [--arm {fm-balanced,fm-e3sm-balanced} ...]
                                [--seed N ...] [--smoke] [--dry-run]
                                [--exclude-job NAME ...]
                                [--beaker-workspace ...] [--beaker-cluster ...]
                                [--beaker-priority ...]
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
from create_fm_v3_jobs import (
    ARMS,
    PARENT_CHECKPOINT,
    SEEDS,
    SMOKE_ARM,
    SMOKE_SEED,
    run_config_path,
)
from submit_fm_jobs import HERE, WANDB_PROJECT, add_default_beaker_args

RUN_SCRIPT = HERE / "run-ace-anneal.sh"
WANDB_GROUP = "ace2-fm-2026-11-06"
N_GPUS = "4"
BUDGET = "ai2/atec-climate"  # as run-ace-anneal.sh
PARENT_LIST_TEXT = "ace2-fm-nc-sfno-v2-fm-"
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
    parent: str, experiments: list[dict], accept_latest: bool
) -> tuple[str, str, bool] | None:
    """(result dataset, checkpoint path, dataset belongs to a running job)
    holding the parent's PARENT_CHECKPOINT, or, with `accept_latest`, its
    latest EMA checkpoint when that is missing.
    """
    latest: tuple[str, str, bool] | None = None
    for job in parent_jobs(parent, experiments):
        dataset = job.get("result", {}).get("beaker")
        if not dataset:
            continue
        running = job_status(job) == RUNNING
        checkpoints = ema_checkpoints(dataset)
        if PARENT_CHECKPOINT in checkpoints:
            return dataset, PARENT_CHECKPOINT, running
        if checkpoints and (latest is None or checkpoints[-1] > latest[1]):
            latest = dataset, checkpoints[-1], running
    return latest if accept_latest else None


def committed_copy(
    dataset: str, checkpoint: str, child: str, workspace: str, dry_run: bool
) -> str:
    """A committed dataset holding only `checkpoint` (same path) copied from
    `dataset`, reusing one made by an earlier call.

    A running job's result dataset is uncommitted and Beaker refuses to mount
    it, so a parent whose only copy of the checkpoint is in its running job's
    dataset is forked from this copy instead.
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
            [
                "beaker",
                "dataset",
                "fetch",
                dataset,
                "--prefix",
                checkpoint,
                "--output",
                tmp,
            ],
            check=True,
        )
        subprocess.run(
            [
                "beaker",
                "dataset",
                "create",
                tmp,
                "--name",
                name,
                "--workspace",
                workspace,
                "--budget",
                BUDGET,
                "--desc",
                f"{checkpoint} of {dataset}, for {child}",
            ],
            check=True,
        )
    # `dataset create` prints upload progress, not JSON; look the dataset up.
    dataset_id = _committed_dataset_id(f"{account}/{name}")
    if dataset_id is None:
        raise RuntimeError(f"dataset {account}/{name} was not created and committed")
    print(f"  copied {checkpoint} from uncommitted {dataset} to {name} ({dataset_id})")
    return dataset_id


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seed", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Submit the smoke config ({SMOKE_ARM} s{SMOKE_SEED}) instead.",
    )
    add_default_beaker_args(parser)
    args = parser.parse_args()
    args.skip_if_in_beaker = True

    if args.smoke:
        pairs = [(SMOKE_ARM, SMOKE_SEED)]
    else:
        pairs = [(arm, seed) for arm in args.arm for seed in args.seed]
    jobs = [
        (path, read_header(path))
        for path in (run_config_path(arm, seed, args.smoke) for arm, seed in pairs)
    ]
    jobs = drop_jobs_in_beaker(jobs, lambda job: job[1]["JOB_NAME"], args)

    print("Looking for parent checkpoints in Beaker...")
    # Parents only: the broad default "ace2-fm-" listing matches ~9k
    # experiments and times out; the parents share this prefix.
    experiments = list_experiments(args.beaker_workspace, text=PARENT_LIST_TEXT)
    ready = []
    for path, header in jobs:
        parent = header["PARENT_JOB"]
        found = find_checkpoint(parent, experiments, accept_latest=args.smoke)
        if found is None:
            print(
                f"Skipping (parent has no {PARENT_CHECKPOINT} yet): "
                f"{header['JOB_NAME']} <- {parent}"
            )
            continue
        dataset, checkpoint, running = found
        print(f"Parent {parent}: {checkpoint} in dataset {dataset}")
        if running:
            dataset = committed_copy(
                dataset,
                checkpoint,
                header["JOB_NAME"],
                args.beaker_workspace,
                args.dry_run,
            )
        ready.append((path, header, dataset, checkpoint))
    print(f"{len(ready)} to submit.")

    if ready and not args.dry_run:
        check_configs_at_head([path for path, *_ in ready])
    for path, header, dataset, checkpoint in ready:
        description = header["DESCRIPTION"]
        if checkpoint != PARENT_CHECKPOINT:
            description += f" (smoke: initialised from {checkpoint})"
        submit_job(
            RUN_SCRIPT,
            [
                str(path.relative_to(HERE)),
                header["JOB_NAME"],
                WANDB_GROUP,
                dataset,
                description,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=HERE,
            extra_env={"N_GPUS": N_GPUS, "PARENT_CHECKPOINT": checkpoint},
        )
        if args.dry_run:
            print(f"  env: N_GPUS={N_GPUS} PARENT_CHECKPOINT={checkpoint}")


if __name__ == "__main__":
    main()
