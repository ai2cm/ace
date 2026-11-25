"""Copy stage-2 fine-tunes' best_ckpt.tar into datasets at chosen epochs (see README).

best_ckpt.tar (lowest validation loss, EMA weights, evaluator-ready) is rewritten at
the end of every epoch whose validation loss improves. For each target epoch T, once a
job's log shows such a save at an epoch >= T, the file is fetched from the experiment's
result dataset, its epoch is read, and a dataset named
<date>-<experiment name>-epoch<E>-best-ckpt is created in ai2/ace holding
training_checkpoints/best_ckpt.tar. Result files reach Beaker with a lag of up to ~40
minutes, so a fetch that still holds an earlier epoch is retried at the next check.
Progress is kept in STATE_FILE, so the watcher can be restarted; it exits when every
target is copied or every experiment has finished.

Needs the beaker CLI (logged in) and torch.

  python snapshot_finetune_checkpoints.py [--epochs 20,30,36] [--workdir DIR]
      [--once] [--dry-run] EXPERIMENT ...
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import time

import torch

STATE_FILE = "snapshot_state.json"
LOG_WINDOW = "4h"
EPOCH_LINE = re.compile(r"Saving checkpoints for epoch (\d+)")
BEST_LINE = re.compile(r"Saving lowest validation loss checkpoint")


def _log(message):
    print(f"{datetime.datetime.now():%m-%d %H:%M} {message}", flush=True)


def _beaker(*args) -> str:
    return subprocess.run(
        ["beaker", *args], capture_output=True, text=True, check=True
    ).stdout


def _experiment(experiment_id):
    data = json.loads(_beaker("experiment", "get", experiment_id, "--format", "json"))
    experiment = data[0] if isinstance(data, list) else data
    job = experiment["jobs"][-1]
    return {
        "name": experiment["name"],
        "job": job["id"],
        "result": job.get("result", {}).get("beaker", ""),
        "finalized": bool(job["status"].get("finalized")),
    }


def best_saved_epochs(job_id) -> list[int]:
    """Epochs in the recent log whose end-of-epoch save included best_ckpt.tar."""
    logs = subprocess.run(
        ["beaker", "job", "logs", job_id, "--since", LOG_WINDOW],
        capture_output=True,
        text=True,
    ).stdout
    epochs, current = [], None
    for line in logs.splitlines():
        found = EPOCH_LINE.search(line)
        if found:
            current = int(found.group(1))
        elif current is not None and BEST_LINE.search(line):
            epochs.append(current)
            current = None
    return epochs


def fetch_best(result_id, directory) -> int | None:
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    subprocess.run(
        [
            "beaker",
            "dataset",
            "fetch",
            result_id,
            "--prefix",
            "training_checkpoints/best_ckpt.tar",
        ],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    path = os.path.join(directory, "training_checkpoints", "best_ckpt.tar")
    if not os.path.exists(path):
        return None
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def create_dataset(name, directory, description):
    _beaker(
        "dataset",
        "create",
        "--workspace",
        "ai2/ace",
        "--name",
        name,
        "--desc",
        description,
        directory,
    )


def check(experiment_id, targets, state, workdir, dry_run) -> bool:
    """Copy any target epoch now available; return whether to keep watching."""
    done = state.setdefault(experiment_id, {})
    pending = [t for t in targets if str(t) not in done]
    if not pending:
        return False
    info = _experiment(experiment_id)
    name = info["name"]
    saved = best_saved_epochs(info["job"])
    ready = info["finalized"] or any(e >= pending[0] for e in saved)
    _log(
        f"{name}: pending {pending}, recent best_ckpt epochs {saved[-3:]}"
        + (", finished" if info["finalized"] else "")
    )
    if not ready:
        return True
    if dry_run:
        _log(f"{name}: would fetch best_ckpt.tar for target {pending[0]} (dry run)")
        return not info["finalized"]
    directory = os.path.join(workdir, experiment_id)
    epoch = fetch_best(info["result"], directory)
    if epoch is None or (epoch < pending[0] and not info["finalized"]):
        _log(f"{name}: fetched epoch {epoch}, waiting for >= {pending[0]}")
        return True
    dataset = f"{datetime.date.today():%Y-%m-%d}-{name}-epoch{epoch}-best-ckpt"
    create_dataset(
        dataset,
        directory,
        f"best_ckpt.tar (EMA, lowest validation loss) of {name} after epoch {epoch}; "
        f"experiment {experiment_id}, 2026-11-23 randco2 retrain round",
    )
    for target in pending:
        if target <= epoch or info["finalized"]:
            done[str(target)] = {"epoch": epoch, "dataset": dataset}
    shutil.rmtree(directory, ignore_errors=True)
    _log(
        f"{name}: created dataset {dataset} (epoch {epoch}) for targets "
        f"{[t for t in pending if str(t) in done]}"
    )
    return any(str(t) not in done for t in targets) and not info["finalized"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument(
        "--epochs", default="20,30,36", help="comma-separated target epochs"
    )
    parser.add_argument("--workdir", default="finetune-snapshots")
    parser.add_argument(
        "--interval", type=int, default=600, help="seconds between checks"
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    state_path = os.path.join(args.workdir, STATE_FILE)
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    targets = sorted(int(e) for e in args.epochs.split(","))
    active = list(args.experiments)
    while active:
        still = []
        for experiment_id in active:
            try:
                if check(experiment_id, targets, state, args.workdir, args.dry_run):
                    still.append(experiment_id)
            except subprocess.CalledProcessError as error:
                _log(
                    f"{experiment_id}: beaker call failed "
                    f"({error.stderr.strip()[:200]}); retrying next check"
                )
                still.append(experiment_id)
            if not args.dry_run:
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=1)
        active = still
        if args.once or not active:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
