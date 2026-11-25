"""Stop stage-2 fine-tunes whose inline inference error has plateaued (see README).

A fine-tune is stopped with ``beaker experiment stop`` once its per-epoch
"Inference error", the metric that selects best_inference_ckpt.tar, has gone
PATIENCE epochs without a new best. Every job of an experiment is read, so the
best carries over a preemption and resume. The stopped experiment's result
dataset keeps best_inference_ckpt.tar.

  python stop_plateaued_finetunes.py [--patience 5] [--dry-run] [--once] EXPERIMENT ...
"""

import argparse
import datetime
import json
import re
import subprocess
import time


def _beaker_json(*args):
    out = subprocess.run(
        ["beaker", *args, "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    return data[0] if isinstance(data, list) else data


def epoch_errors(experiment) -> dict[int, float]:
    errors = {}
    for job in experiment["jobs"]:
        logs = subprocess.run(
            ["beaker", "job", "logs", job["id"]], capture_output=True, text=True
        ).stdout
        pending = None
        for line in logs.splitlines():
            found = re.search(r"Inference error: (\S+)", line)
            if found:
                pending = float(found.group(1))
            saved = re.search(r"Saving checkpoints for epoch (\d+)", line)
            if saved and pending is not None:
                errors[int(saved.group(1))] = pending
                pending = None
    return dict(sorted(errors.items()))


def best_epoch(errors: dict[int, float]) -> int:
    best, best_error = min(errors), float("inf")
    for epoch, error in errors.items():
        if error < best_error:
            best, best_error = epoch, error
    return best


def _log(message):
    print(f"{datetime.datetime.now():%m-%d %H:%M} {message}", flush=True)


def check(experiment_id: str, patience: int, dry_run: bool) -> bool:
    """Report one experiment and stop it if it has plateaued; return whether to keep
    watching it.
    """
    experiment = _beaker_json("experiment", "get", experiment_id)
    name = experiment.get("name", experiment_id)
    running = not experiment["jobs"][-1]["status"].get("finalized")
    errors = epoch_errors(experiment)
    if not errors:
        _log(f"{name}: no completed epochs yet")
        return running
    best, last = best_epoch(errors), max(errors)
    _log(
        f"{name}: epoch {last}, best {errors[best]:.5f} at epoch {best}, "
        f"{last - best} epochs without improvement"
    )
    if not running:
        _log(f"{name}: finished")
        return False
    if last - best < patience:
        return True
    if dry_run:
        _log(f"{name}: would stop (dry run)")
    else:
        subprocess.run(["beaker", "experiment", "stop", experiment_id], check=True)
        _log(f"{name}: STOPPED; best_inference_ckpt.tar is from epoch {best}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--interval", type=int, default=600, help="seconds between checks"
    )
    args = parser.parse_args()
    active = list(args.experiments)
    while active:
        active = [e for e in active if check(e, args.patience, args.dry_run)]
        if args.once or not active:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
