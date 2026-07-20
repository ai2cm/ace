"""Refresh wandb_to_beaker_map.json from wandb run notes.

Resolution path for each run:
  wandb run name
    -> run.notes  (a https://beaker.org/ex/<experiment_id> link)
    -> beaker experiment  (may contain several retried jobs)
    -> latest job with exitCode 0  -> its result dataset ID

A stale map happens when a training job is preempted and retried: the first
job's result dataset has no pre_cooldown_ckpt.tar, while the succeeded retry
writes a *new* result dataset. The map must point at the succeeded job.

Usage:
    python update_beaker_map.py [--dry-run] [--map PATH] [--version {v1,v2,v3}]
"""

import argparse
import json
import pathlib
import re
import subprocess

from generate_masking_configs import (
    BASE_CONFIG_FILENAMES,
    WANDB_ENTITY,
    WANDB_PROJECT,
    stem_has_version,
)

HERE = pathlib.Path(__file__).parent
DEFAULT_MAP = HERE / "wandb_to_beaker_map.json"

EXPERIMENT_RE = re.compile(r"beaker\.org/ex/([0-9A-Za-z]+)")

# Eval/export runs reuse a training run's name with one of these suffixes; they
# are not training runs and must not enter the map.
SKIP_SUFFIXES = ("-bestinf", "-besttrain", "-lastepoch")


def _experiment_id_from_notes(notes: str | None) -> str | None:
    if not notes:
        return None
    match = EXPERIMENT_RE.search(notes)
    return match.group(1) if match else None


def _succeeded_dataset_id(experiment_id: str) -> str | None:
    """Return the result dataset of the latest exitCode==0 job, else None."""
    proc = subprocess.run(
        ["beaker", "experiment", "get", experiment_id, "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    experiment = json.loads(proc.stdout)[0]
    succeeded = [
        job
        for job in experiment.get("jobs", [])
        if job.get("status", {}).get("exitCode") == 0
    ]
    if not succeeded:
        return None
    # Latest by start time wins if a job somehow succeeded more than once.
    succeeded.sort(key=lambda j: j.get("status", {}).get("started", ""))
    return succeeded[-1].get("result", {}).get("beaker")


def _fetch_run_notes() -> dict[str, str | None]:
    import wandb  # lazy import: keeps the module importable without wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    return {run.name: run.notes for run in runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved map without writing it.",
    )
    parser.add_argument(
        "--map",
        type=pathlib.Path,
        default=DEFAULT_MAP,
        help=f"Map file to update (default: {DEFAULT_MAP}).",
    )
    parser.add_argument(
        "--version",
        "-v",
        choices=sorted(BASE_CONFIG_FILENAMES),
        default=None,
        help="Restrict to runs of this baseline version (default: all).",
    )
    args = parser.parse_args()

    old_map: dict[str, str] = {}
    if args.map.exists():
        old_map = json.loads(args.map.read_text())

    run_notes = _fetch_run_notes()

    new_map = dict(old_map)
    for run_name, notes in sorted(run_notes.items()):
        if run_name.endswith(SKIP_SUFFIXES):
            continue
        if args.version is not None and not stem_has_version(run_name, args.version):
            continue
        experiment_id = _experiment_id_from_notes(notes)
        if experiment_id is None:
            print(f"  skip {run_name}: no beaker link in notes")
            continue
        dataset_id = _succeeded_dataset_id(experiment_id)
        if dataset_id is None:
            print(f"  skip {run_name}: no succeeded job / experiment unavailable")
            continue
        previous = old_map.get(run_name)
        if previous == dataset_id:
            continue
        verb = "add " if previous is None else "fix "
        print(f"  {verb}{run_name}: {previous} -> {dataset_id}")
        new_map[run_name] = dataset_id

    if new_map == old_map:
        print("Map already up to date.")
        return

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return

    args.map.write_text(json.dumps(new_map, indent=2) + "\n")
    print(f"\nWrote {args.map.name}")


if __name__ == "__main__":
    main()
