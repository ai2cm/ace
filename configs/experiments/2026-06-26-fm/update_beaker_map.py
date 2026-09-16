"""Refresh wandb_to_beaker_map.json from the Beaker workspace listing.

Resolution path for each run:
  beaker experiment name (gantry job name, hash suffix stripped)
    -> latest job with exitCode 0  -> its result dataset ID

A stale map happens when a training job is preempted and retried: the first
job's result dataset has no pre_cooldown_ckpt.tar, while the succeeded retry
writes a *new* result dataset. The map must point at the succeeded job, which
is why only exit-0 jobs are consulted.

Only runs whose result dataset something downstream mounts enter the map:
training and fine-tuning runs (their checkpoints), and the best-inference SST
sweep (its prediction files). Evaluator runs and the fine-tune epoch sweep are
excluded by name: nothing reads their datasets, and the epoch sweep alone is
thousands of names.

Usage:
    python update_beaker_map.py [--dry-run] [--map PATH]
"""

import argparse
import json
import pathlib
import re

from _beaker_listing import OK, fetch_experiments_by_name

HERE = pathlib.Path(__file__).parent
DEFAULT_MAP = HERE / "wandb_to_beaker_map.json"

# Runs whose result datasets nothing mounts. Evaluator runs reuse a training
# run's name with a checkpoint suffix; fixed-variable and orography evals carry
# their own prefix; the fine-tune epoch sweep ends in an epoch segment.
SKIP_SUFFIXES = ("-bestinf", "-besttrain", "-lastepoch")
SKIP_PREFIXES = ("ace2-fm-fixed-", "ace2-fm-orog-")
SKIP_PATTERNS = (re.compile(r"-sst-(era5|c96)-p\dk-e\d\d$"),)


def is_mapped_run(run_name: str) -> bool:
    if run_name.endswith(SKIP_SUFFIXES):
        return False
    if run_name.startswith(SKIP_PREFIXES):
        return False
    return not any(pattern.search(run_name) for pattern in SKIP_PATTERNS)


def resolve_map(old_map: dict[str, str]) -> dict[str, str]:
    """`old_map` with every succeeded, mapped run added or corrected."""
    new_map = dict(old_map)
    for run_name, named in sorted(fetch_experiments_by_name().items()):
        if not is_mapped_run(run_name):
            continue
        if named.status != OK:
            continue
        dataset_id = named.result_dataset
        if dataset_id is None:
            print(f"  skip {run_name}: succeeded job has no result dataset")
            continue
        previous = old_map.get(run_name)
        if previous == dataset_id:
            continue
        verb = "add " if previous is None else "fix "
        print(f"  {verb}{run_name}: {previous} -> {dataset_id}")
        new_map[run_name] = dataset_id
    return new_map


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
    args = parser.parse_args()

    old_map: dict[str, str] = {}
    if args.map.exists():
        old_map = json.loads(args.map.read_text())

    new_map = resolve_map(old_map)

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
