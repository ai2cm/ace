"""Refresh wandb_to_beaker_map.json from the Beaker workspace listing.

Maps each 2026-10-19 training run (every config in run_configs/, any
architecture) whose Beaker experiment succeeded to the result dataset of its
latest exit-0 job, the dataset holding its checkpoints. A run that was
preempted and retried writes a new result dataset on the retry, so only exit-0
jobs are consulted (see _beaker_listing.succeeded_result_dataset).

Runs still running, failed or canceled are left out and enter the map on a
later refresh. submit_inference_jobs.py reads Beaker directly by default and
only uses this file with --use-map; the file is a record of which dataset each
run's inference jobs mounted.

Usage:
    python update_beaker_map.py [--dry-run] [--map PATH]
"""

import argparse
import json
import pathlib

from _beaker_listing import NamedExperiment, fetch_experiments_by_name
from _inference_common import BEAKER_MAP_PATH, all_training_run_names, finished_runs


def resolve_map(
    old_map: dict[str, str], experiments: dict[str, NamedExperiment] | None = None
) -> dict[str, str]:
    """`old_map` with every succeeded training run added or corrected."""
    ready, _ = finished_runs(all_training_run_names(), experiments)
    new_map = dict(old_map)
    for run_name, dataset_id in sorted(ready.items()):
        previous = old_map.get(run_name)
        if previous == dataset_id:
            continue
        verb = "add " if previous is None else "fix "
        print(f"  {verb}{run_name}: {previous} -> {dataset_id}")
        new_map[run_name] = dataset_id
    return dict(sorted(new_map.items()))


def refresh_map(
    map_path: pathlib.Path = BEAKER_MAP_PATH,
    experiments: dict[str, NamedExperiment] | None = None,
    dry_run: bool = False,
) -> bool:
    """Rewrite the map file if the listing changes it; True if it did."""
    old_map: dict[str, str] = {}
    if map_path.exists():
        old_map = json.loads(map_path.read_text())
    if experiments is None:
        experiments = fetch_experiments_by_name()
    new_map = resolve_map(old_map, experiments)
    if new_map == old_map:
        print("Map already up to date.")
        return False
    if dry_run:
        print("\n--dry-run: not writing.")
        return True
    map_path.write_text(json.dumps(new_map, indent=2) + "\n")
    print(f"\nWrote {map_path.name} ({len(new_map)} runs)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print changes without writing."
    )
    parser.add_argument(
        "--map",
        type=pathlib.Path,
        default=BEAKER_MAP_PATH,
        help=f"Map file to update (default: {BEAKER_MAP_PATH.name}).",
    )
    args = parser.parse_args()
    refresh_map(args.map, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
