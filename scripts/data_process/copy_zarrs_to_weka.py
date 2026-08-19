#!/usr/bin/env python3
"""Find .zarr stores under gs:// paths and copy each to Weka via gcs_to_weka.sh.

For each input path, if it ends in .zarr it is copied directly; otherwise it is
searched recursively for .zarr stores. Each store is copied to
/climate-default, mirroring the source path (minus the bucket name), by
submitting a Beaker/Gantry job through gcs_to_weka.sh.

Alternatively, pass --mapping mapping.yaml to copy the dataset/stats paths
listed under ``dataset_pairs``. Use --dataset-only or --stats-only to restrict
the copy to one side of each pair.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

WEKA_ROOT = "/climate-default"
GCS_TO_WEKA = Path(__file__).resolve().parent / "gcs_to_weka.sh"


def find_zarrs(path: str) -> list[str]:
    """Return the .zarr store roots under a gs:// path."""
    path = path.rstrip("/")
    if path.endswith(".zarr"):
        return [path]
    result = subprocess.run(
        ["gsutil", "ls", "-r", f"{path}/**"],
        capture_output=True,
        text=True,
    )
    zarrs = set()
    for line in result.stdout.splitlines():
        marker = line.find(".zarr/")
        if marker != -1:
            zarrs.add(line[: marker + len(".zarr")])
    return sorted(zarrs)


def weka_dest(path: str) -> str:
    """Map gs://bucket/rest to /climate-default/rest."""
    rest = path.rstrip("/").split("/", 3)[3]  # drop "gs:", "", bucket
    return f"{WEKA_ROOT}/{rest}"


def submit(src: str, dry_run: bool) -> None:
    """Print and (unless dry_run) submit a gcs_to_weka copy job for src."""
    dest = weka_dest(src)
    print(f"  {src} -> {dest}")
    if not dry_run:
        subprocess.run([str(GCS_TO_WEKA), src, dest], check=True)


def paths_from_mapping(
    mapping_file: str, dataset_only: bool, stats_only: bool
) -> list[str]:
    """Return the dataset/stats gs:// paths listed in a mapping.yaml."""
    with open(mapping_file) as f:
        mapping = yaml.safe_load(f)
    keys = ["dataset", "stats"]
    if dataset_only:
        keys = ["dataset"]
    elif stats_only:
        keys = ["stats"]
    paths = []
    for pair in mapping["dataset_pairs"]:
        for key in keys:
            paths.append(pair[key])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gs_paths", nargs="*", help="gs:// paths to search (omit when using --mapping)"
    )
    parser.add_argument(
        "--mapping",
        help="mapping.yaml with dataset_pairs to copy instead of gs_paths",
    )
    only = parser.add_mutually_exclusive_group()
    only.add_argument(
        "--dataset-only",
        action="store_true",
        help="with --mapping, copy only the dataset paths",
    )
    only.add_argument(
        "--stats-only",
        action="store_true",
        help="with --mapping, copy only the stats paths",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print paths and destinations but do not submit jobs",
    )
    args = parser.parse_args()

    if args.mapping:
        if args.gs_paths:
            sys.exit("Error: pass either gs_paths or --mapping, not both")
        # Mirror each dataset/stats directory wholesale: stats dirs hold .nc
        # files, not .zarr stores, so scanning for zarrs would skip them.
        paths = paths_from_mapping(args.mapping, args.dataset_only, args.stats_only)
        for path in paths:
            if not path.startswith("gs://"):
                sys.exit(f"Error: path must start with gs://: {path}")
            print(f"Copying {path} ...")
            submit(path, args.dry_run)
        return

    if args.dataset_only or args.stats_only:
        sys.exit("Error: --dataset-only/--stats-only require --mapping")
    if not args.gs_paths:
        sys.exit("Error: provide gs_paths or --mapping")

    for path in args.gs_paths:
        if not path.startswith("gs://"):
            sys.exit(f"Error: path must start with gs://: {path}")
        print(f"Scanning {path} ...")
        zarrs = find_zarrs(path)
        if not zarrs:
            print("  no .zarr stores found")
            continue
        for zarr in zarrs:
            submit(zarr, args.dry_run)


if __name__ == "__main__":
    main()
