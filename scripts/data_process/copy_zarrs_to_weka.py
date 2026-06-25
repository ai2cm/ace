#!/usr/bin/env python3
"""Find .zarr stores under gs:// paths and copy each to Weka via gcs_to_weka.sh.

For each input path, if it ends in .zarr it is copied directly; otherwise it is
searched recursively for .zarr stores. Each store is copied to
/climate-default, mirroring the source path (minus the bucket name), by
submitting a Beaker/Gantry job through gcs_to_weka.sh.
"""

import argparse
import subprocess
import sys
from pathlib import Path

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


def weka_dest(zarr: str) -> str:
    """Map gs://bucket/rest/foo.zarr to /climate-default/rest/foo.zarr."""
    rest = zarr.split("/", 3)[3]  # drop "gs:", "", bucket
    return f"{WEKA_ROOT}/{rest}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gs_paths", nargs="+", help="gs:// paths to search")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print paths and destinations but do not submit jobs",
    )
    args = parser.parse_args()

    for path in args.gs_paths:
        if not path.startswith("gs://"):
            sys.exit(f"Error: path must start with gs://: {path}")
        print(f"Scanning {path} ...")
        zarrs = find_zarrs(path)
        if not zarrs:
            print("  no .zarr stores found")
            continue
        for zarr in zarrs:
            dest = weka_dest(zarr)
            print(f"  {zarr} -> {dest}")
            if not args.dry_run:
                subprocess.run([str(GCS_TO_WEKA), zarr, dest], check=True)


if __name__ == "__main__":
    main()
