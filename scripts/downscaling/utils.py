import re
import subprocess
from pathlib import Path

# Matching for <event_name>_YYYYMMDD*.nc
_EVENT_FILE_RE = re.compile(r"(.+)_(\d{8}).*\.nc$")


def fetch_beaker_dataset(
    dataset_id: str, target_dir: str, prefix: str | None = None
) -> None:
    """Fetch a beaker dataset to the specified directory.

    Args:
        dataset_id: The beaker dataset ID to fetch.
        target_dir: The directory to download the dataset into.
        prefix: If provided, only fetch files matching this prefix.
    """
    cmd = ["beaker", "dataset", "fetch", dataset_id, "--output", target_dir]
    if prefix is not None:
        cmd += ["--prefix", prefix]
    subprocess.run(cmd, check=True)


def find_event_files(directory: str) -> dict[str, Path]:
    """Find netCDF files matching the event naming pattern, keyed by event name."""
    event_files = {}
    for p in sorted(Path(directory).glob("*.nc")):
        # extract event name
        matched = _EVENT_FILE_RE.match(p.name)
        if matched:
            event_files[matched.group(1)] = p
    return event_files
