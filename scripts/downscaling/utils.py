import re
import subprocess
from pathlib import Path

# Matching for <event_name>_YYYYMMDD*.nc
_EVENT_FILE_RE = re.compile(r"(.+)_(\d{8}).*\.nc$")


def fetch_beaker_dataset(
    dataset_id: str,
    target_dir: str,
    prefix: str | None = None,
    cache_dir: str | None = "~/Downloads/beaker_cache",
) -> str:
    """Fetch a beaker dataset to the specified directory.

    Args:
        dataset_id: The beaker dataset ID to fetch.
        target_dir: The directory to download the dataset into.
        prefix: If provided, only fetch files matching this prefix.
        cache_dir: If provided, datasets are stored under
            ``<cache_dir>/<dataset_id>/`` and reused on subsequent calls.
            When a cached copy exists the download is skipped and files
            are served from the cache. *target_dir* is ignored when a
            cache hit occurs.

    Returns:
        The directory containing the fetched dataset files.
    """
    if cache_dir is not None:
        cached = Path(cache_dir).expanduser() / dataset_id
        if cached.is_dir() and any(cached.iterdir()):
            return str(cached)
        target_dir = str(cached)

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    cmd = ["beaker", "dataset", "fetch", dataset_id, "--output", target_dir]
    if prefix is not None:
        cmd += ["--prefix", prefix]
    subprocess.run(cmd, check=True)
    return target_dir


def find_event_files(directory: str) -> dict[str, Path]:
    """Find netCDF files matching the event naming pattern, keyed by event name."""
    event_files = {}
    for p in sorted(Path(directory).glob("*.nc")):
        # extract event name
        matched = _EVENT_FILE_RE.match(p.name)
        if matched:
            event_files[matched.group(1)] = p
    return event_files
