import os
import re
import subprocess
from pathlib import Path

# Matching for <event_name>_YYYYMMDD*.nc
_EVENT_FILE_RE = re.compile(r"(.+)_(\d{8}).*\.nc$")
_FETCH_COMPLETE_MARKER = ".beaker_fetch_complete"
_BEAKER_CACHE_ENV_VAR = "BEAKER_DATASET_CACHE_DIR"


def fetch_beaker_dataset(
    dataset_id: str,
    target_dir: str,
    prefix: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Fetch a beaker dataset to the specified directory.

    Args:
        dataset_id: The beaker dataset ID to fetch.
        target_dir: The directory to download the dataset into.
        prefix: If provided, only fetch files matching this prefix.
        cache_dir: If provided, datasets are stored under
            ``<cache_dir>/<dataset_id>/`` and reused on subsequent calls.
            A sentinel file is written after a successful fetch; partial
            downloads from failed fetches are not treated as cache hits.
            *target_dir* is ignored when a cache hit occurs. Caching is
            opt-in: when *cache_dir* is ``None``, the environment variable
            ``BEAKER_DATASET_CACHE_DIR`` is used if set; otherwise files are
            fetched to *target_dir* with no caching.

    Returns:
        The directory containing the fetched dataset files.
    """
    if cache_dir is None:
        cache_dir = os.environ.get(_BEAKER_CACHE_ENV_VAR)
    if not cache_dir:
        cache_dir = None
    if cache_dir is not None:
        cached = Path(cache_dir).expanduser() / dataset_id
        if (cached / _FETCH_COMPLETE_MARKER).is_file():
            print(f"Using cached dataset at {cached}")
            return str(cached)
        target_dir = str(cached)
        print(f"Downloading dataset to cache at {target_dir}")
    else:
        print(f"Downloading dataset to {target_dir}")

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    cmd = ["beaker", "dataset", "fetch", dataset_id, "--output", target_dir]
    if prefix is not None:
        cmd += ["--prefix", prefix]
    subprocess.run(cmd, check=True)
    if cache_dir is not None:
        (Path(target_dir) / _FETCH_COMPLETE_MARKER).touch()
    return target_dir


def find_event_files(directory: str) -> dict[str, Path]:
    """Find netCDF files matching the event naming pattern, keyed by filename
    stem (event name including date, so multiple dates of the same event are
    kept)."""
    event_files = {}
    for p in sorted(Path(directory).glob("*.nc")):
        if _EVENT_FILE_RE.match(p.name):
            event_files[p.stem] = p
    return event_files
