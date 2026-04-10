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
    """Find netCDF event outputs under *directory*, including nested paths.

    * Filenames like ``<event_name>_YYYYMMDD*.nc`` are keyed by *event_name*
      (same behavior as before).
    * Other ``*.nc`` files — e.g. plain ``{event_name}.nc`` written by
      :class:`fme.downscaling.evaluator.EventEvaluator` when
      ``save_generated_samples`` is true — are keyed by the path stem relative
      to *directory*, with ``/`` replaced by ``__`` so Beaker's nested layouts
      still produce unique keys.

    Excludes ``evaluator_maps_and_metrics.nc`` (aggregate metrics, not events).
    """
    root = Path(directory).resolve()
    event_files: dict[str, Path] = {}
    for p in sorted(root.rglob("*.nc")):
        print(p.name)
        matched = _EVENT_FILE_RE.match(p.name)
        if matched:
            key = matched.group(1)
        else:
            key = p.relative_to(root).with_suffix("").as_posix().replace("/", "__")
        event_files[key] = p
    return event_files
