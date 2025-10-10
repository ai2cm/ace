from pathlib import Path


def is_local(path: str | Path) -> bool:
    """Check if a given path is on a local filesystem. Assuming fsspec conventions."""
    return "://" not in str(path)
