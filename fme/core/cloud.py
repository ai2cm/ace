import shutil
from pathlib import Path

import fsspec


def inter_filesystem_copy(source: str, destination: str):
    """Copy between any two 'filesystems'. Do not use for large files.

    Args:
        source: Path to source file/object.
        destination: Path to destination.
    """
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)


def is_local(path: str | Path) -> bool:
    """Check if path is on a local filesystem assuming fsspec conventions."""
    fs, _ = fsspec.url_to_fs(path)
    return isinstance(fs, fsspec.implementations.local.LocalFileSystem)


def makedirs(path: str | Path, exist_ok: bool = False):
    """Create directories on any filesystem assuming fsspec conventions."""
    fs, _ = fsspec.url_to_fs(path)
    fs.makedirs(path, exist_ok=exist_ok)
