"""Filesystem helpers shared by the data processing scripts.

The filesystem is resolved from the path's URL scheme via fsspec, so callers
use a single code path for local paths and remote URLs (e.g. ``gs://``).
"""

import fsspec
from fsspec.implementations.local import LocalFileSystem


def _fs_and_path(path: str) -> tuple[fsspec.AbstractFileSystem, str]:
    return fsspec.core.url_to_fs(path)


def path_exists(path: str) -> bool:
    """Return whether a file, directory, or object-store prefix exists."""
    fs, fs_path = _fs_and_path(path)
    return fs.exists(fs_path)


def is_dir(path: str) -> bool:
    """Return whether the path is a directory (or object-store prefix)."""
    fs, fs_path = _fs_and_path(path)
    return fs.isdir(fs_path)


def makedirs(path: str) -> None:
    """Create a directory if it doesn't exist; a no-op on object stores."""
    fs, fs_path = _fs_and_path(path)
    fs.makedirs(fs_path, exist_ok=True)


def is_local(path: str) -> bool:
    """Return whether the path resolves to the local filesystem.

    isinstance is used because fsspec dispatches filesystem behavior by
    class, and local-vs-remote is exactly that distinction.
    """
    fs, _ = _fs_and_path(path)
    return isinstance(fs, LocalFileSystem)
