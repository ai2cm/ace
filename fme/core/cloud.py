import fnmatch
import functools
import glob as _glob
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import obstore
import xarray as xr
from obstore.store import ObjectStore as ObstoreStore
from obstore.store import from_url
from zarr.storage import ObjectStore

_LOCAL_SCHEMES = ("", "file", "local")


def _split(path: str | Path) -> tuple[str, str, str]:
    """Split a path into (scheme, bucket, key)."""
    parsed = urlparse(str(path))
    return parsed.scheme, parsed.netloc, parsed.path.strip("/")


def _local_path(path: str | Path) -> str:
    scheme, netloc, _ = _split(path)
    if scheme in ("file", "local"):
        # e.g. file:///abs/path or file://relative/path
        return netloc + urlparse(str(path)).path
    return str(path)


def _store_from_url(url: str) -> ObstoreStore:
    """Construct an obstore store; credentials are read from the environment."""
    return from_url(url)


@functools.lru_cache
def _get_store_cached(url: str, pid: int) -> ObstoreStore:
    return _store_from_url(url)


def _get_store(url: str) -> ObstoreStore:
    # Keyed on pid so that forked processes (e.g. DataLoader workers) never
    # reuse a store whose async runtime was created in the parent.
    return _get_store_cached(url, os.getpid())


def _bucket_store_and_key(path: str | Path) -> tuple[ObstoreStore, str]:
    scheme, bucket, key = _split(path)
    return _get_store(f"{scheme}://{bucket}"), key


def is_local(path: str | Path) -> bool:
    """Check if path is on a local filesystem."""
    return _split(path)[0] in _LOCAL_SCHEMES


def makedirs(path: str | Path, exist_ok: bool = False):
    """Create directories on a local filesystem. A no-op for object stores,
    which have no directories.
    """
    if is_local(path):
        os.makedirs(_local_path(path), exist_ok=exist_ok)


def exists(path: str | Path) -> bool:
    """Check whether a file or directory exists, locally or in an object store.

    In an object store, a "directory" (e.g. a zarr store) exists if any object
    exists under its prefix.
    """
    if is_local(path):
        return os.path.exists(_local_path(path))
    store, key = _bucket_store_and_key(path)
    try:
        obstore.head(store, key)
        return True
    except FileNotFoundError:
        pass
    for batch in obstore.list(store, prefix=key + "/", chunk_size=1):
        if len(batch) > 0:
            return True
    return False


def read_bytes(path: str | Path) -> bytes:
    """Read the full contents of a file, locally or from an object store."""
    if is_local(path):
        with open(_local_path(path), "rb") as f:
            return f.read()
    store, key = _bucket_store_and_key(path)
    return obstore.get(store, key).bytes().to_bytes()


def write_bytes(path: str | Path, data: bytes):
    """Write data to a file, locally or to an object store."""
    if is_local(path):
        with open(_local_path(path), "wb") as f:
            f.write(data)
    else:
        store, key = _bucket_store_and_key(path)
        obstore.put(store, key, data)


def glob(path: str | Path, pattern: str) -> list[str]:
    """Find files or directories under ``path`` matching a glob ``pattern``.

    For object stores, the pattern is matched one ``/``-separated segment at a
    time, so that directories (e.g. ``*.zarr`` stores) can be matched without
    listing every object beneath them.

    Returns:
        Sorted matching paths, with the same scheme and bucket as ``path``.
    """
    if is_local(path):
        return sorted(_glob.glob(os.path.join(_local_path(path), pattern)))
    scheme, bucket, key = _split(path)
    store = _get_store(f"{scheme}://{bucket}")
    candidates = [key]
    for segment in pattern.strip("/").split("/"):
        next_candidates = []
        for prefix in candidates:
            if not any(char in segment for char in "*?["):
                candidate = f"{prefix}/{segment}".strip("/")
                if exists(f"{scheme}://{bucket}/{candidate}"):
                    next_candidates.append(candidate)
                continue
            listing = obstore.list_with_delimiter(store, prefix=prefix or None)
            names = list(listing["common_prefixes"]) + [
                obj["path"] for obj in listing["objects"]
            ]
            next_candidates.extend(
                name
                for name in names
                if fnmatch.fnmatchcase(name.rsplit("/", 1)[-1], segment)
            )
        candidates = next_candidates
    return sorted(f"{scheme}://{bucket}/{candidate}" for candidate in candidates)


def get_zarr_store(path: str | Path, read_only: bool = True) -> str | ObjectStore:
    """Get a zarr store for a path.

    Local paths are returned unchanged (zarr opens them with its LocalStore).
    Remote paths are opened with obstore, rooted at the store's own prefix.
    """
    if is_local(path):
        return _local_path(path)
    return ObjectStore(_get_store(str(path).rstrip("/")), read_only=read_only)


def inter_filesystem_copy(source: str | Path, destination: str | Path):
    """Copy between any two 'filesystems'. Do not use for large files.

    Args:
        source: Path to source file/object.
        destination: Path to destination.
    """
    write_bytes(destination, read_bytes(source))


def to_netcdf_via_inter_filesystem_copy(ds: xr.Dataset, filename: str | Path):
    """Write an xarray dataset to a netCDF file via an inter-filesystem copy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = os.path.join(tmpdir, "temp.nc")
        ds.to_netcdf(source)
        inter_filesystem_copy(source, filename)


def open_dataset_via_inter_filesystem_copy(
    filename: str | Path, **kwargs
) -> xr.Dataset:
    """Open a netCDF dataset from any filesystem via a local temp copy.

    Counterpart of ``to_netcdf_via_inter_filesystem_copy``. Eagerly loaded
    (``.load()``) so values survive temp-dir cleanup. Small files only (same
    caveat as ``inter_filesystem_copy``); e.g. restart ICs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local = os.path.join(tmpdir, "temp.nc")
        inter_filesystem_copy(filename, local)
        return xr.open_dataset(local, **kwargs).load()
