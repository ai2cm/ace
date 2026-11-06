"""Filesystem helpers shared by the data processing scripts.

The filesystem is resolved from the path's URL scheme via fsspec, so callers
use a single code path for local paths and remote URLs (e.g. ``gs://``).
"""

import os
import tempfile

import fsspec
import xarray as xr
from fsspec.implementations.local import LocalFileSystem


def _fs_and_path(path: str) -> tuple[fsspec.AbstractFileSystem, str]:
    """Resolve a path to its filesystem, dropping any cached listing for it.

    ``url_to_fs`` returns a cached, long-lived instance whose directory
    listing cache is not invalidated by writes made through other instances
    (e.g. zarr writes via a separate asynchronous instance). Callers use these
    helpers to decide whether to overwrite an existing store, so a stale
    negative would silently clobber data rather than raise.
    """
    fs, fs_path = fsspec.core.url_to_fs(path)
    fs.invalidate_cache(fs_path)
    return fs, fs_path


def path_exists(path: str) -> bool:
    """Return whether a file, directory, or object-store prefix exists."""
    fs, fs_path = _fs_and_path(path)
    return fs.exists(fs_path)


def is_dir(path: str) -> bool:
    """Return whether the path is a directory (or object-store prefix)."""
    fs, fs_path = _fs_and_path(path)
    return fs.isdir(fs_path)


def makedirs(path: str) -> None:
    """Create a directory if it doesn't exist.

    Object stores have implicit directories, so this does nothing for a
    prefix within an existing bucket. It is not unconditionally a no-op on
    them: gcsfs attempts to create a bucket that doesn't exist, which raises
    if the caller lacks permission. Existing data is never overwritten.
    """
    fs, fs_path = _fs_and_path(path)
    fs.makedirs(fs_path, exist_ok=True)


def is_local(path: str) -> bool:
    """Return whether the path resolves to the local filesystem.

    isinstance is used because fsspec dispatches filesystem behavior by
    class, and local-vs-remote is exactly that distinction.
    """
    fs, _ = _fs_and_path(path)
    return isinstance(fs, LocalFileSystem)


def load_netcdf(path: str) -> xr.Dataset:
    """Load a small netCDF file, e.g. a stats file, fully into memory.

    A remote file is first downloaded to a temporary local file in a single
    read. Parsing a netCDF4 (HDF5) file directly from a remote file object
    instead issues a range request for each of its many small, scattered
    metadata reads, which for a file with hundreds of variables on GCS is
    tens of times slower. Reading from a local file rather than an in-memory
    buffer also lets the default netCDF4 engine be used, so h5netcdf is not
    required.

    The returned dataset's encoding["source"] is set to path.
    """
    fs, fs_path = _fs_and_path(path)
    if isinstance(fs, LocalFileSystem):
        ds = _load_local_netcdf(fs_path)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(fs_path))
            fs.get_file(fs_path, local_path)
            ds = _load_local_netcdf(local_path)
    ds.encoding["source"] = path
    return ds


def _load_local_netcdf(path: str) -> xr.Dataset:
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        return ds.load()
