import os
import shutil
import tempfile
from pathlib import Path

import fsspec
import xarray as xr


def inter_filesystem_copy(source: str | Path, destination: str | Path):
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


def exists(path: str | Path) -> bool:
    """Check whether a path exists on any filesystem assuming fsspec conventions."""
    fs, _ = fsspec.url_to_fs(path)
    return fs.exists(path)


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
