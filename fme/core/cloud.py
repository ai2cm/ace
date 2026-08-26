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


class StagedFile:
    """A file streamed to local disk, copied to its destination when closed.

    Writers that stream into an open local file handle (e.g. a
    ``netCDF4.Dataset``) cannot target a remote store directly. This stages
    such a file: writes go to ``path``, and ``upload`` copies it to
    ``destination``. Call ``upload`` once the file handle is closed, so a
    complete file is what lands remotely.

    When ``destination`` is already local there is no staging at all:
    ``path`` is ``destination`` and ``upload`` does nothing, so local
    behavior is unchanged.
    """

    def __init__(self, destination: str | Path):
        self._destination = str(destination)
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        if is_local(self._destination):
            self._path = self._destination
        else:
            self._tmpdir = tempfile.TemporaryDirectory()
            self._path = os.path.join(
                self._tmpdir.name, os.path.basename(self._destination)
            )

    @property
    def path(self) -> str:
        """The local path to write to."""
        return self._path

    @property
    def destination(self) -> str:
        """The final path of the file, local or remote."""
        return self._destination

    def upload(self):
        """Copy the staged file to its destination and discard the staging copy.

        A no-op for a local destination, or if already called.
        """
        if self._tmpdir is not None:
            inter_filesystem_copy(self._path, self._destination)
            self._tmpdir.cleanup()
            self._tmpdir = None


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
