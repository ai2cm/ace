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
    ``destination`` with the destination filesystem's ``put_file``. Call
    ``upload`` once the file handle is closed, so a complete file is what
    lands remotely.

    ``put_file`` rather than ``inter_filesystem_copy`` because staged netCDF
    files are multi-GB: gcsfs's ``put_file`` uploads in 50 MiB chunks and can
    verify the uploaded object's checksum, where a stream through
    ``fsspec.open`` uses a ~5 MiB write block and checks nothing, so silent
    corruption would go undetected. The checksum request is gcsfs-specific;
    other fsspec backends ignore it (s3fs drops unrecognized kwargs) and
    upload with whatever integrity check they do by default.

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

    def upload(self):
        """Copy the staged file to its destination and discard the staging copy.

        A no-op for a local destination, or if already called. Errors from
        the upload propagate, and a failed upload leaves the staged file in
        place, so a caller that catches the error can call ``upload`` again to
        retry. The staging directory is still a temporary directory: it is
        removed when the process exits, so an uncaught failure loses the file.
        """
        if self._tmpdir is not None:
            fs, _ = fsspec.url_to_fs(self._destination)
            # The integrity check is the main reason to use put_file over a
            # plain stream, and gcsfs leaves it off by default
            # (consistency="none" is a no-op checker), so ask for it
            # explicitly. md5 rather than crc32c because crc32c needs the
            # crcmod package, which is not a dependency. Hashing a multi-GB
            # file costs CPU, which is cheap next to the transfer itself and
            # is the point for output we cannot re-derive.
            fs.put_file(self._path, self._destination, consistency="md5")
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
