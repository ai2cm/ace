import os
from pathlib import Path

import fsspec
import pytest
import xarray as xr

from fme.core.cloud import (
    exists,
    inter_filesystem_copy,
    is_local,
    makedirs,
    open_dataset_via_inter_filesystem_copy,
    to_netcdf_via_inter_filesystem_copy,
)


@pytest.mark.parametrize(
    ("path, expected"),
    [
        ("/absolute/path/somefile", True),
        ("relative/path/somefile", True),
        ("file://path/somefile", True),
        ("local://path/somefile", True),
        pytest.param(Path("/absolute/path/somefile"), True, id="Path object"),
        ("https://path/somefile", False),
    ],
)
def test_is_local(path: str | Path, expected: bool):
    assert is_local(path) == expected


@pytest.mark.parametrize("use_str_input", [True, False])
def test_makedirs(tmp_path: Path, use_str_input: bool):
    path = tmp_path / "test" / "makedirs"
    assert not path.exists()
    input_path = str(path) if use_str_input else path

    makedirs(input_path)
    assert path.exists()
    assert path.is_dir()

    makedirs(input_path, exist_ok=True)
    assert path.exists()
    assert path.is_dir()

    with pytest.raises(FileExistsError):
        makedirs(input_path)


def test_inter_filesystem_copy(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("test")
    destination = "memory://destination/destination.txt"

    inter_filesystem_copy(source, destination)

    fs, _ = fsspec.url_to_fs(destination)
    assert fs.exists(destination)
    assert fs.read_text(destination) == "test"

    fs.rm(destination, recursive=True)


def test_to_netcdf_via_inter_filesystem_copy(tmp_path: Path):
    ds = xr.Dataset(
        data_vars={"var": ("x", [1, 2, 3])},
        coords={"x": [1, 2, 3]},
    )
    filename = os.path.join(tmp_path, "test.nc")
    to_netcdf_via_inter_filesystem_copy(ds, filename)
    result = xr.open_dataset(filename)
    xr.testing.assert_identical(ds, result)


def test_exists(tmp_path: Path):
    local = tmp_path / "f.txt"
    assert not exists(local)
    local.write_text("x")
    assert exists(local)

    remote = "memory://exists-test/f.txt"
    fs, _ = fsspec.url_to_fs(remote)
    assert not exists(remote)
    inter_filesystem_copy(local, remote)
    assert exists(remote)
    fs.rm(remote, recursive=True)


def test_open_dataset_via_inter_filesystem_copy():
    # round-trip through a NON-LOCAL filesystem (memory://) -- the case plain
    # xr.open_dataset on a netCDF can't handle (e.g. restart.nc on gs://).
    ds = xr.Dataset(
        data_vars={"var": ("x", [1.0, 2.0, 3.0])},
        coords={"x": [1, 2, 3]},
    )
    filename = "memory://roundtrip-test/test.nc"
    to_netcdf_via_inter_filesystem_copy(ds, filename)
    result = open_dataset_via_inter_filesystem_copy(filename)
    # values are present even though the temporary copy is gone (helper calls .load())
    xr.testing.assert_identical(ds, result)

    fs, _ = fsspec.url_to_fs(filename)
    fs.rm(filename, recursive=True)
