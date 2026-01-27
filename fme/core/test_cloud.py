import os
from pathlib import Path

import fsspec
import pytest
import xarray as xr

from fme.core.cloud import (
    inter_filesystem_copy,
    is_local,
    makedirs,
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
