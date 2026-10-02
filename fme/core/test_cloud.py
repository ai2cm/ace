import os
from pathlib import Path

import pytest
import xarray as xr
import zarr

from fme.core import cloud
from fme.core.cloud import (
    exists,
    get_zarr_store,
    glob,
    inter_filesystem_copy,
    is_local,
    makedirs,
    open_dataset_via_inter_filesystem_copy,
    read_bytes,
    to_netcdf_via_inter_filesystem_copy,
    write_bytes,
)
from fme.core.testing import mock_object_store


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


def test_makedirs_remote_is_noop(tmp_path: Path):
    with mock_object_store(tmp_path / "remote"):
        makedirs("memory://bucket/a/b")
        makedirs("memory://bucket/a/b")
        assert not exists("memory://bucket/a/b")


def test_inter_filesystem_copy(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("test")
    destination = "memory://destination/destination.txt"

    with mock_object_store(tmp_path / "remote"):
        inter_filesystem_copy(source, destination)
        assert exists(destination)
        assert read_bytes(destination) == b"test"
        inter_filesystem_copy(destination, tmp_path / "roundtrip.txt")
    assert (tmp_path / "roundtrip.txt").read_text() == "test"


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
    with mock_object_store(tmp_path / "remote"):
        assert not exists(remote)
        inter_filesystem_copy(local, remote)
        assert exists(remote)


def test_exists_remote_directory(tmp_path: Path):
    with mock_object_store(tmp_path / "remote"):
        write_bytes("memory://bucket/dir/sub/f.txt", b"x")
        assert exists("memory://bucket/dir")
        assert exists("memory://bucket/dir/sub")
        assert not exists("memory://bucket/di")
        assert not exists("memory://bucket/other")


def test_open_dataset_via_inter_filesystem_copy(tmp_path: Path):
    # round-trip through a NON-LOCAL filesystem (memory://) -- the case plain
    # xr.open_dataset on a netCDF can't handle (e.g. restart.nc on gs://).
    ds = xr.Dataset(
        data_vars={"var": ("x", [1.0, 2.0, 3.0])},
        coords={"x": [1, 2, 3]},
    )
    filename = "memory://roundtrip-test/test.nc"
    with mock_object_store(tmp_path / "remote"):
        to_netcdf_via_inter_filesystem_copy(ds, filename)
        result = open_dataset_via_inter_filesystem_copy(filename)
    # values are present even though the temporary copy is gone (helper calls .load())
    xr.testing.assert_identical(ds, result)


def _write_glob_tree(base: str):
    for name in [
        "a.nc",
        "b.nc",
        "c.txt",
        "sub/d.nc",
        "data.zarr/zarr.json",
        "data.zarr/var/c/0",
        "other.zarr/zarr.json",
    ]:
        write_bytes(f"{base}/{name}", b"x")


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("*.nc", ["a.nc", "b.nc"]),
        ("?.nc", ["a.nc", "b.nc"]),
        ("*/*.nc", ["sub/d.nc"]),
        ("sub/*.nc", ["sub/d.nc"]),
        ("*.zarr", ["data.zarr", "other.zarr"]),
        ("data.zarr", ["data.zarr"]),
        ("missing.zarr", []),
        ("a.nc", ["a.nc"]),
    ],
)
def test_glob(tmp_path: Path, pattern: str, expected: list[str]):
    local_base = str(tmp_path / "local")
    remote_base = "memory://bucket/base"
    with mock_object_store(tmp_path / "remote"):
        for base in (local_base, remote_base):
            makedirs(os.path.join(base, "sub"), exist_ok=True)
            makedirs(os.path.join(base, "data.zarr", "var", "c"), exist_ok=True)
            makedirs(os.path.join(base, "other.zarr"), exist_ok=True)
            _write_glob_tree(base)
            assert glob(base, pattern) == [f"{base}/{name}" for name in expected]


def test_get_zarr_store_roundtrip(tmp_path: Path):
    path = "memory://bucket/experiment/data.zarr"
    with mock_object_store(tmp_path / "remote"):
        group = zarr.open_group(get_zarr_store(path, read_only=False), mode="w")
        group.create_array("var", shape=(3,), dtype="f4", dimension_names=["x"])[:] = [
            1.0,
            2.0,
            3.0,
        ]
        assert exists(path)
        # no parent group metadata is written outside the store's own path
        assert not exists("memory://bucket/experiment/zarr.json")
        ds = xr.open_zarr(get_zarr_store(path), consolidated=False)
        assert ds["var"].values.tolist() == [1.0, 2.0, 3.0]
        with pytest.raises(Exception):
            zarr.open_group(get_zarr_store(path), mode="w")


def test_get_zarr_store_local_returns_path(tmp_path: Path):
    assert get_zarr_store(tmp_path / "data.zarr") == str(tmp_path / "data.zarr")


def test_store_cache_is_per_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    with mock_object_store(tmp_path / "remote"):
        store = cloud._get_store("memory://bucket")
        assert cloud._get_store("memory://bucket") is store
        monkeypatch.setattr(os, "getpid", lambda: -1)
        assert cloud._get_store("memory://bucket") is not store
