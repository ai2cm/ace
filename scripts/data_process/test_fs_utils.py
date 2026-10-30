import os

import fsspec
import numpy as np
import xarray as xr
from fs_utils import is_dir, is_local, load_netcdf, makedirs, path_exists
from fsspec.implementations.memory import MemoryFileSystem


class _StaleCacheFileSystem(MemoryFileSystem):
    """Memory filesystem that answers existence from a cache until invalidated.

    Stands in for object stores whose cached listings go stale when another
    instance writes, which is what the helpers guard against.
    """

    protocol = "stalecache"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._seen: dict[str, bool] = {}

    def exists(self, path, **kwargs):
        if path not in self._seen:
            self._seen[path] = super().exists(path, **kwargs)
        return self._seen[path]

    def invalidate_cache(self, path=None):
        self._seen.clear()


def test_path_helpers_local(tmp_path):
    subdir = tmp_path / "some" / "nested" / "dir"
    assert not path_exists(str(subdir))
    makedirs(str(subdir))
    assert path_exists(str(subdir))
    assert is_dir(str(subdir))
    makedirs(str(subdir))  # idempotent

    file_path = subdir / "file.txt"
    file_path.write_text("data")
    assert path_exists(str(file_path))
    assert not is_dir(str(file_path))
    assert is_local(str(file_path))


def test_path_helpers_resolve_filesystem_from_url_scheme():
    path = "memory://bucket/prefix/store"
    assert not is_local(path)
    assert not path_exists(path)
    makedirs(path)
    with fsspec.open(os.path.join(path, "obj.txt"), "wb") as f:
        f.write(b"data")
    assert path_exists(os.path.join(path, "obj.txt"))
    assert is_dir(path)


def test_path_exists_sees_write_made_after_an_earlier_miss():
    fsspec.register_implementation("stalecache", _StaleCacheFileSystem)
    store = "stalecache://bucket/prefix/store"

    assert not path_exists(store)  # caches the miss on the long-lived instance

    with fsspec.open(os.path.join(store, "obj.txt"), "wb") as f:
        f.write(b"data")

    assert path_exists(store)


def test_load_netcdf_from_local_and_remote_paths(tmp_path):
    ds = xr.Dataset(
        {f"var_{i}": ("lat", np.arange(3.0) + i, {"units": "K"}) for i in range(5)},
        coords={"lat": [-45.0, 0.0, 45.0]},
        attrs={"input_samples": 1460},
    )
    local_path = str(tmp_path / "centering.nc")
    ds.to_netcdf(local_path)
    remote_path = "memory://stats/centering.nc"
    fsspec.filesystem("memory").put_file(local_path, remote_path)

    for path in [local_path, remote_path]:
        loaded = load_netcdf(path)
        xr.testing.assert_identical(loaded, ds)
        assert loaded.encoding["source"] == path
