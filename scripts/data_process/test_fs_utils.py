import os

import fsspec
from fs_utils import is_dir, is_local, makedirs, path_exists


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
