from pathlib import Path

from .cloud import is_local


def test_is_local():
    assert is_local("/absolute/path/somefile")
    assert is_local("relative/path/somefile")
    assert is_local(Path("/absolute/path/somefile"))
    assert not is_local("gs://mybucket/somefile")
