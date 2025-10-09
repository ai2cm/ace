from .cloud import is_local


def test_is_local():
    assert is_local("/tmp/somefile")
    assert is_local("relative/path/somefile")
    assert not is_local("s3://mybucket/somefile")
    assert not is_local("gs://mybucket/somefile")
