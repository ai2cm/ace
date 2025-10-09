import fsspec


def is_local(path: str) -> bool:
    """Check if a given path is on a local filesystem."""
    return fsspec.url_to_fs(path)[0] == fsspec.filesystem("file")
