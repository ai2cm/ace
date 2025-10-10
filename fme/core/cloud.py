def is_local(path: str) -> bool:
    """Check if a given path is on a local filesystem. Assuming fsspec conventions."""
    return "://" not in path
