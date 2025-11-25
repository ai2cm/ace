from importlib.util import find_spec


def _deps_available() -> bool:
    # Both packages must be importable
    return find_spec("trimesh") is not None and find_spec("rtree") is not None


GRAPHCAST_AVAIL = _deps_available()
