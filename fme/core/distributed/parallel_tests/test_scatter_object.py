import pytest

from fme.core.distributed import Distributed


@pytest.mark.parallel
def test_scatter_object_scatters_from_global_root():
    dist = Distributed.get_instance()
    if dist.is_root():
        result = dist.scatter_object(["a", dist.rank])
    else:
        result = dist.scatter_object(None)
    assert result == ["a", 0]
