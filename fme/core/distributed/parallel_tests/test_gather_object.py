import pytest

from fme.core.distributed import Distributed


@pytest.mark.parallel
def test_gather_object_gathers_tuple():
    dist = Distributed.get_instance()
    n_ranks = dist.world_size
    gathered = dist.gather_object((dist.rank,))
    if dist.is_root():
        assert gathered is not None
        seen_values = set()
        for i in range(n_ranks):
            item = gathered[i]
            assert isinstance(item, tuple)
            assert len(item) == 1
            seen_values.add(item[0])
        assert seen_values == set(range(n_ranks))
    else:
        assert gathered is None
