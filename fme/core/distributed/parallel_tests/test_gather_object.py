import pytest

from fme.core.distributed import Distributed


@pytest.mark.parallel
def test_gather_object_gathers_tuple():
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    gathered = dist.gather_object((dist.data_parallel_rank,))
    if dist.is_root():
        assert gathered is not None
        seen_values = set()
        for i in range(n_dp):
            item = gathered[i]
            assert isinstance(item, tuple)
            assert len(item) == 1
            seen_values.add(item[0])
        assert seen_values == set(range(n_dp))
    else:
        assert gathered is None
