from fme.core.distributed import Distributed


def test_gather_object_gathers_tuple():
    dist = Distributed.get_instance()
    gathered = dist.gather_object((dist.rank,))
    if dist.is_root():
        assert gathered is not None
        seen_values = set()
        for i in range(dist.world_size):
            item = gathered[i]
            assert isinstance(item, tuple)
            assert len(item) == 1
            seen_values.add(item[0])
        assert seen_values == set(range(dist.world_size))
    else:
        assert gathered is None
