import torch

from fme.core import get_device
from fme.core.distributed import Distributed


def test_gather_tensor_from_local_slices():
    dist = Distributed.get_instance()
    global_shape = (4, 4)
    x_global = (
        torch.arange(global_shape[0] * global_shape[1], device=get_device()).reshape(
            global_shape
        )
        + 1
    )
    x_local = x_global[dist.get_local_slices(global_shape, dist.rank)]
    gathered = dist.gather_global(x_local, global_shape=global_shape)
    if dist.is_root():
        assert gathered is not None
        torch.testing.assert_close(gathered, x_global)
    else:
        assert gathered is None
