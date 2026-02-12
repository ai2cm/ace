import torch

from fme.core.distributed import Distributed


def test_gather_tensor_from_local_slices():
    dist = Distributed.get_instance()
    global_shape = (4, 4)
    x_global = torch.arange(global_shape[0] * global_shape[1]).reshape(global_shape) + 1
    x_local = x_global[dist.get_local_slices(global_shape, dist.rank)]
    gathered = dist.gather(x_local)
    if dist.is_root():
        if gathered is None:
            raise RuntimeError("expected non-none gathered on root rank")
        gathered_global = torch.zeros_like(x_global)
        for i, local in enumerate(gathered):
            gathered_global[dist.get_local_slices(global_shape, i)] = local
        torch.testing.assert_close(gathered_global, x_global)
