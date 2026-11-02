import pytest
import torch

from fme.core import get_device
from fme.core.distributed import Distributed
from fme.core.sht.test_distributed_compile import _get_local_block


class _RoundTrip(torch.nn.Module):
    def __init__(self, sht, isht):
        super().__init__()
        self.sht = sht
        self.isht = isht
        self.w = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.isht(self.sht(x * self.w))


@pytest.mark.parallel
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("shape", [(8, 16), (16, 32), (9, 18)])
@pytest.mark.parametrize("dynamic", [None, True])
def test_stress(backend, shape, dynamic):
    torch.manual_seed(0)
    nlat, nlon = shape
    dist = Distributed.get_instance()
    m = _RoundTrip(dist.get_sht(nlat, nlon), dist.get_isht(nlat, nlon)).to(
        get_device()
    )
    torch._dynamo.reset()
    compiled = torch.compile(m, backend=backend, dynamic=dynamic)
    try:
        for i in range(5):
            batch = 2 + (i % 2)  # vary batch to trigger recompiles / dynamic
            x = _get_local_block(torch.randn(batch, 4, nlat, nlon, device=get_device()))
            x.requires_grad_(True)
            y = compiled(x)
            y.square().sum().backward()
            g_c, w_c = x.grad.clone(), m.w.grad.clone()
            x.grad = None
            m.w.grad = None
            y_e = m(x)
            y_e.square().sum().backward()
            torch.testing.assert_close(y, y_e)
            torch.testing.assert_close(x.grad, g_c)
            torch.testing.assert_close(m.w.grad, w_c)
            m.w.grad = None
    finally:
        torch._dynamo.reset()
