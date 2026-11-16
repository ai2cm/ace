import pytest
import torch

from fme.core import get_device
from fme.core.compile import compile_backend, compile_module
from fme.core.distributed import Distributed


@pytest.mark.parallel
def test_compiled_disco_conv_matches_eager_on_odd_latitudes():
    """A compiled DISCO convolution matches eager in forward and backward when
    an odd number of latitudes splits unevenly across spatial ranks.

    Under the model backend the convolution's backward gathers the uneven
    latitude shards, which neither dynamo (tracing) nor gloo (eager) accepts
    unless the shards are padded to equal size.
    """
    dist = Distributed.get_instance()
    img_shape = (9, 18)  # 9 latitudes split unevenly over 2 ranks
    torch.manual_seed(0)
    conv = dist.get_disco_conv_s2(
        2, 3, in_shape=img_shape, out_shape=img_shape, kernel_shape=(3,)
    ).to(get_device())
    global_shape = (2, 2, *img_shape)
    x_global = torch.randn(global_shape, device=get_device())
    x_local = x_global[dist.get_local_slices(global_shape)]

    def forward_backward(module):
        x = x_local.clone().requires_grad_(True)
        conv.zero_grad()
        out = module(x)
        out.square().sum().backward()
        assert x.grad is not None
        return out.detach(), x.grad, conv.weight.grad.clone()

    eager = forward_backward(conv)
    with compile_backend("aot_eager"):
        compiled_conv = compile_module(conv)
    compiled = forward_backward(compiled_conv)
    for expected, actual in zip(eager, compiled):
        torch.testing.assert_close(actual, expected)
