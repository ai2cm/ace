from collections.abc import Callable

import pytest
import torch
import torch_harmonics.distributed as thd

from fme.core import get_device
from fme.core.compile import compile_backend, compile_module
from fme.core.distributed import Distributed
from fme.core.sht import (
    DistributedDiscreteContinuousConvS2,
    DistributedInverseRealSHT,
    DistributedRealSHT,
)

NLAT, NLON = 8, 16
BATCH_CHANNEL_SHAPE = (2, 3)

# (lmax, mmax): defaults, and SFNO-style truncation (hard_thresholding_fraction=0.5)
MODES = pytest.mark.parametrize(
    "lmax, mmax",
    [
        pytest.param(None, None, id="default_modes"),
        pytest.param(NLAT // 2, (NLON // 2 + 1) // 2, id="truncated_modes"),
    ],
)


def _get_local_block(x: torch.Tensor) -> torch.Tensor:
    """This rank's (polar, azimuth) block of ``x``."""
    polar_shapes = thd.compute_split_shapes(x.shape[-2], thd.polar_group_size())
    azimuth_shapes = thd.compute_split_shapes(x.shape[-1], thd.azimuth_group_size())
    polar_start = sum(polar_shapes[: thd.polar_group_rank()])
    azimuth_start = sum(azimuth_shapes[: thd.azimuth_group_rank()])
    return x[
        ...,
        polar_start : polar_start + polar_shapes[thd.polar_group_rank()],
        azimuth_start : azimuth_start + azimuth_shapes[thd.azimuth_group_rank()],
    ]


def _random_local_input(
    spatial_shape: tuple[int, int], dtype: torch.dtype
) -> Callable[[], torch.Tensor]:
    """Factory of this rank's block of a random global input."""

    def make_input() -> torch.Tensor:
        x = torch.randn(
            *BATCH_CHANNEL_SHAPE, *spatial_shape, dtype=dtype, device=get_device()
        )
        return _get_local_block(x)

    return make_input


def _sht_input(sht: torch.nn.Module) -> Callable[[], torch.Tensor]:
    return _random_local_input((sht.nlat, sht.nlon), torch.float32)


def _isht_input(isht: torch.nn.Module) -> Callable[[], torch.Tensor]:
    return _random_local_input((isht.lmax, isht.mmax), torch.complex64)


def _assert_compiled_matches_eager(
    module: torch.nn.Module, make_input: Callable[[], torch.Tensor]
):
    """Two compiled calls match eager. Without the clone, the second call
    fails on torch 2.7-2.9 when H > 1."""
    torch.manual_seed(0)
    torch._dynamo.reset()
    compiled = torch.compile(module, backend="aot_eager")
    try:
        for _ in range(2):
            x = make_input()
            torch.testing.assert_close(compiled(x), module(x))
    finally:
        torch._dynamo.reset()


def _assert_matches_upstream(
    module: torch.nn.Module,
    upstream: torch.nn.Module,
    make_input: Callable[[], torch.Tensor],
):
    """``module`` and ``upstream`` give identical eager outputs."""
    torch.manual_seed(0)
    x = make_input()
    torch.testing.assert_close(module(x), upstream(x), rtol=0, atol=0)


@pytest.mark.parallel
@MODES
def test_compiled_sht_matches_eager(lmax, mmax):
    """Compiled SHT matches eager; patched class under spatial parallelism."""
    dist = Distributed.get_instance()
    sht = dist.get_sht(NLAT, NLON, lmax=lmax, mmax=mmax)
    if dist.has_spatial_parallelism:
        assert isinstance(sht, DistributedRealSHT)
    _assert_compiled_matches_eager(sht.to(get_device()), _sht_input(sht))


@pytest.mark.parallel
@MODES
def test_compiled_isht_matches_eager(lmax, mmax):
    """Compiled ISHT matches eager; patched class under spatial parallelism."""
    dist = Distributed.get_instance()
    isht = dist.get_isht(NLAT, NLON, lmax=lmax, mmax=mmax)
    if dist.has_spatial_parallelism:
        assert isinstance(isht, DistributedInverseRealSHT)
    _assert_compiled_matches_eager(isht.to(get_device()), _isht_input(isht))


@pytest.mark.parallel
@MODES
def test_sht_matches_upstream(lmax, mmax):
    """Matches upstream torch-harmonics."""
    sht = DistributedRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax).to(get_device())
    upstream = thd.DistributedRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax).to(get_device())
    _assert_matches_upstream(sht, upstream, _sht_input(sht))


@pytest.mark.parallel
@MODES
def test_isht_matches_upstream(lmax, mmax):
    """Matches upstream torch-harmonics."""
    isht = DistributedInverseRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax).to(get_device())
    upstream = thd.DistributedInverseRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax).to(
        get_device()
    )
    _assert_matches_upstream(isht, upstream, _isht_input(isht))


@pytest.mark.parallel
@MODES
@pytest.mark.parametrize("inverse", [False, True], ids=["sht", "isht"])
def test_transform_compiles_without_graph_break(lmax, mmax, inverse):
    """Forward and transpose backward compile to one graph."""
    transform: torch.nn.Module
    if inverse:
        transform = DistributedInverseRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax)
        make_input = _isht_input(transform)
    else:
        transform = DistributedRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax)
        make_input = _sht_input(transform)
    transform = transform.to(get_device())
    # an input requiring grad makes dynamo also trace the transposes' backward
    x = make_input().requires_grad_(True)
    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(transform)(x)
    finally:
        torch._dynamo.reset()
    assert explanation.graph_break_count == 0, explanation.break_reasons


# (in_shape, out_shape): same grid, an odd latitude count that splits unevenly
# across polar ranks, and a 2x downsampling as SFNO's DISCO encoders use.
DISCO_SHAPES = pytest.mark.parametrize(
    "in_shape, out_shape",
    [
        pytest.param((NLAT, NLON), (NLAT, NLON), id="same_grid"),
        pytest.param((9, 18), (9, 18), id="odd_latitudes"),
        pytest.param((NLAT, NLON), (NLAT // 2, NLON // 2), id="downsample"),
    ],
)


def _disco_conv_forward_backward(
    conv: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Output and input/weight/bias grads of a squared-sum loss."""
    x = x.clone().requires_grad_(True)
    conv.zero_grad()
    out = conv(x)
    out.square().sum().backward()
    assert x.grad is not None
    return out.detach(), x.grad, conv.weight.grad.clone(), conv.bias.grad.clone()


@pytest.mark.parallel
@DISCO_SHAPES
def test_disco_conv_matches_upstream(in_shape, out_shape):
    """Matches upstream torch-harmonics in forward and backward."""
    torch.manual_seed(0)
    kwargs = dict(in_shape=in_shape, out_shape=out_shape, kernel_shape=(3,))
    conv = DistributedDiscreteContinuousConvS2(3, 2, **kwargs).to(get_device())
    upstream = thd.DistributedDiscreteContinuousConvS2(3, 2, **kwargs).to(get_device())
    upstream.load_state_dict(conv.state_dict())
    torch.nn.init.normal_(conv.bias)
    upstream.bias.data.copy_(conv.bias.data)
    x = _random_local_input(in_shape, torch.float32)()
    # the weight and bias gradients are local: upstream only reduces them
    # across ranks through the data-parallel wrapper
    for actual, expected in zip(
        _disco_conv_forward_backward(conv, x),
        _disco_conv_forward_backward(upstream, x),
    ):
        # the squared-sum loss accumulates float32 rounding in a different
        # summation order than upstream's sparse contraction
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parallel
@DISCO_SHAPES
def test_compiled_disco_conv_matches_eager(in_shape, out_shape):
    """``Distributed.get_disco_conv_s2`` compiles to one graph and matches eager
    in forward and backward.

    On odd latitudes the backward gathers uneven shards, which neither dynamo
    nor gloo accepts unless they are padded to equal size.
    """
    dist = Distributed.get_instance()
    torch.manual_seed(0)
    conv = dist.get_disco_conv_s2(
        3, 2, in_shape=in_shape, out_shape=out_shape, kernel_shape=(3,)
    ).to(get_device())
    x = _random_local_input(in_shape, torch.float32)()
    eager = _disco_conv_forward_backward(conv, x)
    with compile_backend("aot_eager"):
        compiled_conv = compile_module(conv, fullgraph=True)
    compiled = _disco_conv_forward_backward(compiled_conv, x)
    for actual, expected in zip(compiled, eager):
        torch.testing.assert_close(actual, expected)
