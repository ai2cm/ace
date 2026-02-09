import dataclasses
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor

from .layers import Context, ContextConfig
from .sfnonet import FourierNeuralOperatorBlock, get_lat_lon_sfnonet
from .sht import InverseRealSHT, RealSHT
from .timer import CUDATimer

DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize(
    "conditional_embed_dim_scalar, conditional_embed_dim_labels, "
    "conditional_embed_dim_noise, "
    "conditional_embed_dim_pos, residual_filter_factor",
    [
        (0, 0, 0, 0, 1),
        (16, 8, 0, 0, 1),
        (16, 0, 16, 0, 1),
        (16, 15, 14, 13, 1),
        (0, 0, 0, 16, 1),
        (0, 0, 16, 0, 1),
        (16, 0, 0, 0, 4),
    ],
)
def test_can_call_sfnonet(
    conditional_embed_dim_scalar: int,
    conditional_embed_dim_labels: int,
    conditional_embed_dim_noise: int,
    conditional_embed_dim_pos: int,
    residual_filter_factor: int,
):
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = SimpleNamespace(
        embed_dim=16,
        num_layers=2,
        residual_filter_factor=residual_filter_factor,
        filter_type="makani-linear",
    )
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        context_config=ContextConfig(
            embed_dim_scalar=conditional_embed_dim_scalar,
            embed_dim_labels=conditional_embed_dim_labels,
            embed_dim_noise=conditional_embed_dim_noise,
            embed_dim_pos=conditional_embed_dim_pos,
        ),
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)
    context_embedding = torch.randn(
        n_samples, conditional_embed_dim_scalar, device=device
    )
    context_embedding_labels = torch.randn(
        n_samples, conditional_embed_dim_labels, device=device
    )
    context_embedding_noise = torch.randn(
        n_samples, conditional_embed_dim_noise, *img_shape, device=device
    )
    context_embedding_pos = torch.randn(
        n_samples, conditional_embed_dim_pos, *img_shape, device=device
    )
    context = Context(
        embedding_scalar=context_embedding,
        labels=context_embedding_labels,
        noise=context_embedding_noise,
        embedding_pos=context_embedding_pos,
    )
    output = model(x, context)
    assert output.shape == (n_samples, output_channels, *img_shape)


def test_scale_factor_not_implemented():
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    device = get_device()
    params = SimpleNamespace(embed_dim=16, num_layers=2, scale_factor=2)
    with pytest.raises(NotImplementedError):
        # if this ever gets implemented, we need to instead test that the scale factor
        # is used to determine the nlat/nlon of the image in the network
        get_lat_lon_sfnonet(
            params=params,
            img_shape=img_shape,
            in_chans=input_channels,
            out_chans=output_channels,
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_noise=0,
                embed_dim_labels=0,
                embed_dim_pos=0,
            ),
        ).to(device)


def test_sfnonet_output_is_unchanged():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    conditional_embed_dim_scalar = 8
    conditional_embed_dim_labels = 4
    conditional_embed_dim_noise = 16
    conditional_embed_dim_pos = 0
    device = get_device()
    params = SimpleNamespace(
        embed_dim=16, num_layers=2, filter_type="linear", operator_type="dhconv"
    )
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        context_config=ContextConfig(
            embed_dim_scalar=conditional_embed_dim_scalar,
            embed_dim_labels=conditional_embed_dim_labels,
            embed_dim_noise=conditional_embed_dim_noise,
            embed_dim_pos=conditional_embed_dim_pos,
        ),
    ).to(device)
    # must initialize on CPU to get the same results on GPU
    x = torch.randn(n_samples, input_channels, *img_shape).to(device)
    context_embedding = torch.randn(n_samples, conditional_embed_dim_scalar).to(device)
    context_embedding_labels = torch.randn(
        n_samples, conditional_embed_dim_labels, device=device
    )
    context_embedding_noise = torch.randn(
        n_samples, conditional_embed_dim_noise, *img_shape, device=device
    ).to(device)
    context_embedding_pos = None
    context = Context(
        embedding_scalar=context_embedding,
        labels=context_embedding_labels,
        noise=context_embedding_noise,
        embedding_pos=context_embedding_pos,
    )
    with torch.no_grad():
        output = model(x, context)
    validate_tensor(
        output,
        os.path.join(DIR, "testdata/test_sfnonet_output_is_unchanged.pt"),
    )


@pytest.mark.parametrize("normalize_big_skip", [True, False])
def test_all_inputs_get_layer_normed(normalize_big_skip: bool):
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    conditional_embed_dim_scalar = 8
    conditional_embed_dim_noise = 16
    conditional_embed_dim_labels = 3
    conditional_embed_dim_pos = 12
    device = get_device()

    class SetToZero(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return torch.zeros_like(x)

    original_layer_norm = nn.LayerNorm
    try:
        nn.LayerNorm = SetToZero
        params = SimpleNamespace(
            embed_dim=16,
            num_layers=2,
            normalize_big_skip=normalize_big_skip,
            global_layer_norm=True,  # so it uses nn.LayerNorm
            operator_type="dhconv",
        )
        model = get_lat_lon_sfnonet(
            params=params,
            img_shape=img_shape,
            in_chans=input_channels,
            out_chans=output_channels,
            context_config=ContextConfig(
                embed_dim_scalar=conditional_embed_dim_scalar,
                embed_dim_noise=conditional_embed_dim_noise,
                embed_dim_labels=conditional_embed_dim_labels,
                embed_dim_pos=conditional_embed_dim_pos,
            ),
        ).to(device)
    finally:
        nn.LayerNorm = original_layer_norm
    x = torch.full((n_samples, input_channels, *img_shape), torch.nan).to(device)
    context_embedding = torch.randn(n_samples, conditional_embed_dim_scalar).to(device)
    context_embedding_noise = torch.randn(
        n_samples, conditional_embed_dim_noise, *img_shape
    ).to(device)
    context_embedding_labels = torch.randn(n_samples, conditional_embed_dim_labels).to(
        device
    )
    context_embedding_pos = torch.randn(
        n_samples, conditional_embed_dim_pos, *img_shape
    ).to(device)
    context = Context(
        embedding_scalar=context_embedding,
        embedding_pos=context_embedding_pos,
        noise=context_embedding_noise,
        labels=context_embedding_labels,
    )
    with torch.no_grad():
        output = model(x, context)
    if normalize_big_skip:
        assert not torch.isnan(output).any()
    else:
        assert torch.isnan(output).any()


@dataclasses.dataclass
class BenchmarkResult:
    ms_total: float
    ms_per: float
    max_alloc: int
    max_reserved: int
    y_shape: tuple
    y_dtype: torch.dtype


def benchmark(fn, iters=10, warmup=1) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    for _ in range(iters):
        y = fn()
    ender.record()
    torch.cuda.synchronize()

    ms = starter.elapsed_time(ender)
    return BenchmarkResult(
        ms_total=ms,
        ms_per=ms / iters,
        max_alloc=torch.cuda.max_memory_allocated(),
        max_reserved=torch.cuda.max_memory_reserved(),
        y_shape=tuple(y.shape),
        y_dtype=y.dtype,
    )


@pytest.mark.skipif(
    get_device().type != "cuda",
    reason=(
        "This test is only relevant for CUDA since "
        "it's testing speed of SFNO blocks on GPU."
    ),
)  # noqa: E501
def test_block_speed():
    B = 2
    C = 512
    H = 180
    L = 360
    G = 8
    device = get_device()
    conditional_embed_dim_scalar = 0
    conditional_embed_dim_noise = 64
    conditional_embed_dim_labels = 3
    conditional_embed_dim_pos = 32
    embedding_scalar = None
    context_embedding_noise = torch.randn(B, conditional_embed_dim_noise, H, L).to(
        device
    )
    context_embedding_labels = torch.randn(B, conditional_embed_dim_labels).to(device)
    context_embedding_pos = torch.randn(B, conditional_embed_dim_pos, H, L).to(device)
    context = Context(
        embedding_scalar=embedding_scalar,
        embedding_pos=context_embedding_pos,
        noise=context_embedding_noise,
        labels=context_embedding_labels,
    )
    x = torch.randn(B, C, H, L, device=get_device())
    forward = RealSHT(nlat=H, nlon=L)
    inverse = InverseRealSHT(nlat=H, nlon=L)
    context_config = ContextConfig(
        embed_dim_scalar=conditional_embed_dim_scalar,
        embed_dim_noise=conditional_embed_dim_noise,
        embed_dim_labels=conditional_embed_dim_labels,
        embed_dim_pos=conditional_embed_dim_pos,
    )
    block = FourierNeuralOperatorBlock(
        forward_transform=forward,
        inverse_transform=inverse,
        embed_dim=C,
        img_shape=(H, L),
        filter_type="linear",
        operator_type="dhconv",
        use_mlp=True,
        context_config=context_config,
    ).to(device)
    timer = CUDATimer()
    grouped_block = FourierNeuralOperatorBlock(
        forward_transform=forward,
        inverse_transform=inverse,
        embed_dim=C,
        img_shape=(H, L),
        filter_type="linear",
        operator_type="dhconv",
        use_mlp=True,
        context_config=context_config,
        filter_num_groups=G,
    ).to(device)
    grouped_timer = CUDATimer()

    def call_block():
        return block(x, context, timer=timer)

    def call_grouped_block():
        return grouped_block(x, context, timer=grouped_timer)

    for _ in range(10):
        block(x, context)
    ungrouped = benchmark(call_block, warmup=0, iters=10)
    for _ in range(10):
        grouped_block(x, context)
    grouped = benchmark(call_grouped_block, warmup=0, iters=10)

    print("ungrouped timers: ", timer.report())
    print("grouped timers: ", grouped_timer.report())

    assert grouped.ms_per < 2 / G * ungrouped.ms_per, (
        "Expected grouped DHConv to be faster than ungrouped, but got "
        f"{grouped.ms_per:.6f} ms for grouped and "
        f"{ungrouped.ms_per:.6f} ms for ungrouped."
    )
    assert grouped.max_alloc < ungrouped.max_alloc, (
        "Expected grouped DHConv to use less memory than ungrouped, but got "
        f"{grouped.max_alloc/1024/1024:.2f} MB for grouped and "
        f"{ungrouped.max_alloc/1024/1024:.2f} MB for ungrouped."
    )
