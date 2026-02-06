import dataclasses
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor

from .layers import Context, ContextConfig
from .s2convolutions import _contract_dhconv
from .sfnonet import get_lat_lon_sfnonet

DIR = os.path.abspath(os.path.dirname(__file__))


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
        "it's testing speed of DHConv groups on GPU."
    ),
)  # noqa: E501
def test_contract_dhconv_groups_are_faster():
    B = 2
    C = 512
    H = 180
    L = 360
    G = 8
    x = torch.randn(B, 1, C, H, L, dtype=torch.complex64, device=get_device())
    w = torch.randn(1, C, C, H, 2, dtype=torch.float32, device=get_device())

    def contract_ungrouped():
        return _contract_dhconv(x, w)

    ungrouped_result = benchmark(contract_ungrouped)

    x_grouped = x.reshape(B, G, C // G, H, L)
    w_grouped = torch.randn(
        G, C // G, C // G, H, 2, dtype=torch.float32, device=get_device()
    )

    def contract_grouped():
        return _contract_dhconv(x_grouped, w_grouped)

    grouped_result = benchmark(contract_grouped)

    assert grouped_result.ms_per < 2 / G * ungrouped_result.ms_per, (
        "Expected grouped DHConv to be faster than ungrouped, but got "
        f"{grouped_result.ms_per:.6f} seconds for grouped and "
        f"{ungrouped_result.ms_per:.6f} seconds for ungrouped."
    )
    assert grouped_result.max_alloc < ungrouped_result.max_alloc, (
        "Expected grouped DHConv to use less memory than ungrouped, but got "
        f"{grouped_result.max_alloc/1024/1024:.2f} MB for grouped and "
        f"{ungrouped_result.max_alloc/1024/1024:.2f} MB for ungrouped."
    )


@pytest.mark.parametrize(
    "conditional_embed_dim_scalar, conditional_embed_dim_labels, conditional_embed_dim_noise, residual_filter_factor",  # noqa: E501
    [
        (0, 0, 0, 1),
        (16, 0, 0, 1),
        (16, 8, 0, 1),
        (16, 0, 16, 1),
        (0, 0, 16, 1),
        (16, 0, 0, 4),
    ],
)
def test_can_call_sfnonet(
    conditional_embed_dim_scalar: int,
    conditional_embed_dim_labels: int,
    conditional_embed_dim_noise: int,
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
    context = Context(
        embedding_scalar=context_embedding,
        labels=context_embedding_labels,
        noise=context_embedding_noise,
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
    context = Context(
        embedding_scalar=context_embedding,
        labels=context_embedding_labels,
        noise=context_embedding_noise,
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
    context = Context(
        embedding_scalar=context_embedding,
        labels=context_embedding_labels,
        noise=context_embedding_noise,
    )
    with torch.no_grad():
        output = model(x, context)
    if normalize_big_skip:
        assert not torch.isnan(output).any()
    else:
        assert torch.isnan(output).any()
