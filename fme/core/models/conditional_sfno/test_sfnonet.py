import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor

from .layers import Context, ContextConfig
from .sfnonet import get_lat_lon_sfnonet

DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize(
    "conditional_embed_dim_scalar, conditional_embed_dim_2d, residual_filter_factor",
    [
        (0, 0, 1),
        (16, 0, 1),
        (16, 16, 1),
        (0, 16, 1),
        (16, 0, 4),
    ],
)
def test_can_call_sfnonet(
    conditional_embed_dim_scalar: int,
    conditional_embed_dim_2d: int,
    residual_filter_factor: int,
):
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = SimpleNamespace(
        embed_dim=16, num_layers=2, residual_filter_factor=residual_filter_factor
    )
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        context_config=ContextConfig(
            embed_dim_scalar=conditional_embed_dim_scalar,
            embed_dim_2d=conditional_embed_dim_2d,
        ),
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)
    context_embedding = torch.randn(
        n_samples, conditional_embed_dim_scalar, device=device
    )
    context_embedding_2d = torch.randn(
        n_samples, conditional_embed_dim_2d, *img_shape, device=device
    )
    context = Context(context_embedding, context_embedding_2d)
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
                embed_dim_2d=0,
            ),
        ).to(device)


def test_sfnonet_output_is_unchanged():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    conditional_embed_dim_scalar = 8
    conditional_embed_dim_2d = 16
    device = get_device()
    params = SimpleNamespace(embed_dim=16, num_layers=2)
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        context_config=ContextConfig(
            embed_dim_scalar=conditional_embed_dim_scalar,
            embed_dim_2d=conditional_embed_dim_2d,
        ),
    ).to(device)
    # must initialize on CPU to get the same results on GPU
    x = torch.randn(n_samples, input_channels, *img_shape).to(device)
    context_embedding = torch.randn(n_samples, conditional_embed_dim_scalar).to(device)
    context_embedding_2d = torch.randn(
        n_samples, conditional_embed_dim_2d, *img_shape
    ).to(device)
    context = Context(context_embedding, context_embedding_2d)
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
    conditional_embed_dim_2d = 16
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
            embed_dim=16, num_layers=2, normalize_big_skip=normalize_big_skip
        )
        model = get_lat_lon_sfnonet(
            params=params,
            img_shape=img_shape,
            in_chans=input_channels,
            out_chans=output_channels,
            context_config=ContextConfig(
                embed_dim_scalar=conditional_embed_dim_scalar,
                embed_dim_2d=conditional_embed_dim_2d,
            ),
        ).to(device)
    finally:
        nn.LayerNorm = original_layer_norm
    x = torch.full((n_samples, input_channels, *img_shape), torch.nan).to(device)
    context_embedding = torch.randn(n_samples, conditional_embed_dim_scalar).to(device)
    context_embedding_2d = torch.randn(
        n_samples, conditional_embed_dim_2d, *img_shape
    ).to(device)
    context = Context(context_embedding, context_embedding_2d)
    with torch.no_grad():
        output = model(x, context)
    if normalize_big_skip:
        assert not torch.isnan(output).any()
    else:
        assert torch.isnan(output).any()
