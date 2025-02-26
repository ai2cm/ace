import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor

from .layers import Context, ContextConfig
from .sfnonet import SphericalFourierNeuralOperatorNet

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
    model = SphericalFourierNeuralOperatorNet(
        params=None,
        embed_dim=16,
        num_layers=2,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        context_config=ContextConfig(
            embed_dim_scalar=conditional_embed_dim_scalar,
            embed_dim_2d=conditional_embed_dim_2d,
        ),
        residual_filter_factor=residual_filter_factor,
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


def test_sfnonet_output_is_unchanged():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    conditional_embed_dim_scalar = 8
    conditional_embed_dim_2d = 16
    device = get_device()
    model = SphericalFourierNeuralOperatorNet(
        params=None,
        embed_dim=16,
        num_layers=2,
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
