import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor

from .layers import Context, ContextConfig
from .sfnonet import SphericalFourierNeuralOperatorNet

DIR = os.path.abspath(os.path.dirname(__file__))


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
    model = SphericalFourierNeuralOperatorNet(
        params=None,
        embed_dim=16,
        num_layers=2,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        context_config=ContextConfig(
            embed_dim_scalar=conditional_embed_dim_scalar,
            embed_dim_labels=conditional_embed_dim_labels,
            embed_dim_noise=conditional_embed_dim_noise,
        ),
        residual_filter_factor=residual_filter_factor,
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


def test_sfnonet_output_is_unchanged():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    conditional_embed_dim_scalar = 8
    conditional_embed_dim_labels = 0
    conditional_embed_dim_noise = 16
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
