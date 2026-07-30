import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor

from .ankur import AnkurLocalNetConfig, get_lat_lon_ankur_localnet
from .layers import Context, ContextConfig

DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("use_disco_encoder", [True, False])
@pytest.mark.parametrize("pos_embed", [True, False])
def test_can_call_ankur_localnet(use_disco_encoder: bool, pos_embed: bool):
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = AnkurLocalNetConfig(
        embed_dim=16,
        use_disco_encoder=use_disco_encoder,
        pos_embed=pos_embed,
    )
    model = get_lat_lon_ankur_localnet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)
    context = Context(
        embedding_scalar=torch.randn(n_samples, 0, device=device),
        labels=torch.randn(n_samples, 0, device=device),
        noise=torch.randn(n_samples, 0, *img_shape, device=device),
        embedding_pos=torch.randn(n_samples, 0, *img_shape, device=device),
    )
    output = model(x, context)
    assert output.shape == (n_samples, output_channels, *img_shape)


def test_ankur_localnet_with_context_config():
    """AnkurLocalNet accepts context_config in factory but ignores context."""
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = AnkurLocalNetConfig(embed_dim=16)
    model = get_lat_lon_ankur_localnet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        context_config=ContextConfig(
            embed_dim_scalar=8,
            embed_dim_labels=4,
            embed_dim_noise=16,
            embed_dim_pos=0,
        ),
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)
    context = Context(
        embedding_scalar=torch.randn(n_samples, 8, device=device),
        labels=torch.randn(n_samples, 4, device=device),
        noise=torch.randn(n_samples, 16, *img_shape, device=device),
        embedding_pos=None,
    )
    output = model(x, context)
    assert output.shape == (n_samples, output_channels, *img_shape)


def test_ankur_localnet_context_does_not_affect_output():
    """Verify that different context values produce the same output."""
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 2
    device = get_device()
    params = AnkurLocalNetConfig(embed_dim=16)
    model = get_lat_lon_ankur_localnet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)
    context1 = Context(
        embedding_scalar=torch.randn(n_samples, 0, device=device),
        labels=torch.randn(n_samples, 0, device=device),
        noise=torch.randn(n_samples, 0, *img_shape, device=device),
        embedding_pos=torch.randn(n_samples, 0, *img_shape, device=device),
    )
    context2 = Context(
        embedding_scalar=torch.randn(n_samples, 0, device=device),
        labels=torch.randn(n_samples, 0, device=device),
        noise=torch.randn(n_samples, 0, *img_shape, device=device),
        embedding_pos=torch.randn(n_samples, 0, *img_shape, device=device),
    )
    with torch.no_grad():
        output1 = model(x, context1)
        output2 = model(x, context2)
    torch.testing.assert_close(output1, output2)


def test_ankur_localnet_backward():
    """Test that gradients flow through the network."""
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 2
    device = get_device()
    params = AnkurLocalNetConfig(embed_dim=16, use_disco_encoder=True)
    model = get_lat_lon_ankur_localnet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)
    context = Context(
        embedding_scalar=torch.randn(n_samples, 0, device=device),
        labels=torch.randn(n_samples, 0, device=device),
        noise=torch.randn(n_samples, 0, *img_shape, device=device),
        embedding_pos=torch.randn(n_samples, 0, *img_shape, device=device),
    )
    output = model(x, context)
    loss = output.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_ankur_localnet_disco_kernel_size():
    """Test that a non-default kernel size works."""
    input_channels = 4
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 2
    device = get_device()
    params = AnkurLocalNetConfig(
        embed_dim=16,
        use_disco_encoder=True,
        disco_kernel_size=5,
    )
    model = get_lat_lon_ankur_localnet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)
    context = Context(
        embedding_scalar=torch.randn(n_samples, 0, device=device),
        labels=torch.randn(n_samples, 0, device=device),
        noise=torch.randn(n_samples, 0, *img_shape, device=device),
        embedding_pos=torch.randn(n_samples, 0, *img_shape, device=device),
    )
    output = model(x, context)
    assert output.shape == (n_samples, output_channels, *img_shape)


def setup_ankur_localnet():
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = AnkurLocalNetConfig(embed_dim=16, use_disco_encoder=True)
    model = get_lat_lon_ankur_localnet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
    ).to(device)
    # must initialize on CPU to get the same results on GPU
    x = torch.randn(n_samples, input_channels, *img_shape).to(device)
    context = Context(
        embedding_scalar=torch.randn(n_samples, 0).to(device),
        labels=torch.randn(n_samples, 0).to(device),
        noise=torch.randn(n_samples, 0, *img_shape).to(device),
        embedding_pos=torch.randn(n_samples, 0, *img_shape).to(device),
    )
    return model, x, context


def test_ankur_localnet_output_is_unchanged():
    torch.manual_seed(0)
    model, x, context = setup_ankur_localnet()
    with torch.no_grad():
        output = model(x, context)
    validate_tensor(
        output,
        os.path.join(DIR, "testdata/test_ankur_localnet_output.pt"),
    )
