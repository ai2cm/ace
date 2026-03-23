import pytest
import torch
from torch import nn

from fme.core.device import get_device

from .layers import Context, ContextConfig
from .localnet import LocalNetConfig, get_lat_lon_localnet


@pytest.mark.parametrize(
    "conditional_embed_dim_scalar, conditional_embed_dim_labels, "
    "conditional_embed_dim_noise, "
    "conditional_embed_dim_pos",
    [
        (0, 0, 0, 0),
        (16, 8, 0, 0),
        (16, 0, 16, 0),
        (16, 15, 14, 13),
        (0, 0, 0, 16),
        (0, 0, 16, 0),
    ],
)
def test_can_call_localnet_disco(
    conditional_embed_dim_scalar: int,
    conditional_embed_dim_labels: int,
    conditional_embed_dim_noise: int,
    conditional_embed_dim_pos: int,
):
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = LocalNetConfig(
        embed_dim=16,
        block_types=["disco", "disco"],
    )
    model = get_lat_lon_localnet(
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


def test_can_call_localnet_conv1x1():
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = LocalNetConfig(
        embed_dim=16,
        block_types=["conv1x1", "conv1x1"],
    )
    model = get_lat_lon_localnet(
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


def test_can_call_localnet_mixed_blocks():
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = LocalNetConfig(
        embed_dim=16,
        block_types=["disco", "conv1x1", "disco", "conv1x1"],
    )
    model = get_lat_lon_localnet(
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
        params = LocalNetConfig(
            embed_dim=16,
            block_types=["disco", "disco"],
            normalize_big_skip=normalize_big_skip,
            global_layer_norm=True,  # so it uses nn.LayerNorm
        )
        model = get_lat_lon_localnet(
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


def test_no_big_skip():
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = LocalNetConfig(
        embed_dim=16,
        block_types=["disco", "disco"],
        big_skip=False,
    )
    model = get_lat_lon_localnet(
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


def test_unknown_filter_type_raises():
    with pytest.raises(ValueError, match="Invalid block type"):
        LocalNetConfig(block_types=["spectral"])  # type: ignore[list-item]


def test_backward_pass():
    """Test that gradients flow through the network."""
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 2
    device = get_device()
    params = LocalNetConfig(
        embed_dim=16,
        block_types=["disco", "disco"],
    )
    model = get_lat_lon_localnet(
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
