import pytest
import torch

from fme.core.device import get_device

from .layers import ConditionalLayerNorm, Context, ContextConfig


@pytest.mark.parametrize("global_layer_norm", [True, False])
@pytest.mark.parametrize("n_channels", [32])
@pytest.mark.parametrize("embed_dim_scalar", [9, 0])
@pytest.mark.parametrize("embed_dim_noise", [10, 0])
@pytest.mark.parametrize("embed_dim_labels", [11, 0])
@pytest.mark.parametrize("img_shape", [(8, 16)])
def test_conditional_layer_norm(
    n_channels: int,
    img_shape: tuple[int, int],
    global_layer_norm: bool,
    embed_dim_scalar: int,
    embed_dim_labels: int,
    embed_dim_noise: int,
):
    epsilon = 1e-6
    device = get_device()
    conditional_layer_norm = ConditionalLayerNorm(
        n_channels,
        img_shape,
        context_config=ContextConfig(
            embed_dim_scalar=embed_dim_scalar,
            embed_dim_labels=embed_dim_labels,
            embed_dim_noise=embed_dim_noise,
        ),
        global_layer_norm=global_layer_norm,
        epsilon=epsilon,
    ).to(device)
    x = torch.randn(1, n_channels, *img_shape, device=device) * 5 + 2
    context_embedding_scalar = torch.randn(1, embed_dim_scalar, device=device)
    context_embedding_labels = torch.randn(1, embed_dim_labels, device=device)
    context_embedding_noise = torch.randn(1, embed_dim_noise, *img_shape, device=device)
    context = Context(
        embedding_scalar=context_embedding_scalar,
        labels=context_embedding_labels,
        noise=context_embedding_noise,
    )
    output = conditional_layer_norm(x, context)
    assert output.shape == x.shape
    torch.testing.assert_close(
        output.mean(), torch.tensor(0.0, device=device), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        output.std(), torch.tensor(1.0, device=device), atol=1e-3, rtol=0
    )
    if not global_layer_norm:
        zero = torch.zeros(1, *img_shape, device=device)
        torch.testing.assert_close(output.mean(dim=1), zero, atol=1e-3, rtol=0)
        torch.testing.assert_close(
            (((n_channels - 1) / n_channels) ** 0.5 * output.std(dim=1) - 1),
            zero,
            atol=1e-3,
            rtol=0,
        )
