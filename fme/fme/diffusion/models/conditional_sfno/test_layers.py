from typing import Tuple

import pytest
import torch

from .layers import ConditionalLayerNorm


@pytest.mark.parametrize("global_layer_norm", [True, False])
@pytest.mark.parametrize("n_channels", [32])
@pytest.mark.parametrize("n_context_embedding", [128, 0])
@pytest.mark.parametrize("img_shape", [(8, 16)])
def test_conditional_layer_norm(
    n_channels: int,
    img_shape: Tuple[int, int],
    global_layer_norm: bool,
    n_context_embedding,
):
    epsilon = 1e-6
    conditional_layer_norm = ConditionalLayerNorm(
        n_channels, img_shape, global_layer_norm, n_context_embedding, epsilon
    )
    x = torch.randn(1, n_channels, *img_shape) * 5 + 2
    context_embedding = torch.randn(1, n_context_embedding)
    output: torch.Tensor = conditional_layer_norm(x, context_embedding)
    assert output.shape == x.shape
    torch.testing.assert_close(output.mean(), torch.tensor(0.0), atol=1e-3, rtol=0)
    torch.testing.assert_close(output.std(), torch.tensor(1.0), atol=1e-3, rtol=0)
    if not global_layer_norm:
        zero = torch.zeros(1, *img_shape)
        torch.testing.assert_close(output.mean(dim=1), zero, atol=1e-3, rtol=0)
        torch.testing.assert_close(
            (((n_channels - 1) / n_channels) ** 0.5 * output.std(dim=1) - 1),
            zero,
            atol=1e-3,
            rtol=0,
        )
