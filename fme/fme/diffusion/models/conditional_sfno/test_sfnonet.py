import pytest
import torch

from .sfnonet import SphericalFourierNeuralOperatorNet


@pytest.mark.parametrize("conditional_embed_dim", [16, 0])
def test_can_call_sfnonet(conditional_embed_dim: int):
    input_channels = 2
    output_channels = 3
    img_shape = (16, 32)
    n_samples = 4
    model = SphericalFourierNeuralOperatorNet(
        params=None,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        conditional_embed_dim=conditional_embed_dim,
    )
    x = torch.randn(n_samples, input_channels, *img_shape)
    context_embedding = torch.randn(n_samples, conditional_embed_dim)
    output = model(x, context_embedding)
    assert output.shape == (n_samples, output_channels, *img_shape)
