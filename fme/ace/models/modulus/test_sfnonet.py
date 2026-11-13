import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.testing import validate_tensor

from .sfnonet import SphericalFourierNeuralOperatorNet

DIR = os.path.abspath(os.path.dirname(__file__))


def test_sfnonet_output_is_unchanged():
    torch.manual_seed(0)
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
    ).to(device)
    # must initialize on CPU to get the same results on GPU
    x = torch.randn(n_samples, input_channels, *img_shape).to(device)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (n_samples, output_channels, *img_shape)
    validate_tensor(
        output,
        os.path.join(DIR, "testdata/test_sfnonet_output_is_unchanged.pt"),
    )


@pytest.mark.parametrize("checkpointing", [1, 2, 3])
def test_checkpointing_preserves_encoder_gradients(checkpointing):
    """Checkpointed SFNO keeps encoder gradients when inputs do not require them."""
    torch.manual_seed(0)
    model_args = {
        "params": None,
        "img_shape": (8, 16),
        "embed_dim": 8,
        "num_layers": 1,
        "in_chans": 2,
        "out_chans": 2,
        "use_mlp": True,
        "pos_embed": False,
        "big_skip": False,
        "spectral_layers": 1,
    }
    reference = SphericalFourierNeuralOperatorNet(**model_args, checkpointing=0)
    torch.manual_seed(0)
    checkpointed = SphericalFourierNeuralOperatorNet(
        **model_args, checkpointing=checkpointing
    )
    x = torch.randn(1, 2, *model_args["img_shape"])

    reference(x).square().mean().backward()
    checkpointed(x).square().mean().backward()

    for reference_parameter, checkpointed_parameter in zip(
        reference.encoder.parameters(), checkpointed.encoder.parameters()
    ):
        assert reference_parameter.grad is not None
        assert checkpointed_parameter.grad is not None
        torch.testing.assert_close(
            checkpointed_parameter.grad, reference_parameter.grad
        )
