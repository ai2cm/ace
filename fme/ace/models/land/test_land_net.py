import pytest
import torch

from fme.ace.models.land.land_net import LandNet


@pytest.mark.parametrize("hidden_dims", [[64], [64, 64], [128, 64]])
@pytest.mark.parametrize("use_positional_embedding", [False, True])
def test_land_net(hidden_dims, use_positional_embedding):
    # Create a LandNet instance
    input_channels = 8
    output_channels = 8

    height = 8
    width = 16
    img_shape = (height, width)

    model = LandNet(
        img_shape,
        input_channels,
        hidden_dims,
        output_channels,
        network_type="MLP",
        use_positional_embedding=use_positional_embedding,
    )

    # Create a random input tensor with the shape (batch_size, channels, height, width)
    batch_size = 4
    x = torch.randn(batch_size, input_channels, height, width)

    # Forward pass
    output = model(x)

    # Check the output shape
    assert output.shape == (batch_size, output_channels, height, width), (
        f"Expected output shape {(batch_size, output_channels, height, width)}, "
        f"but got {output.shape}"
    )
