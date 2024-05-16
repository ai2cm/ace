import torch

from fme.downscaling.modules.unet_diffusion import UNetDiffusionModule


class AddNoiseModule(torch.nn.Module):
    """Simple module for testing."""

    def __init__(self, n_output_channels):
        self.n_output_channels = n_output_channels
        super(AddNoiseModule, self).__init__()

    def forward(self, inputs, noise, ignored=None):
        output = inputs + noise
        return output[:, : self.n_output_channels, ...]


def test_UNetDiffusionModule_runs():
    downscale_factor = 2
    coarse_shape = (8, 16)
    fine_shape = coarse_shape[0] * downscale_factor, coarse_shape[1] * downscale_factor
    n_channels = 3

    net = AddNoiseModule(n_channels)
    module = UNetDiffusionModule(net, coarse_shape, (16, 32), downscale_factor, None)

    batch_size = 1
    coarse = torch.randn(batch_size, n_channels, *coarse_shape)
    latent = torch.randn(batch_size, n_channels, *fine_shape)
    noise = torch.randn(batch_size, 1, 1, 1)

    assert (batch_size, n_channels, *fine_shape) == module(coarse, latent, noise).shape
