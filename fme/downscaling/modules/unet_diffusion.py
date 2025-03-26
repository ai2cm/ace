import torch

from fme.core.device import get_device


class UNetDiffusionModule(torch.nn.Module):
    """
    Maps interpolated coarse grid fields with fine grid latent variables,
    combined with embedded noise to a fine output using a U-Net.
    The latent variables are conditioned on the coarse image via simple
    concatenation in the channel dimension, which is why it is required to
    pass coarse fields interpolated to the target grid resolution. This
    is intended to be used for denoising diffusion models where the latent
    variables are noised fine grid targets and coarse are the paired coarse
    grid fields.

    Args:
        unet: The U-Net model.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
    ):
        super(UNetDiffusionModule, self).__init__()
        self.unet = unet.to(get_device())

    def forward(
        self,
        latent: torch.Tensor,
        conditioning: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the UNetDiffusion module.

        Args:
            conditioning: The conditioning input fields (same shape as target latents).
            latent: The latent diffusion variable on the fine grid.
            noise_level: The noise level of each example in the batch.
        """
        return self.unet(
            latent.to(get_device()),
            conditioning.to(get_device()),
            sigma=noise_level.to(get_device()),
            class_labels=None,
        )
