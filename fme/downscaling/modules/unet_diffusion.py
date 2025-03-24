import torch

from fme.core.device import get_device


class UNetDiffusionModule(torch.nn.Module):
    """
    Maps a coarse image and fine grid latent variables, combined with embedded
    noise to a fine output using a U-Net. The latent variables are conditioned
    on the coarse image via simple concatenation in the channel dimension. This
    is intended to be used for denoising diffusion models where the latent
    variables are noised fine grid targets and coarse are the paired coarse
    grid fields.

    Args:
        unet: The U-Net model.
        coarse_shape: The shape of the coarse input.
        target_shape: The input and output shape of the u-net.
        downscale_factor: The factor by which the coarse input is downscaled to
            the target output.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        downscale_factor: int,
    ):
        super(UNetDiffusionModule, self).__init__()
        self.unet = unet.to(get_device())
        self.downscale_factor = downscale_factor

    def forward(
        self,
        latent: torch.Tensor,
        coarse: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the UNetDiffusion module.

        Args:
            coarse: The coarse fields.
            latent: The latent diffusion variable on the fine grid.
            noise_level: The noise level of each example in the batch.
        """
        interpolated = torch.nn.functional.interpolate(
            coarse.to(get_device()),
            scale_factor=(self.downscale_factor, self.downscale_factor),
            mode="bicubic",
            align_corners=True,
        )

        return self.unet(
            latent.to(get_device()),
            interpolated,
            sigma=noise_level.to(get_device()),
            class_labels=None,
        )
