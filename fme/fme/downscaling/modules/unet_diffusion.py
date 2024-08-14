from typing import Optional, Tuple

import torch

from fme.core.device import get_device
from fme.downscaling.modules.registry import pad_right


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
        fine_topography: Optional fine topography to condition the latent
            variables on.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        coarse_shape: Tuple[int, int],
        target_shape: Tuple[int, int],
        downscale_factor: int,
        fine_topography: Optional[torch.Tensor],
    ):
        super(UNetDiffusionModule, self).__init__()
        self.unet = unet.to(get_device())
        self.coarse_shape = coarse_shape
        self.target_shape = target_shape
        self.downscale_factor = downscale_factor
        if fine_topography is not None:
            fine_topography = fine_topography.to(get_device())
        self.fine_topography = fine_topography

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

        interpolated = pad_right(interpolated, self.target_shape)
        latent = pad_right(latent, self.target_shape)

        if self.fine_topography is not None:
            batch_size = interpolated.shape[0]
            topography = (
                self.fine_topography.unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .to(get_device())
            )
            topography = pad_right(topography, self.target_shape)
            inputs = torch.concat((interpolated, topography), dim=1)
        else:
            inputs = interpolated

        outputs = self.unet(
            latent.to(get_device()),
            inputs,
            sigma=noise_level.to(get_device()),
            class_labels=None,
        )
        fine_shape = tuple(s * self.downscale_factor for s in self.coarse_shape)
        return outputs[..., : fine_shape[0], : fine_shape[1]]
