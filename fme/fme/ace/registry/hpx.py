import dataclasses
from typing import Tuple

import torch.nn as nn

from fme.ace.models.healpix.healpix_decoder import UNetDecoderConfig
from fme.ace.models.healpix.healpix_encoder import UNetEncoderConfig
from fme.ace.models.healpix.healpix_recunet import HEALPixRecUNet
from fme.ace.registry.registry import ModuleConfig, register


@register("HEALPixRecUNet")
@dataclasses.dataclass
class HEALPixRecUNetBuilder(ModuleConfig):
    """
    Configuration for the HEALPixRecUNet architecture used in DLWP.

    Attributes:
        presteps: Number of pre-steps, by default 1.
        delta_time: Delta time interval, by default "6h".
        input_time_dim: Input time dimension, by default 0.
        output_time_dim: Output time dimension, by default 0.
        reset_cycle: Reset cycle interval, by default "24h".
        n_constants: Number of constant input channels, by default 2.
        decoder_input_channels: Number of input channels for the decoder, by default 1.
        enable_nhwc: Flag to enable NHWC data format, by default False.
        enable_healpixpad: Flag to enable HEALPix padding, by default False.
    """

    encoder: UNetEncoderConfig
    decoder: UNetDecoderConfig
    presteps: int = 1
    input_time_dim: int = 0
    output_time_dim: int = 0
    delta_time: str = "6h"
    reset_cycle: str = "24h"
    n_constants: int = 2
    decoder_input_channels: int = 1
    enable_nhwc: bool = False
    enable_healpixpad: bool = False

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        """
        Builds the HEALPixRecUNet model.

        Args:
            n_in_channels: Number of input channels.
            n_out_channels: Number of output channels.
            img_shape: Shape of the input image.

        Returns:
            HEALPixRecUNet model.
        """
        # Construct the HEALPixRecUNet module here using the parameters.
        return HEALPixRecUNet(
            encoder=self.encoder,
            decoder=self.decoder,
            input_channels=n_in_channels,
            output_channels=n_out_channels,
            n_constants=self.n_constants,
            decoder_input_channels=self.decoder_input_channels,
            input_time_dim=self.input_time_dim,
            output_time_dim=self.output_time_dim,
            delta_time=self.delta_time,
            reset_cycle=self.reset_cycle,
            presteps=self.presteps,
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
        )
