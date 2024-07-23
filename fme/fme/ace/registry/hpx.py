import dataclasses
from typing import Optional, Sequence, Tuple

import torch.nn as nn

from fme.ace.models.healpix.healpix_decoder import UNetDecoder
from fme.ace.models.healpix.healpix_encoder import UNetEncoder
from fme.ace.models.healpix.healpix_recunet import HEALPixBlockConfig, HEALPixRecUNet
from fme.ace.registry.hpx_activations import DownsamplingBlockConfig
from fme.ace.registry.hpx_components import ConvBlockConfig, RecurrentBlockConfig
from fme.ace.registry.registry import ModuleConfig, register


@dataclasses.dataclass
class UNetEncoderConfig(HEALPixBlockConfig):
    """
    Configuration for the UNet Encoder.

    Attributes:
        conv_block: Configuration for the convolutional block.
        down_sampling_block: Configuration for the down-sampling block.
        input_channels: Number of input channels, by default 3.
        n_channels: Number of channels for each layer, by default (136, 68, 34).
        n_layers: Number of layers in each block, by default (2, 2, 1).
        dilations: List of dilation rates for the layers, by default None.
        enable_nhwc: Flag to enable NHWC data format, by default False.
        enable_healpixpad: Flag to enable HEALPix padding, by default False.
    """

    conv_block: ConvBlockConfig
    down_sampling_block: DownsamplingBlockConfig
    input_channels: int = 3
    n_channels: Sequence[int] = (136, 68, 34)
    n_layers: Sequence[int] = (2, 2, 1)
    dilations: Optional[list] = None
    enable_nhwc: bool = False
    enable_healpixpad: bool = False

    def build(self) -> nn.Module:
        """
        Builds the UNet Encoder model.

        Returns:
            UNet Encoder model.
        """
        return UNetEncoder(
            conv_block=self.conv_block,
            down_sampling_block=self.down_sampling_block,
            input_channels=self.input_channels,
            n_channels=self.n_channels,
            n_layers=self.n_layers,
            dilations=self.dilations,
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
        )


@dataclasses.dataclass
class UNetDecoderConfig(HEALPixBlockConfig):
    """
    Configuration for the UNet Decoder.

    Attributes:
        conv_block: Configuration for the convolutional block.
        up_sampling_block: Configuration for the up-sampling block.
        output_layer: Configuration for the output layer block.
        recurrent_block: Configuration for the recurrent block, by default None.
        n_channels: Number of channels for each layer, by default (34, 68, 136).
        n_layers: Number of layers in each block, by default (1, 2, 2).
        output_channels: Number of output channels, by default 1.
        dilations: List of dilation rates for the layers, by default None.
        enable_nhwc: Flag to enable NHWC data format, by default False.
        enable_healpixpad: Flag to enable HEALPix padding, by default False.
    """

    conv_block: ConvBlockConfig
    up_sampling_block: ConvBlockConfig
    output_layer: ConvBlockConfig
    recurrent_block: Optional[RecurrentBlockConfig] = None
    n_channels: Sequence[int] = (34, 68, 136)
    n_layers: Sequence[int] = (1, 2, 2)
    output_channels: int = 1
    dilations: Optional[list] = None
    enable_nhwc: bool = False
    enable_healpixpad: bool = False

    def build(self) -> nn.Module:
        """
        Builds the UNet Decoder model.

        Returns:
            UNet Decoder model.
        """
        return UNetDecoder(
            conv_block=self.conv_block,
            up_sampling_block=self.up_sampling_block,
            output_layer=self.output_layer,
            recurrent_block=self.recurrent_block,
            n_channels=self.n_channels,
            n_layers=self.n_layers,
            output_channels=self.output_channels,
            dilations=self.dilations,
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
        )


@register("HEALPixRecUNet")
@dataclasses.dataclass
class HEALPixRecUNetBuilder(ModuleConfig):
    """
    Configuration for the HEALPixRecUNet architecture used in DLWP.

    Attributes:
        presteps: Number of pre-steps, by default 1.
        input_time_dim: Input time dimension, by default 0.
        output_time_dim: Output time dimension, by default 0.
        delta_time: Delta time interval, by default "6h".
        reset_cycle: Reset cycle interval, by default "24h".
        input_channels: Number of input channels, by default 8.
        output_channels: Number of output channels, by default 8.
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
    input_channels: int = 8
    output_channels: int = 8
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
        # Construct the HEALPixRecUNet module here using the parameters
        return HEALPixRecUNet(
            encoder=self.encoder,
            decoder=self.decoder,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
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
