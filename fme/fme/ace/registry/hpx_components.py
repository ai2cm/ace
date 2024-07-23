import dataclasses
from typing import Literal, Optional, cast

import torch.nn as nn

from fme.ace.models.healpix.healpix_blocks import (
    BasicConvBlock,
    ConvGRUBlock,
    ConvLSTMBlock,
    ConvNeXtBlock,
    SymmetricConvNeXtBlock,
    TransposedConvUpsample,
)
from fme.ace.models.healpix.healpix_recunet import HEALPixBlockConfig

from .hpx_activations import CappedGELUConfig


@dataclasses.dataclass
class RecurrentBlockConfig(HEALPixBlockConfig):
    """
    Configuration for the recurrent block.

    Attributes:
        in_channels: Number of input channels, default is 3.
        kernel_size: Size of the kernel, default is 1.
        enable_nhwc: Flag to enable NHWC data format, default is False.
        enable_healpixpad: Flag to enable HEALPix padding, default is False.
        block_type: Type of recurrent block, either "ConvGRUBlock" or "ConvLSTMBlock",
        default is "ConvGRUBlock".
    """

    in_channels: int = 3
    kernel_size: int = 1
    enable_nhwc: bool = False
    enable_healpixpad: bool = False
    block_type: Literal["ConvGRUBlock", "ConvLSTMBlock"] = "ConvGRUBlock"

    def build(self) -> nn.Module:
        """
        Builds the recurrent block model.

        Returns:
            Recurrent block.
        """
        if self.block_type == "ConvGRUBlock":
            return ConvGRUBlock(
                in_channels=self.in_channels,
                kernel_size=self.kernel_size,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        elif self.block_type == "ConvLSTMBlock":
            return ConvLSTMBlock(
                in_channels=self.in_channels,
                kernel_size=self.kernel_size,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        else:
            raise ValueError(f"Unsupported block type: {self.block_type}")


@dataclasses.dataclass
class ConvBlockConfig(HEALPixBlockConfig):
    """
    Configuration for the convolutional block.

    Attributes:
        in_channels: Number of input channels, default is 3.
        out_channels: Number of output channels, default is 1.
        kernel_size: Size of the kernel, default is 3.
        dilation: Dilation rate, default is 1.
        n_layers: Number of layers, default is 1.
        upsampling: Upsampling factor for TransposedConvUpsample, default is 2.
        upscale_factor: Upscale factor for ConvNeXtBlock and SymmetricConvNeXtBlock,
            default is 4.
        latent_channels: Number of latent channels, default is None.
        activation: Activation configuration, default is None.
        enable_nhwc: Flag to enable NHWC data format, default is False.
        enable_healpixpad: Flag to enable HEALPix padding, default is False.
        block_type: Type of block, default is "BasicConvBlock".
    """

    in_channels: int = 3
    out_channels: int = 1
    kernel_size: int = 3
    dilation: int = 1
    n_layers: int = 1
    upsampling: int = 2
    upscale_factor: int = 4
    latent_channels: Optional[int] = None
    activation: Optional[CappedGELUConfig] = None
    enable_nhwc: bool = False
    enable_healpixpad: bool = False
    block_type: Literal[
        "BasicConvBlock",
        "ConvNeXtBlock",
        "SymmetricConvNeXtBlock",
        "TransposedConvUpsample",
    ] = "BasicConvBlock"

    def build(self) -> nn.Module:
        """
        Builds the convolutional block model.

        Returns:
            Convolutional block model.
        """
        if self.block_type == "BasicConvBlock":
            return BasicConvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                n_layers=self.n_layers,
                latent_channels=self.latent_channels,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        elif self.block_type == "ConvNeXtBlock":
            if self.latent_channels is None:
                self.latent_channels = 1
            return ConvNeXtBlock(
                in_channels=self.in_channels,
                latent_channels=cast(int, self.latent_channels),
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                upscale_factor=self.upscale_factor,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        elif self.block_type == "SymmetricConvNeXtBlock":
            if self.latent_channels is None:
                self.latent_channels = 1
            return SymmetricConvNeXtBlock(
                in_channels=self.in_channels,
                latent_channels=cast(int, self.latent_channels),
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                upscale_factor=self.upscale_factor,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        elif self.block_type == "TransposedConvUpsample":
            return TransposedConvUpsample(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                upsampling=self.upsampling,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        else:
            raise ValueError(f"Unsupported block type: {self.block_type}")
