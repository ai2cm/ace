# flake8: noqa
# Copied from https://github.com/NVIDIA/modulus/commit/89a6091bd21edce7be4e0539cbd91507004faf08
# Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import List, Optional, Sequence

import torch as th
import torch.nn as nn

from .healpix_blocks import ConvBlockConfig, RecurrentBlockConfig


@dataclasses.dataclass
class UNetDecoderConfig:
    """
    Configuration for the UNet Decoder.

    Parameters:
        conv_block: Configuration for the convolutional block.
        up_sampling_block: Configuration for the up-sampling block.
        output_layer: Configuration for the output layer block.
        recurrent_block: Configuration for the recurrent block, by default None.
        n_channels: Number of channels for each layer, by default (34, 68, 136).
        n_layers: Number of layers in each block, by default (1, 2, 2).
        output_channels: Number of output channels, by default 1.
        dilations: List of dilation rates for the layers, by default None.
        enable_nhwc: Flag to enable NHWC data format, by default False.
        enable_healpixpad: Deprecated legacy toggle; omitted (``None``) unless migrating
            old configs without ``hpx_padding_mode``.
    """

    conv_block: ConvBlockConfig
    up_sampling_block: ConvBlockConfig
    output_layer: ConvBlockConfig
    recurrent_block: Optional[RecurrentBlockConfig] = None
    n_channels: List[int] = dataclasses.field(default_factory=lambda: [34, 68, 136])
    n_layers: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 2])
    output_channels: int = 1
    dilations: Optional[list] = None
    enable_nhwc: bool = False
    enable_healpixpad: Optional[bool] = None
    hpx_padding_mode: Optional[str] = None
    nside: Optional[int] = None
    compile_padding: bool = False

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
            hpx_padding_mode=self.hpx_padding_mode,
            nside=self.nside,
            compile_padding=self.compile_padding,
        )


class UNetDecoder(nn.Module):
    """Generic UNetDecoder that can be applied to arbitrary meshes."""

    def __init__(
        self,
        conv_block: ConvBlockConfig,
        up_sampling_block: ConvBlockConfig,
        output_layer: ConvBlockConfig,
        recurrent_block: Optional[RecurrentBlockConfig] = None,
        n_channels: Sequence = (64, 32, 16),
        n_layers: Sequence = (1, 2, 2),
        output_channels: int = 1,
        dilations: Optional[list] = None,
        enable_nhwc: bool = False,
        enable_healpixpad: Optional[bool] = None,
        hpx_padding_mode: Optional[str] = None,
        nside: Optional[int] = None,
        compile_padding: bool = False,
    ):
        """
        Initialize the UNetDecoder.

        Args:
            conv_block: Configuration for the convolutional block.
            up_sampling_block: Configuration for the upsampling block.
            output_layer: Configuration for the output layer.
            recurrent_block: Configuration for the recurrent block. If None, recurrent blocks are not used.
            n_channels: Sequence specifying the number of channels in each decoder layer.
            n_layers: Sequence specifying the number of layers in each block.
            output_channels: Number of output channels.
            dilations: List of dilations to use for the convolutional blocks.
            enable_nhwc: If True, use channel last format.
            enable_healpixpad: If True, use the healpixpad library if installed.
            hpx_padding_mode: HEALPix padding backend. Default ``"earth2grid"``;
                also supports ``"karlbauer"`` and ``"isolatitude"``.
        """
        super().__init__()
        self.channel_dim = 1
        face_nside = nside

        if dilations is None:
            dilations = [1 for _ in range(len(n_channels))]

        self.decoder = []
        n_levels = len(n_channels)
        for n, curr_channel in enumerate(n_channels):
            up_sample_module = None
            if n != 0:
                up_sampling_block.in_channels = curr_channel
                up_sampling_block.out_channels = curr_channel
                up_sampling_block.enable_nhwc = enable_nhwc
                up_sampling_block.enable_healpixpad = enable_healpixpad
                up_sampling_block.hpx_padding_mode = hpx_padding_mode
                up_sampling_block.compile_padding = compile_padding
                up_sampling_block.nside = face_nside
                up_sample_module = up_sampling_block.build()
                if face_nside is not None:
                    face_nside = face_nside * 2

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            conv_block.in_channels = curr_channel * 2 if n > 0 else curr_channel
            conv_block.latent_channels = curr_channel
            conv_block.out_channels = next_channel
            conv_block.dilation = dilations[n]
            conv_block.n_layers = n_layers[n]
            conv_block.enable_nhwc = enable_nhwc
            conv_block.enable_healpixpad = enable_healpixpad
            conv_block.hpx_padding_mode = hpx_padding_mode
            conv_block.compile_padding = compile_padding
            conv_block.nside = face_nside
            conv_module = conv_block.build()

            rec_module = None
            if recurrent_block is not None:
                recurrent_block.in_channels = next_channel
                recurrent_block.enable_healpixpad = enable_healpixpad
                recurrent_block.hpx_padding_mode = hpx_padding_mode
                recurrent_block.compile_padding = compile_padding
                recurrent_block.nside = face_nside
                rec_module = recurrent_block.build()

            self.decoder.append(
                nn.ModuleDict(
                    {
                        "upsamp": up_sample_module,
                        "conv": conv_module,
                        "recurrent": rec_module,
                    }
                )
            )

        self.decoder = nn.ModuleList(self.decoder)

        output_layer.in_channels = curr_channel
        output_layer.out_channels = output_channels
        output_layer.dilation = dilations[-1]
        output_layer.enable_nhwc = enable_nhwc
        output_layer.enable_healpixpad = enable_healpixpad
        output_layer.hpx_padding_mode = hpx_padding_mode
        output_layer.compile_padding = compile_padding
        output_layer.nside = face_nside

        self.output_layer = output_layer.build()

    def forward(self, inputs):
        """
        Forward pass of the UNetDecoder.

        Args:
            inputs: The inputs to the forward pass.

        Returns:
            The decoded values.
        """
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            if layer["upsamp"] is not None:
                up = layer["upsamp"](x)
                x = th.cat([up, inputs[-1 - n]], dim=self.channel_dim)
            x = layer["conv"](x)
            if layer["recurrent"] is not None:
                x = layer["recurrent"](x)
        return self.output_layer(x)

    def reset(self):
        """Resets the state of the decoder layers."""
        for layer in self.decoder:
            if layer["recurrent"] is not None:
                layer["recurrent"].reset()
