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

import torch.nn as nn

from fme.ace.models.healpix.healpix_activations import DownsamplingBlockConfig

from .healpix_blocks import ConvBlockConfig


@dataclasses.dataclass
class UNetEncoderConfig:
    """
    Configuration for the UNet Encoder.

    Parameters:
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
    n_channels: List[int] = dataclasses.field(default_factory=lambda: [136, 68, 34])
    n_layers: List[int] = dataclasses.field(default_factory=lambda: [2, 2, 1])
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


class UNetEncoder(nn.Module):
    """Generic UNetEncoder that can be applied to arbitrary meshes."""

    def __init__(
        self,
        conv_block: ConvBlockConfig,
        down_sampling_block: DownsamplingBlockConfig,
        input_channels: int = 3,
        n_channels: Sequence = (16, 32, 64),
        n_layers: Sequence = (2, 2, 1),
        dilations: Optional[list] = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Args:
            conv_block: config for the convolutional block
            down_sampling_block: DownsamplingBlockConfig for the downsample block
            input_channels: # of input channels
            n_channels: # of channels in each encoder layer
            n_layers:, # of layers to use for the convolutional blocks
            dilations: list of dilations to use for the the convolutional blocks
            enable_nhwc: if channel last format should be used
            enable_healpixpad: if healpixpad library should be used (true if installed)
        """
        super().__init__()
        self.n_channels = n_channels

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        # Build encoder
        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                down_sampling_block.enable_nhwc = enable_nhwc
                down_sampling_block.enable_healpixpad = enable_healpixpad
                modules.append(
                    down_sampling_block.build()  # Shapes are not used in these calls.
                )

            # Set up conv block
            conv_block.in_channels = old_channels
            conv_block.latent_channels = curr_channel
            conv_block.out_channels = curr_channel
            conv_block.dilation = dilations[n]
            conv_block.n_layers = n_layers[n]
            conv_block.enable_nhwc = enable_nhwc
            conv_block.enable_healpixpad = enable_healpixpad
            modules.append(conv_block.build())  # Shapes are not used in these calls.
            old_channels = curr_channel

            self.encoder.append(nn.Sequential(*modules))

        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        """
        Forward pass of the HEALPix Unet encoder

        Args:
            inputs: The inputs to enccode

        Returns:
            The encoded values
        """
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs

    def reset(self):
        """Resets the state of the decoder layers"""
        pass
