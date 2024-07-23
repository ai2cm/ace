# flake8: noqa
# Copied from https://github.com/ai2cm/modulus/commit/22df4a9427f5f12ff6ac891083220e7f2f54d229
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

from typing import Optional, Sequence

import torch as th
import torch.nn as nn

from fme.ace.registry.hpx_activations import DownsamplingBlockConfig
from fme.ace.registry.hpx_components import ConvBlockConfig, RecurrentBlockConfig


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
