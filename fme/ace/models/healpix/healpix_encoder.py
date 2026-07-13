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

import torch
import torch.nn as nn

from .healpix_blocks import (
    ConvBlockConfig,
    DownsamplingBlockConfig,
    HEALPixBuildContext,
)


@dataclasses.dataclass
class UNetEncoderConfig:
    """
    Configuration for the UNet Encoder.

    Parameters:
        conv_block: Configuration for the convolutional block.
        down_sampling_block: Configuration for the down-sampling block.
        n_channels: Number of channels for each layer, by default (136, 68, 34).
        n_layers: Number of layers in each block, by default (2, 2, 1).
        dilations: List of dilation rates for the layers, by default None.
    """

    conv_block: ConvBlockConfig
    down_sampling_block: DownsamplingBlockConfig
    n_channels: List[int] = dataclasses.field(default_factory=lambda: [136, 68, 34])
    n_layers: List[int] = dataclasses.field(default_factory=lambda: [2, 2, 1])
    dilations: Optional[list] = None

    def build(
        self,
        input_channels: int,
        *,
        ctx: HEALPixBuildContext,
    ) -> nn.Module:
        """
        Builds the UNet Encoder model.

        Args:
            input_channels: Number of input channels (determined at build time).
            ctx: Shared HEALPix runtime settings for all child modules.

        Returns:
            UNet Encoder model.
        """
        return UNetEncoder(
            conv_block=self.conv_block,
            down_sampling_block=self.down_sampling_block,
            input_channels=input_channels,
            n_channels=self.n_channels,
            n_layers=self.n_layers,
            dilations=self.dilations,
            ctx=ctx,
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
        ctx: HEALPixBuildContext | None = None,
    ):
        """
        Args:
            conv_block: config for the convolutional block
            down_sampling_block: DownsamplingBlockConfig for the downsample block
            input_channels: # of input channels
            n_channels: # of channels in each encoder layer
            n_layers:, # of layers to use for the convolutional blocks
            dilations: list of dilations to use for the the convolutional blocks
            ctx: Shared HEALPix runtime settings for all child modules.
        """
        super().__init__()
        build_ctx = ctx or HEALPixBuildContext()
        self.n_channels = n_channels
        self.hpx_padding_mode = build_ctx.hpx_padding_mode

        if dilations is None:
            dilations = [1 for _ in range(len(n_channels))]

        nside_levels = build_ctx.nside_levels
        if nside_levels is not None and len(nside_levels) != len(n_channels):
            raise ValueError(
                f"nside length must match encoder levels; got {len(nside_levels)} "
                f"vs {len(n_channels)}"
            )

        conv_tpl = conv_block
        down_tpl = down_sampling_block
        down_factor = down_tpl.downsample_spatial_factor()

        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules: List[nn.Module] = []
            if n > 0:
                if nside_levels is not None:
                    coarse, fine = nside_levels[n - 1], nside_levels[n]
                    if coarse != fine * down_factor:
                        raise ValueError(
                            f"encoder nside[{n - 1}]={coarse} must equal "
                            f"nside[{n}] * downsample factor ({down_factor}), "
                            f"but nside[{n}]={fine}"
                        )
                modules.append(
                    down_tpl.build(
                        in_channels=old_channels,
                        ctx=build_ctx.layer(n - 1),
                    )
                )

            modules.append(
                conv_tpl.build(
                    in_channels=old_channels,
                    out_channels=curr_channel,
                    latent_channels=curr_channel,
                    dilation=dilations[n],
                    n_layers=n_layers[n],
                    ctx=build_ctx.layer(n),
                )
            )
            old_channels = curr_channel

            self.encoder.append(nn.Sequential(*modules))

        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Forward pass of the HEALPix Unet encoder

        Args:
            inputs: The inputs to encode

        Returns:
            The encoded values
        """
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs
