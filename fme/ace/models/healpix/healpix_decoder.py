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

from .healpix_blocks import ConvBlockConfig, HEALPixBuildContext, UpsamplingBlockConfig


@dataclasses.dataclass
class UNetDecoderConfig:
    """
    Configuration for the UNet Decoder.

    Parameters:
        conv_block: Configuration for the convolutional block.
        up_sampling_block: Configuration for the spatial upsampling block
            (transpose conv, smoothed interpolate+conv, etc.).
        output_layer: Configuration for the output layer block.
        n_channels: Number of channels for each layer, by default (34, 68, 136).
        n_layers: Number of layers in each block, by default (1, 2, 2).
        dilations: List of dilation rates for the layers, by default None.
    """

    conv_block: ConvBlockConfig
    up_sampling_block: UpsamplingBlockConfig
    output_layer: ConvBlockConfig
    n_channels: List[int] = dataclasses.field(default_factory=lambda: [34, 68, 136])
    n_layers: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 2])
    dilations: Optional[list] = None

    def build(
        self,
        output_channels: int,
        *,
        ctx: HEALPixBuildContext,
    ) -> nn.Module:
        """
        Builds the UNet Decoder model.

        Args:
            output_channels: Number of output channels (determined at build time).
            ctx: Shared HEALPix runtime settings for all child modules.

        Returns:
            UNet Decoder model.
        """
        return UNetDecoder(
            conv_block=self.conv_block,
            up_sampling_block=self.up_sampling_block,
            output_layer=self.output_layer,
            n_channels=self.n_channels,
            n_layers=self.n_layers,
            output_channels=output_channels,
            dilations=self.dilations,
            ctx=ctx,
        )


class UNetDecoder(nn.Module):
    """Generic UNetDecoder that can be applied to arbitrary meshes."""

    def __init__(
        self,
        conv_block: ConvBlockConfig,
        up_sampling_block: UpsamplingBlockConfig,
        output_layer: ConvBlockConfig,
        n_channels: Sequence = (64, 32, 16),
        n_layers: Sequence = (1, 2, 2),
        output_channels: int = 1,
        dilations: Optional[list] = None,
        ctx: HEALPixBuildContext | None = None,
    ):
        """
        Initialize the UNetDecoder.

        Args:
            conv_block: Configuration for the convolutional block.
            up_sampling_block: Configuration for the upsampling block.
            output_layer: Configuration for the output layer.
            n_channels: Sequence specifying the number of channels in each decoder layer.
            n_layers: Sequence specifying the number of layers in each block.
            output_channels: Number of output channels.
            dilations: List of dilations to use for the convolutional blocks.
            ctx: Shared HEALPix runtime settings for all child modules.
        """
        super().__init__()
        build_ctx = ctx or HEALPixBuildContext()
        self.channel_dim = 1

        if dilations is None:
            dilations = [1 for _ in range(len(n_channels))]

        nside_levels = build_ctx.nside_levels
        if nside_levels is not None and len(nside_levels) != len(n_channels):
            raise ValueError(
                f"nside length must match decoder levels; got {len(nside_levels)} "
                f"vs {len(n_channels)}"
            )

        conv_tpl = conv_block
        up_tpl = up_sampling_block
        up_factor = up_tpl.stride
        n_levels = len(n_channels)

        self.decoder = []
        for n, curr_channel in enumerate(n_channels):
            up_sample_module = None
            level_nside = (
                None if nside_levels is None else nside_levels[n_levels - 1 - n]
            )
            if n != 0:
                if nside_levels is not None:
                    before = nside_levels[n_levels - n]
                    after = level_nside
                    if before * up_factor != after:
                        raise ValueError(
                            f"decoder nside upsample: nside[{n_levels - 1 - n}]={after} "
                            f"must equal nside[{n_levels - n}] * upsample factor "
                            f"({up_factor}), but nside[{n_levels - n}]={before}"
                        )
                up_sample_module = up_tpl.build(
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    ctx=build_ctx.layer(
                        n_levels - n,
                        nside_after=level_nside,
                    ),
                )

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            conv_module = conv_tpl.build(
                in_channels=curr_channel * 2 if n > 0 else curr_channel,
                out_channels=next_channel,
                latent_channels=curr_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                ctx=build_ctx.layer(n_levels - 1 - n),
            )

            self.decoder.append(
                nn.ModuleDict(
                    {
                        "upsamp": up_sample_module,
                        "conv": conv_module,
                    }
                )
            )

        self.decoder = nn.ModuleList(self.decoder)

        self.output_layer = output_layer.build(
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            ctx=build_ctx.layer(0),
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the UNetDecoder.

        Args:
            inputs: Sequence of tensors, one for each decoder level.

        Returns:
            The decoded values.
        """
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            if layer["upsamp"] is not None:
                up = layer["upsamp"](x)
                x = torch.cat([up, inputs[-1 - n]], dim=self.channel_dim)
            x = layer["conv"](x)
        return self.output_layer(x)
