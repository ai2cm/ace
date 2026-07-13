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
        nside_levels = ctx.nside_levels
        if nside_levels is not None and len(nside_levels) != len(self.n_channels):
            raise ValueError(
                f"nside length must match encoder levels; got {len(nside_levels)} "
                f"vs {len(self.n_channels)}"
            )
        return self._build(input_channels, ctx=ctx)

    def _build(
        self,
        input_channels: int,
        *,
        ctx: HEALPixBuildContext,
    ) -> "UNetEncoder":
        """Construct the ordered per-level encoder modules and the impl.

        Builds one ``nn.Sequential(down?, conv)`` per level, threading channel
        counts and validating the nside downsample ratios, then passes the built
        module list to :class:`UNetEncoder`.
        """
        dilations = self.dilations
        if dilations is None:
            dilations = [1 for _ in range(len(self.n_channels))]

        nside_levels = ctx.nside_levels
        down_factor = self.down_sampling_block.downsample_spatial_factor()

        old_channels = input_channels
        encoder: List[nn.Module] = []
        for n, curr_channel in enumerate(self.n_channels):
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
                    self.down_sampling_block.build(
                        in_channels=old_channels,
                        ctx=ctx.layer(n - 1),
                    )
                )

            modules.append(
                self.conv_block.build(
                    in_channels=old_channels,
                    out_channels=curr_channel,
                    latent_channels=curr_channel,
                    dilation=dilations[n],
                    n_layers=self.n_layers[n],
                    ctx=ctx.layer(n),
                )
            )
            old_channels = curr_channel

            encoder.append(nn.Sequential(*modules))

        return UNetEncoder(encoder=nn.ModuleList(encoder))


class UNetEncoder(nn.Module):
    """Generic UNetEncoder that can be applied to arbitrary meshes."""

    def __init__(self, encoder: nn.ModuleList):
        """
        Args:
            encoder: Ordered per-level encoder modules built by
                :meth:`UNetEncoderConfig._build`.
        """
        super().__init__()
        self.encoder = encoder

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
