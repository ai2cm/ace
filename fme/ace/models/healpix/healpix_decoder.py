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
from typing import List, Literal, Optional, Sequence

import torch as th
import torch.nn as nn

from .healpix_blocks import ConvBlockConfig, UpsamplingBlockConfig
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
        output_channels: Number of output channels, by default 1.
        dilations: List of dilation rates for the layers, by default None.
        enable_nhwc: Flag to enable NHWC data format, by default False.
        hpx_padding_mode: HEALPix padding backend (``"earth2grid"``, ``"karlbauer"``,
            or ``"isolatitude"``), by default ``"earth2grid"``.
        nside: Face height/width per decoder level (shallowest to deepest), or
            ``None`` to omit per-level padding resolution.
    """

    conv_block: ConvBlockConfig
    up_sampling_block: UpsamplingBlockConfig
    output_layer: ConvBlockConfig
    n_channels: List[int] = dataclasses.field(default_factory=lambda: [34, 68, 136])
    n_layers: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 2])
    output_channels: int = 1
    dilations: Optional[list] = None
    enable_nhwc: bool = False
    hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid"
    nside: Optional[Sequence[int]] = None

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
            n_channels=self.n_channels,
            n_layers=self.n_layers,
            output_channels=self.output_channels,
            dilations=self.dilations,
            enable_nhwc=self.enable_nhwc,
            hpx_padding_mode=self.hpx_padding_mode,
            nside=self.nside,
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
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid",
        nside: Optional[Sequence[int]] = None,
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
            enable_nhwc: If True, use channel last format.
            hpx_padding_mode: HEALPix padding backend. Default ``"earth2grid"``;
                also supports ``"karlbauer"`` and ``"isolatitude"``.
            nside: Face height/width per level (shallowest to deepest). Length must
                match ``len(n_channels)`` when set. Decoder stage ``n`` (deepest first)
                uses ``nside[-(n + 1)]``.
        """
        super().__init__()
        self.channel_dim = 1

        if dilations is None:
            dilations = [1 for _ in range(len(n_channels))]

        nside_levels: Optional[tuple[int, ...]] = None
        if nside is not None:
            nside_levels = tuple(int(v) for v in nside)
            if len(nside_levels) != len(n_channels):
                raise ValueError(
                    f"nside length must match decoder levels; got {len(nside_levels)} "
                    f"vs {len(n_channels)}"
                )

        conv_tpl = dataclasses.replace(conv_block)
        up_tpl = dataclasses.replace(up_sampling_block)
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
                up_cfg = dataclasses.replace(
                    up_tpl,
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    enable_nhwc=enable_nhwc,
                    hpx_padding_mode=hpx_padding_mode,
                    nside=None if nside_levels is None else nside_levels[n_levels - n],
                    nside_after=level_nside,
                )
                up_sample_module = up_cfg.build()

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            conv_cfg = dataclasses.replace(
                conv_tpl,
                in_channels=curr_channel * 2 if n > 0 else curr_channel,
                latent_channels=curr_channel,
                out_channels=next_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                enable_nhwc=enable_nhwc,
                hpx_padding_mode=hpx_padding_mode,
                nside=level_nside,
            )
            conv_module = conv_cfg.build()

            self.decoder.append(
                nn.ModuleDict(
                    {
                        "upsamp": up_sample_module,
                        "conv": conv_module,
                    }
                )
            )

        self.decoder = nn.ModuleList(self.decoder)

        out_nside = None if nside_levels is None else nside_levels[0]
        out_cfg = dataclasses.replace(
            output_layer,
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            enable_nhwc=enable_nhwc,
            hpx_padding_mode=hpx_padding_mode,
            nside=out_nside,
        )
        self.output_layer = out_cfg.build()

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
        return self.output_layer(x)
