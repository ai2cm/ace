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

from .healpix_blocks import ConvBlockConfig, DownsamplingBlockConfig
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
        hpx_padding_mode: HEALPix padding backend (``"earth2grid"``, ``"karlbauer"``,
            or ``"isolatitude"``), by default ``"earth2grid"``.
        nside: Face height/width per encoder level (shallowest to deepest), or
            ``None`` to omit per-level padding resolution.
    """

    conv_block: ConvBlockConfig
    down_sampling_block: DownsamplingBlockConfig
    input_channels: int = 3
    n_channels: List[int] = dataclasses.field(default_factory=lambda: [136, 68, 34])
    n_layers: List[int] = dataclasses.field(default_factory=lambda: [2, 2, 1])
    dilations: Optional[list] = None
    enable_nhwc: bool = False
    hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid"
    nside: Optional[Sequence[int]] = None

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
            hpx_padding_mode=self.hpx_padding_mode,
            nside=self.nside,
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
        hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid",
        nside: Optional[Sequence[int]] = None,
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
            hpx_padding_mode: HEALPix padding backend. Default ``"earth2grid"``;
                also supports ``"karlbauer"`` and ``"isolatitude"``.
            nside: Face height/width per encoder level (shallowest to deepest). Length
                must match ``len(n_channels)`` when set.
        """
        super().__init__()
        self.n_channels = n_channels
        self.hpx_padding_mode = hpx_padding_mode

        if dilations is None:
            dilations = [1 for _ in range(len(n_channels))]

        nside_levels: Optional[tuple[int, ...]] = None
        if nside is not None:
            nside_levels = tuple(int(v) for v in nside)
            if len(nside_levels) != len(n_channels):
                raise ValueError(
                    f"nside length must match encoder levels; got {len(nside_levels)} "
                    f"vs {len(n_channels)}"
                )

        conv_tpl = dataclasses.replace(conv_block)
        down_tpl = dataclasses.replace(down_sampling_block)
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
                down_cfg = dataclasses.replace(
                    down_tpl,
                    enable_nhwc=enable_nhwc,
                    hpx_padding_mode=hpx_padding_mode,
                    nside=None if nside_levels is None else nside_levels[n - 1],
                    in_channels=old_channels,
                )
                modules.append(down_cfg.build())

            conv_cfg = dataclasses.replace(
                conv_tpl,
                in_channels=old_channels,
                latent_channels=curr_channel,
                out_channels=curr_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                enable_nhwc=enable_nhwc,
                hpx_padding_mode=hpx_padding_mode,
                nside=None if nside_levels is None else nside_levels[n],
            )
            modules.append(conv_cfg.build())
            old_channels = curr_channel

            self.encoder.append(nn.Sequential(*modules))

        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, inputs: th.Tensor) -> Sequence[th.Tensor]:
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
