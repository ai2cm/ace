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
        nside_levels = ctx.nside_levels
        if nside_levels is not None and len(nside_levels) != len(self.n_channels):
            raise ValueError(
                f"nside length must match decoder levels; got {len(nside_levels)} "
                f"vs {len(self.n_channels)}"
            )
        return self._build(output_channels, ctx=ctx)

    def _build(
        self,
        output_channels: int,
        *,
        ctx: HEALPixBuildContext,
    ) -> "UNetDecoder":
        """Construct the ordered per-level decoder modules plus output layer.

        Builds one :class:`DecoderLevel` per level (threading channel counts and
        validating the nside upsample ratios) and the output layer, then passes
        the built modules to :class:`UNetDecoder`.
        """
        dilations = self.dilations
        if dilations is None:
            dilations = [1 for _ in range(len(self.n_channels))]

        nside_levels = ctx.nside_levels
        up_factor = self.up_sampling_block.stride
        n_channels = self.n_channels
        n_levels = len(n_channels)

        decoder: List[nn.Module] = []
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
                up_sample_module = self.up_sampling_block.build(
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    ctx=ctx.layer(
                        n_levels - n,
                        nside_after=level_nside,
                    ),
                )

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            conv_module = self.conv_block.build(
                in_channels=curr_channel * 2 if n > 0 else curr_channel,
                out_channels=next_channel,
                latent_channels=curr_channel,
                dilation=dilations[n],
                n_layers=self.n_layers[n],
                ctx=ctx.layer(n_levels - 1 - n),
            )

            decoder.append(DecoderLevel(upsamp=up_sample_module, conv=conv_module))

        output_layer = self.output_layer.build(
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            ctx=ctx.layer(0),
        )

        return UNetDecoder(
            decoder=nn.ModuleList(decoder),
            output_layer=output_layer,
        )


class DecoderLevel(nn.Module):
    """One decoder level: optional upsample + skip-concat, then a conv block.

    The deepest level (built first, ``upsamp is None``) receives the encoder
    bottleneck and applies only its conv block. Each shallower level upsamples
    the running activation, concatenates the matching encoder skip connection
    along the channel dimension, then applies its conv block. Bundling the
    upsample and conv into one module keeps this per-level branching structure
    with the module that runs it, so :class:`UNetDecoder.forward` never reaches
    into the level's internals.
    """

    def __init__(self, upsamp: Optional[nn.Module], conv: nn.Module):
        super().__init__()
        self.upsamp = upsamp
        self.conv = conv
        self.channel_dim = 1

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Running activation from the deeper level.
            skip: Matching encoder skip connection (unused when ``upsamp`` is
                ``None``, i.e. the deepest level).
        """
        if self.upsamp is not None:
            x = torch.cat([self.upsamp(x), skip], dim=self.channel_dim)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """Applies the decoder levels deepest-to-shallowest, then the output layer.

    Receives the ordered per-level :class:`DecoderLevel` modules and the output
    layer already built by :meth:`UNetDecoderConfig._build`; it constructs
    nothing itself, so it is directly constructible from its signature and its
    ``forward`` needs no knowledge of how the levels were assembled.
    """

    def __init__(
        self,
        decoder: nn.ModuleList,
        output_layer: nn.Module,
    ):
        """
        Args:
            decoder: Ordered per-level :class:`DecoderLevel` modules built by
                :meth:`UNetDecoderConfig._build`.
            output_layer: Final output-projection module.
        """
        super().__init__()
        self.decoder = decoder
        self.output_layer = output_layer

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the UNetDecoder.

        Args:
            inputs: Sequence of tensors, one per decoder level (encoder outputs,
                shallow-to-deep).

        Returns:
            The decoded values.
        """
        x = inputs[-1]
        for n, level in enumerate(self.decoder):
            x = level(x, inputs[-1 - n])
        return self.output_layer(x)
