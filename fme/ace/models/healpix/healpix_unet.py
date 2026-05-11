# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

"""
Non-recurrent HEALPix UNet model.

Adapted from the modulus-uw ``physicsnemo.models.dlwp_healpix.HEALPixUNet`` to the
ace codebase. Unlike the modulus-uw implementation, this version is a pure neural
network module that takes a single tensor in the canonical ace HEALPix layout and
returns a single tensor in the same layout. All data reshaping, time stepping,
prognostic/diagnostic splitting, and residual prediction are delegated to the
ace stepper.
"""

from typing import Optional, Sequence

import torch as th
import torch.nn as nn

from .healpix_decoder import UNetDecoderConfig
from .healpix_encoder import UNetEncoderConfig
from .healpix_layers import HEALPixFoldFaces, HEALPixUnfoldFaces
from .healpix_paddings import warn_deprecated_enable_healpixpad


class HEALPixUNet(nn.Module):
    """Non-recurrent UNet on the HEALPix mesh.

    The model operates on tensors with shape ``[B, F, C, H, W]`` where ``F=12``
    is the number of HEALPix faces and ``C`` is the channel count. Faces are
    folded into the batch dimension before the encoder/decoder and unfolded
    after the decoder, so the encoder/decoder operate on plain 4D tensors of
    shape ``[B*F, C, H, W]``.

    This module is intended to be a drop-in replacement for other ace
    architectures (e.g. ``SphericalFourierNeuralOperatorNet``) when the input
    grid is HEALPix. It does not handle multi-step rolling or residual
    prediction; both are handled by the stepper.
    """

    CHANNEL_DIM = 2  # [B, F, C, H, W]

    def __init__(
        self,
        encoder: UNetEncoderConfig,
        decoder: UNetDecoderConfig,
        input_channels: int,
        output_channels: int,
        enable_nhwc: bool = False,
        enable_healpixpad: Optional[bool] = None,
        hpx_padding_mode: Optional[str] = None,
        compile_padding: bool = False,
        nside: Sequence[int] | int | None = (64, 32, 16),
    ):
        """
        Initialize the HEALPixUNet model.

        Args:
            encoder: Configuration for the U-net encoder.
            decoder: Configuration for the U-net decoder. ``recurrent_block``
                must be ``None``; recurrence belongs in the stepper, not the
                UNet decoder here.
            input_channels: Number of channels in the input tensor (i.e. the
                size of the channel dimension of the tensor passed to
                ``forward``).
            output_channels: Number of channels in the output tensor.
            enable_nhwc: Use NHWC tensor layout for child modules.
            enable_healpixpad: Deprecated. When ``hpx_padding_mode`` is omitted,
                ``True`` maps to ``"earth2grid"`` and ``False`` to
                ``"karlbauer"`` with a deprecation warning. Prefer setting
                ``hpx_padding_mode`` directly.
            hpx_padding_mode: HEALPix padding backend. One of ``"earth2grid"``,
                ``"karlbauer"``, or ``"isolatitude"``. Defaults to
                ``"earth2grid"`` when both ``hpx_padding_mode`` and
                ``enable_healpixpad`` are omitted.
            compile_padding: If ``True``, apply ``torch.compile`` to
                isolatitude padding modules.
            nside: Face size(s) per UNet level (shallowest to deepest). May be
                a sequence with ``len(encoder.n_channels)`` entries, an int
                (treated as the shallowest level with halving per level), or
                ``None`` (defaults to ``64`` halving per level).
        """
        super().__init__()
        if decoder.recurrent_block is not None:
            raise ValueError(
                "HEALPixUNet is non-recurrent; set decoder.recurrent_block "
                "to None and handle time/recurrence in the stepper."
            )

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.enable_nhwc = enable_nhwc
        self.compile_padding = compile_padding
        self.hpx_padding_mode = warn_deprecated_enable_healpixpad(
            enable_healpixpad, hpx_padding_mode
        )

        levels = len(encoder.n_channels)
        if isinstance(nside, int):
            nside_levels = tuple(max(1, nside // (2**i)) for i in range(levels))
        elif nside is None:
            nside_levels = tuple(max(1, 64 // (2**i)) for i in range(levels))
        else:
            nside_levels = tuple(int(v) for v in nside)
        if len(nside_levels) != levels:
            raise ValueError(
                f"nside length must match UNet levels; got {len(nside_levels)} "
                f"vs {levels}"
            )
        if len(decoder.n_channels) != levels:
            raise ValueError(
                "encoder and decoder must have same number of levels for "
                "nside mapping"
            )
        self.nside = nside_levels

        self.fold = HEALPixFoldFaces(enable_nhwc=enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces=12, enable_nhwc=enable_nhwc)

        encoder.input_channels = input_channels
        encoder.enable_nhwc = enable_nhwc
        encoder.enable_healpixpad = None
        encoder.hpx_padding_mode = self.hpx_padding_mode
        encoder.compile_padding = compile_padding
        encoder.nside = self.nside[0]
        self.encoder = encoder.build()

        decoder.output_channels = output_channels
        decoder.enable_nhwc = enable_nhwc
        decoder.enable_healpixpad = None
        decoder.hpx_padding_mode = self.hpx_padding_mode
        decoder.compile_padding = compile_padding
        decoder.nside = self.nside[-1]
        self.decoder = decoder.build()

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            inputs: Tensor of shape ``[B, F=12, input_channels, H, W]``.

        Returns:
            Tensor of shape ``[B, F=12, output_channels, H, W]``.
        """
        if inputs.ndim != 5:
            raise ValueError(
                "HEALPixUNet expects a 5D input [B, F, C, H, W]; got tensor "
                f"with shape {tuple(inputs.shape)}"
            )
        if inputs.shape[self.CHANNEL_DIM] != self.input_channels:
            raise ValueError(
                f"Expected input to have {self.input_channels} channels at "
                f"dim {self.CHANNEL_DIM}, got {inputs.shape[self.CHANNEL_DIM]}."
            )

        folded = self.fold(inputs)
        encodings = self.encoder(folded)
        decodings = self.decoder(encodings)
        return self.unfold(decodings)
