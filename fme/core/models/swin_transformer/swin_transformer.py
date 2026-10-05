# Adapted from ArchesWeatherGen [*], a 3D Swin U-Net for weather forecasting.
#
# [*] Urbain et al., "ArchesWeatherGen: a Generative Model for Ensemble Weather
#     Forecasting", arXiv:2412.12971 (2024).
#     https://github.com/INRIA/geoarches
#     https://doi.org/10.48550/arXiv.2412.12971
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, ARCHES team @ INRIA. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""A Swin U-Net backbone for ACE.

An adaptation of ArchesWeather's 3D Swin U-Net to ACE's ``(B, C, H, W)``
interface. By default all vertical levels are stacked into the channel
dimension, giving a purely 2D network. When a ``ColumnLayout`` is supplied,
levels are instead embedded individually and carried as a separate axis
folded into the batch dimension, so that each block can attend along the
vertical (see ``CrossLevelAttention``); window attention is unaffected either
way, matching ArchesWeather's vertical window size of 1.
"""

import dataclasses
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.models.swin_transformer.boundary_padding import TensorPadding
from fme.core.stacker import ColumnLayout

from .swin_layers import BasicLayer, ChannelMixer, PatchExpanding, PatchMerging


class SwinTransformerNet(nn.Module):
    """2D Swin U-Net with column interaction and optional AdaLN conditioning.

    The network pads the input to a multiple of ``2 * window_size``, encodes
    it with a Conv2d to ``embed_dim`` channels, applies a ``ChannelMixer``,
    then runs a U-Net of ``BasicLayer`` stages (depths ``[2, 6, 6, 2] *
    depth_multiplier``) with one downsample / upsample and an optional skip
    connection, before decoding back to ``out_chans`` and cropping to the
    original shape.

    Args:
        in_chans: Number of input channels.
        out_chans: Number of output channels.
        img_shape: ``(H, W)`` of the input data (before padding).
        embed_dim: Channel dimension of the first/last U-Net stage.
        depth_multiplier: Scales the per-stage depths ``[2, 6, 6, 2]``.
        num_heads: Attention heads for each of the four stages.
        window_size: ``(ws_h, ws_w)`` attention window.
        mlp_ratio: Hidden-dim multiplier for block MLPs.
        drop_path_rate: Maximum stochastic-depth rate.
        use_skip: Whether to concatenate the layer-1 skip into the decoder.
        context_config: Conditioning configuration.  In ``"adaln"`` mode,
            scalar and label conditioning are applied as independent additive
            AdaLN projections; ``None`` (or both 0) disables AdaLN.  In
            ``"cln"`` mode, ``embed_dim_noise`` drives per-block
            ``ConditionalLayerNorm``, and a non-zero ``embed_dim_scalar``
            additionally conditions those norms on a scalar embedding.
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
        conditioning: ``"adaln"`` (default) for native per-stage DiT AdaLN, or
            ``"cln"`` for ``ConditionalLayerNorm``-based noise conditioning.
        in_layout: Vertical layout of the input channels. When given (together
            with ``out_layout``), levels are embedded individually and each
            block attends along the vertical axis. When ``None``, levels stay
            stacked in the channel dimension and the network is purely 2D.
        out_layout: Vertical layout of the output channels. Must have the same
            number of levels as ``in_layout``.
        column_num_heads: Number of heads for cross-level attention.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        img_shape: tuple[int, int],
        embed_dim: int = 96,
        depth_multiplier: int = 1,
        num_heads: tuple[int, ...] = (3, 6, 6, 3),
        window_size: tuple[int, int] = (4, 8),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        use_skip: bool = True,
        context_config: ContextConfig | None = None,
        mlp_layer: str = "mlp",
        conditioning: Literal["adaln", "cln"] = "adaln",
        cpb_hidden_dim: int = 64,
        lat_coords: torch.Tensor | None = None,
        padding_conf: dict | None = None,
        in_layout: ColumnLayout | None = None,
        out_layout: ColumnLayout | None = None,
        column_num_heads: int = 8,
    ):
        super().__init__()
        if depth_multiplier < 1:
            raise ValueError(f"depth_multiplier must be >= 1, got {depth_multiplier}")
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.img_shape = img_shape
        self.use_skip = use_skip
        self.window_size = window_size
        self.conditioning = conditioning
        self.in_layout = in_layout
        self.out_layout = out_layout
        if (in_layout is None) != (out_layout is None):
            raise ValueError(
                "in_layout and out_layout must both be provided or both be None"
            )
        if in_layout is not None and out_layout is not None:
            if in_layout.n_levels != out_layout.n_levels:
                raise ValueError(
                    "Input and output must have the same number of levels, got "
                    f"{in_layout.n_levels} and {out_layout.n_levels}"
                )
            # The surface fields are projected to a single extra token that
            # participates in cross-level attention alongside the levels,
            # matching ArchesWeather's encoder.
            self.n_tokens = in_layout.n_levels + 1
        else:
            self.n_tokens = 1

        ws_h, ws_w = window_size
        self.pad_mult = (ws_h * 2, ws_w * 2)

        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = padding_conf["activate"]
        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)
            pl = padding_conf["pad_lat"]
            pw = padding_conf["pad_lon"]
            H0 = img_shape[0] + pl[0] + pl[1]
            W0 = img_shape[1] + pw[0] + pw[1]
        else:
            H0, W0 = img_shape
        Hp = math.ceil(H0 / self.pad_mult[0]) * self.pad_mult[0]
        Wp = math.ceil(W0 / self.pad_mult[1]) * self.pad_mult[1]
        self.padded_shape = (Hp, Wp)

        if self.use_padding and lat_coords is not None:
            padded_lat_coords = []
            if pl[0] > 0:
                padded_lat_coords.append(torch.flip(lat_coords[: pl[0]], dims=[0]))
            padded_lat_coords.append(lat_coords)
            if pl[1] > 0:
                padded_lat_coords.append(torch.flip(lat_coords[-pl[1] :], dims=[0]))
            lat_coords = torch.cat(padded_lat_coords)

        if context_config is not None:
            self.embed_dim_scalar = context_config.embed_dim_scalar
            self.embed_dim_labels = context_config.embed_dim_labels
            self.embed_dim_noise = context_config.embed_dim_noise
        else:
            self.embed_dim_scalar = 0
            self.embed_dim_labels = 0
            self.embed_dim_noise = 0

        if lat_coords is not None:
            pad_h = Hp - H0
            lat_full: torch.Tensor | None = (
                torch.cat([lat_coords, lat_coords[-1:].expand(pad_h)])
                if pad_h > 0
                else lat_coords
            )  # (Hp,)
            lat_half: torch.Tensor | None = (
                lat_full[::2] + lat_full[1::2]  # type: ignore[index]
            ) / 2  # (Hp//2,)
        else:
            lat_full = lat_half = None

        if in_layout is not None and out_layout is not None:
            # Separate stems keep vertical identity: every level of the input
            # column is embedded by the same weights (so the level axis is
            # meaningful), while surface fields get their own projection and
            # become one extra token.
            self.level_encoder: nn.Module = nn.Conv2d(
                in_layout.n_vars, embed_dim, kernel_size=3, padding=1
            )
            self.surface_encoder: nn.Module = nn.Conv2d(
                len(in_layout.surface_indices), embed_dim, kernel_size=3, padding=1
            )
            self.level_decoder: nn.Module = nn.Conv2d(
                embed_dim, out_layout.n_vars, kernel_size=3, padding=1
            )
            self.surface_decoder: nn.Module = nn.Conv2d(
                embed_dim, len(out_layout.surface_indices), kernel_size=3, padding=1
            )
            self.encoder = nn.Identity()
            self.decoder: nn.Module = nn.Identity()
        else:
            self.encoder = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.channel_mixer = ChannelMixer(embed_dim)

        d = depth_multiplier
        # DropPath schedule matching ArchesWeather: shallow stages (1 & 4)
        # share the first 2*d rates, deep stages (2 & 3) share the last 6*d.
        dpr = torch.linspace(0, drop_path_rate / d, 8 * d).tolist()
        dpr_shallow = dpr[: 2 * d]
        # Intentionally shared between layer2 and layer3 (matching ArchesWeather):
        # both deep stages get the same drop-path schedule.
        dpr_deep = dpr[2 * d : 8 * d]

        self.layer1 = BasicLayer(
            embed_dim,
            (Hp, Wp),
            2 * d,
            num_heads[0],
            window_size,
            mlp_ratio,
            dpr_shallow,
            embed_dim_scalar=self.embed_dim_scalar,
            embed_dim_labels=self.embed_dim_labels,
            mlp_layer=mlp_layer,
            conditioning=conditioning,
            context_config=context_config,
            cpb_hidden_dim=cpb_hidden_dim,
            lat_coords=lat_full,
            n_levels=self.n_tokens,
            column_num_heads=column_num_heads,
        )
        self.downsample = PatchMerging(embed_dim)
        self.layer2 = BasicLayer(
            2 * embed_dim,
            (Hp // 2, Wp // 2),
            6 * d,
            num_heads[1],
            window_size,
            mlp_ratio,
            dpr_deep,
            embed_dim_scalar=self.embed_dim_scalar,
            embed_dim_labels=self.embed_dim_labels,
            mlp_layer=mlp_layer,
            conditioning=conditioning,
            context_config=context_config,
            cpb_hidden_dim=cpb_hidden_dim,
            lat_coords=lat_half,
            n_levels=self.n_tokens,
            column_num_heads=column_num_heads,
        )
        self.layer3 = BasicLayer(
            2 * embed_dim,
            (Hp // 2, Wp // 2),
            6 * d,
            num_heads[2],
            window_size,
            mlp_ratio,
            dpr_deep,
            embed_dim_scalar=self.embed_dim_scalar,
            embed_dim_labels=self.embed_dim_labels,
            mlp_layer=mlp_layer,
            conditioning=conditioning,
            context_config=context_config,
            cpb_hidden_dim=cpb_hidden_dim,
            lat_coords=lat_half,
            n_levels=self.n_tokens,
            column_num_heads=column_num_heads,
        )
        self.upsample = PatchExpanding(2 * embed_dim)  # -> embed_dim, 2x spatial

        decoder_dim = 2 * embed_dim if use_skip else embed_dim
        self.layer4 = BasicLayer(
            decoder_dim,
            (Hp, Wp),
            2 * d,
            num_heads[3],
            window_size,
            mlp_ratio,
            dpr_shallow,
            embed_dim_scalar=self.embed_dim_scalar,
            embed_dim_labels=self.embed_dim_labels,
            mlp_layer=mlp_layer,
            conditioning=conditioning,
            context_config=context_config,
            cpb_hidden_dim=cpb_hidden_dim,
            lat_coords=lat_full,
            n_levels=self.n_tokens,
            column_num_heads=column_num_heads,
        )
        self.final_linear = nn.Linear(decoder_dim, embed_dim, bias=False)
        if in_layout is None:
            self.decoder = nn.Conv2d(embed_dim, out_chans, kernel_size=3, padding=1)

    def _repeat_over_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat a per-sample conditioning tensor across the vertical axis.

        The latent folds the vertical axis into the batch dimension, so
        conditioning defined per sample must be repeated to match. Uses
        ``repeat_interleave`` so that the layout is sample-major, matching
        ``_encode_columns``.
        """
        if self.n_tokens == 1:
            return x
        return x.repeat_interleave(self.n_tokens, dim=0)

    def _encode_columns(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B * n_tokens, embed_dim, H, W)``.

        Levels are embedded by shared weights so that the vertical axis is
        meaningful, and the surface fields become one additional token which
        participates in cross-level attention alongside them.
        """
        assert self.in_layout is not None
        layout = self.in_layout
        B, _, H, W = x.shape
        n_vars, n_levels = layout.n_vars, layout.n_levels
        # (B, n_vars * n_levels, H, W) -> (B * n_levels, n_vars, H, W), so that
        # each level is embedded independently by the same weights.
        column = x[:, layout.start : layout.stop]
        column = column.view(B, n_vars, n_levels, H, W).transpose(1, 2)
        column = column.reshape(B * n_levels, n_vars, H, W)
        column = self.level_encoder(column)  # (B * n_levels, embed_dim, H, W)
        surface = x[:, list(layout.surface_indices)]
        surface = self.surface_encoder(surface)  # (B, embed_dim, H, W)
        # Concatenate along the level axis: surface first, matching
        # ArchesWeather's ``cat([surface.unsqueeze(2), level], dim=2)``.
        column = column.view(B, n_levels, -1, H, W)
        tokens = torch.cat([surface.unsqueeze(1), column], dim=1)
        return tokens.reshape(B * self.n_tokens, -1, H, W)

    def _decode_columns(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Inverse of ``_encode_columns``, back to ``(B, out_chans, H, W)``."""
        assert self.out_layout is not None
        layout = self.out_layout
        _, embed_dim, H, W = x.shape
        n_vars, n_levels = layout.n_vars, layout.n_levels
        tokens = x.view(batch_size, self.n_tokens, embed_dim, H, W)
        surface = self.surface_decoder(tokens[:, 0])
        column = tokens[:, 1:].reshape(batch_size * n_levels, embed_dim, H, W)
        column = self.level_decoder(column)  # (B * n_levels, n_vars, H, W)
        # (B * n_levels, n_vars, H, W) -> (B, n_vars * n_levels, H, W), undoing
        # the level-major transpose applied when encoding.
        column = column.view(batch_size, n_levels, n_vars, H, W).transpose(1, 2)
        column = column.reshape(batch_size, n_vars * n_levels, H, W)
        out = x.new_empty((batch_size, self.out_chans, H, W))
        out[:, layout.start : layout.stop] = column
        out[:, list(layout.surface_indices)] = surface
        return out

    def forward(self, x: torch.Tensor, context: Context | None = None) -> torch.Tensor:
        if self.use_padding:
            x = self.padding_opt.pad(x)
        _, _, H, W = x.shape
        Hp, Wp = self.padded_shape
        pad_h = Hp - H
        pad_w = Wp - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        batch_size = x.shape[0]
        if self.in_layout is not None:
            x = self._encode_columns(x)  # (B * n_tokens, embed_dim, Hp, Wp)
        else:
            x = self.encoder(x)  # (B, embed_dim, Hp, Wp)
        x = x.permute(0, 2, 3, 1)  # (B[*Z], Hp, Wp, embed_dim)
        x = self.channel_mixer(x)

        # AdaLN conditioning: extract scalar/label embeddings from context.
        cond_scalar: torch.Tensor | None = None
        cond_labels: torch.Tensor | None = None
        if self.conditioning == "adaln" and (
            self.embed_dim_scalar > 0 or self.embed_dim_labels > 0
        ):
            if context is None:
                raise ValueError(
                    "context is required for a conditioned SwinTransformerNet"
                )
            if self.embed_dim_scalar > 0:
                if context.embedding_scalar is None:
                    raise ValueError("embedding_scalar is required")
                cond_scalar = self._repeat_over_tokens(context.embedding_scalar)
            if self.embed_dim_labels > 0:
                # May be None; BasicLayer skips the label path when it is.
                cond_labels = (
                    None
                    if context.labels is None
                    else self._repeat_over_tokens(context.labels)
                )

        # CLN conditioning: pad and subsample noise to match U-Net resolutions.
        ctx_full: Context | None = context
        ctx_half: Context | None = context
        if self.conditioning == "cln" and self.embed_dim_noise > 0:
            if context is None or context.noise is None:
                raise ValueError(
                    "context.noise is required for a cln-conditioned SwinTransformerNet"
                )
            noise = context.noise  # (B, embed_dim_noise, H, W)
            if self.use_padding:
                noise = self.padding_opt.pad(noise)
            if pad_h > 0 or pad_w > 0:
                noise = F.pad(noise, (0, pad_w, 0, pad_h))
            # The latent carries the vertical axis in its batch dimension, so
            # every per-sample conditioning tensor is shared across a sample's
            # levels.
            noise = self._repeat_over_tokens(noise)
            noise_half = noise[..., ::2, ::2]
            scalar = (
                None
                if context.embedding_scalar is None
                else self._repeat_over_tokens(context.embedding_scalar)
            )
            labels = (
                None
                if context.labels is None
                else self._repeat_over_tokens(context.labels)
            )
            ctx_full = dataclasses.replace(
                context, noise=noise, embedding_scalar=scalar, labels=labels
            )
            ctx_half = dataclasses.replace(
                context, noise=noise_half, embedding_scalar=scalar, labels=labels
            )

        x = self.layer1(x, cond_scalar, cond_labels, context=ctx_full)
        skip = x
        x = self.downsample(x)
        x = self.layer2(x, cond_scalar, cond_labels, context=ctx_half)
        x = self.layer3(x, cond_scalar, cond_labels, context=ctx_half)
        x = self.upsample(x)
        if self.use_skip:
            x = torch.cat([x, skip], dim=-1)
        x = self.layer4(x, cond_scalar, cond_labels, context=ctx_full)

        x = self.final_linear(x)  # (B[*Z], Hp, Wp, embed_dim)
        x = x.permute(0, 3, 1, 2)  # (B[*Z], embed_dim, Hp, Wp)
        if self.out_layout is not None:
            x = self._decode_columns(x, batch_size)  # (B, out_chans, Hp, Wp)
        else:
            x = self.decoder(x)  # (B, out_chans, Hp, Wp)
        x = x[..., :H, :W]
        if self.use_padding:
            x = self.padding_opt.unpad(x)
        return x
