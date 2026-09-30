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

"""A 2D Swin U-Net backbone for ACE.

This is a 2D adaptation of ArchesWeather's 3D Swin U-Net to ACE's
``(B, C, H, W)`` interface, where all vertical levels are stacked into the
channel dimension.
"""

import dataclasses
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.models.swin_transformer.boundary_padding import TensorPadding

from .swin_layers import BasicLayer, ChannelMixer, PatchExpanding, PatchMerging


class SwinTransformerNet(nn.Module):
    """2D Swin U-Net with column interaction and optional AdaLN conditioning.

    The network pads the input to a multiple of ``2 * window_size *
    patch_size``, encodes it with a strided Conv2d to ``embed_dim`` channels
    on a token grid that is the padded pixel grid divided by ``patch_size``,
    applies a ``ChannelMixer``, then runs a U-Net of ``BasicLayer`` stages
    (depths ``[2, 6, 6, 2] * depth_multiplier``) with one downsample /
    upsample and an optional skip connection, before projecting each token
    back to its ``patch_size`` block of pixels and decoding to ``out_chans``
    and cropping to the original shape.

    With the default ``patch_size = (1, 1)`` there is one token per pixel and
    the patch embed is a plain 3x3 stride-1 convolution, so the module is
    identical to the pre-``patch_size`` network in structure, parameter shapes
    and numerics.

    Args:
        in_chans: Number of input channels.
        out_chans: Number of output channels.
        img_shape: ``(H, W)`` of the input data (before padding).
        embed_dim: Channel dimension of the first/last U-Net stage.
        depth_multiplier: Scales the per-stage depths ``[2, 6, 6, 2]``.
        num_heads: Attention heads for each of the four stages.
        window_size: ``(ws_h, ws_w)`` attention window, measured in tokens.
        patch_size: ``(p_h, p_w)`` pixels per token. The token grid the U-Net
            runs on is the padded pixel grid divided by ``patch_size``, so this
            is the knob for running e.g. a 1-degree (180x360) model at the
            token cost of a 4-degree (45x90) one. Default ``(1, 1)`` is one
            token per pixel.
        mlp_ratio: Hidden-dim multiplier for block MLPs.
        drop_path_rate: Maximum stochastic-depth rate.
        use_skip: Whether to concatenate the layer-1 skip into the decoder.
        skip_projection: When True (and ``use_skip``), project the concatenated
            ``2 * embed_dim`` skip back to ``embed_dim`` with a linear layer so
            the decoder stage runs at ``embed_dim`` channels rather than
            ``2 * embed_dim``. The decoder runs on the full-resolution grid, so
            this removes roughly a quarter of the network's FLOPs at the cost
            of decoder width. Ignored when ``use_skip`` is False.
        context_config: Conditioning configuration.  In ``"adaln"`` mode,
            scalar and label conditioning are applied as independent additive
            AdaLN projections; ``None`` (or both 0) disables AdaLN.  In
            ``"cln"`` mode, ``embed_dim_noise`` drives per-block
            ``ConditionalLayerNorm``; ``embed_dim_scalar`` must be 0.
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
        conditioning: ``"adaln"`` (default) for native per-stage DiT AdaLN, or
            ``"cln"`` for ``ConditionalLayerNorm``-based noise conditioning.
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
        patch_size: tuple[int, int] = (1, 1),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        use_skip: bool = True,
        context_config: ContextConfig | None = None,
        mlp_layer: str = "mlp",
        conditioning: Literal["adaln", "cln"] = "adaln",
        cpb_hidden_dim: int = 64,
        lat_coords: torch.Tensor | None = None,
        padding_conf: dict | None = None,
        skip_projection: bool = False,
    ):
        super().__init__()
        if depth_multiplier < 1:
            raise ValueError(f"depth_multiplier must be >= 1, got {depth_multiplier}")
        if patch_size[0] < 1 or patch_size[1] < 1:
            raise ValueError(f"patch_size entries must be >= 1, got {patch_size}")
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.img_shape = img_shape
        self.use_skip = use_skip
        self.skip_projection = skip_projection and use_skip
        self.window_size = window_size
        self.patch_size = patch_size
        self.conditioning = conditioning

        ws_h, ws_w = window_size
        p_h, p_w = patch_size
        # The padded pixel grid must divide evenly into tokens, and the token
        # grid must be a multiple of 2 * window_size so the half-resolution
        # U-Net stages tile into whole attention windows.
        self.pad_mult = (p_h * ws_h * 2, p_w * ws_w * 2)

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
        # Token grid the U-Net stages operate on.
        self.token_shape = (Hp // p_h, Wp // p_w)
        Ht, Wt = self.token_shape

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
            lat_pixel = (
                torch.cat([lat_coords, lat_coords[-1:].expand(pad_h)])
                if pad_h > 0
                else lat_coords
            )  # (Hp,)
            # Each token row spans p_h pixel rows; use their mean latitude.
            lat_full: torch.Tensor | None = lat_pixel.reshape(Ht, p_h).mean(dim=1)
            lat_half: torch.Tensor | None = (
                lat_full[::2] + lat_full[1::2]  # type: ignore[index]
            ) / 2  # (Ht//2,)
        else:
            lat_full = lat_half = None

        # Overlapping patch embed: a (p_h + 2, p_w + 2) kernel with stride
        # patch_size and padding 1 maps each (p_h, p_w) pixel block to one
        # token while mixing one pixel across each patch edge. At patch_size
        # (1, 1) this is exactly a 3x3 stride-1 convolution.
        self.encoder = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(p_h + 2, p_w + 2),
            stride=(p_h, p_w),
            padding=1,
        )
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
            (Ht, Wt),
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
        )
        self.downsample = PatchMerging(embed_dim)
        self.layer2 = BasicLayer(
            2 * embed_dim,
            (Ht // 2, Wt // 2),
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
        )
        self.layer3 = BasicLayer(
            2 * embed_dim,
            (Ht // 2, Wt // 2),
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
        )
        self.upsample = PatchExpanding(2 * embed_dim)  # -> embed_dim, 2x spatial

        if self.skip_projection:
            self.skip_proj: nn.Module | None = nn.Linear(
                2 * embed_dim, embed_dim, bias=False
            )
            decoder_dim = embed_dim
        else:
            self.skip_proj = None
            decoder_dim = 2 * embed_dim if use_skip else embed_dim
        self.layer4 = BasicLayer(
            decoder_dim,
            (Ht, Wt),
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
        )
        # Unpatchify projection: each token predicts the embed_dim features of
        # all p_h * p_w pixels in its patch.
        self.final_linear = nn.Linear(decoder_dim, embed_dim * p_h * p_w, bias=False)
        self.decoder = nn.Conv2d(embed_dim, out_chans, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, context: Context | None = None) -> torch.Tensor:
        if self.use_padding:
            x = self.padding_opt.pad(x)
        _, _, H, W = x.shape
        Hp, Wp = self.padded_shape
        p_h, p_w = self.patch_size
        pad_h = Hp - H
        pad_w = Wp - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.encoder(x)  # (B, embed_dim, Ht, Wt)
        x = x.permute(0, 2, 3, 1)  # (B, Ht, Wt, embed_dim)
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
                cond_scalar = context.embedding_scalar
            if self.embed_dim_labels > 0:
                cond_labels = context.labels  # may be None; BasicLayer skips when None

        # CLN conditioning: pad and subsample noise to match U-Net resolutions,
        # then move it to channels-last once so every block's CLN can consume
        # it without transposing activations.
        ctx_full: Context | None = context
        ctx_half: Context | None = context
        if self.conditioning == "cln" and self.embed_dim_noise > 0:
            if context is None or context.noise is None:
                raise ValueError(
                    "context.noise is required for a cln-conditioned SwinTransformerNet"
                )
            if context.embedding_pos is not None:
                raise ValueError(
                    "embedding_pos is not supported by a cln-conditioned "
                    "SwinTransformerNet"
                )
            noise = context.noise  # (B, embed_dim_noise, H, W)
            if self.use_padding:
                noise = self.padding_opt.pad(noise)
            if pad_h > 0 or pad_w > 0:
                noise = F.pad(noise, (0, pad_w, 0, pad_h))
            noise = noise.permute(0, 2, 3, 1)  # (B, Hp, Wp, embed_dim_noise)
            # Subsample from the pixel grid to the token grid, then to the
            # half-resolution token grid used by the middle U-Net stages.
            noise = noise[:, ::p_h, ::p_w, :]  # (B, Ht, Wt, embed_dim_noise)
            noise_half = noise[:, ::2, ::2, :]
            ctx_full = dataclasses.replace(context, noise=noise)
            ctx_half = dataclasses.replace(context, noise=noise_half)

        x = self.layer1(x, cond_scalar, cond_labels, context=ctx_full)
        skip = x
        x = self.downsample(x)
        x = self.layer2(x, cond_scalar, cond_labels, context=ctx_half)
        x = self.layer3(x, cond_scalar, cond_labels, context=ctx_half)
        x = self.upsample(x)
        if self.use_skip:
            x = torch.cat([x, skip], dim=-1)
            if self.skip_proj is not None:
                x = self.skip_proj(x)
        x = self.layer4(x, cond_scalar, cond_labels, context=ctx_full)

        x = self.final_linear(x)  # (B, Ht, Wt, embed_dim * p_h * p_w)
        if p_h > 1 or p_w > 1:
            # Anisotropic pixel shuffle. The final_linear output channel axis
            # is interpreted as (p_h, p_w, embed_dim), i.e. the within-patch
            # row index varies slowest and the embedding index fastest, so
            # pixel (i, j) of the patch owns channels
            # [(i * p_w + j) * embed_dim : (i * p_w + j + 1) * embed_dim].
            # F.pixel_shuffle is not usable here: it only supports square
            # upsampling factors.
            B, Ht, Wt, _ = x.shape
            x = x.view(B, Ht, Wt, p_h, p_w, -1)
            x = x.permute(0, 1, 3, 2, 4, 5)  # (B, Ht, p_h, Wt, p_w, embed_dim)
            x = x.reshape(B, Ht * p_h, Wt * p_w, -1)
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, Hp, Wp)
        x = self.decoder(x)  # (B, out_chans, Hp, Wp)
        x = x[..., :H, :W]
        if self.use_padding:
            x = self.padding_opt.unpad(x)
        return x
