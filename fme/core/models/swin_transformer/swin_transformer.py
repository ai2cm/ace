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

from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.models.swin_transformer.boundary_padding import TensorPaddingConfig

from .swin_layers import BasicLayer, ChannelMixer, PatchExpanding, PatchMerging


def _padding_index(
    mode: Literal["mirror", "earth"],
    img_shape: tuple[int, int],
    pad_north: int,
    pad_west: int,
    padded_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Data row and column of each point of the padded grid.

    Point ``(i, j)`` of the padded grid sits at data row ``i - pad_north`` and
    data column ``j - pad_west``. Longitude wraps circularly. Past either
    latitude edge, ``"earth"`` crosses the pole (rows reverse and longitude
    rotates by 180 degrees) and ``"mirror"`` reflects about the edge row, for
    any padding size.

    Returns:
        ``(rows, cols)`` index tensors, each of shape ``padded_shape``.
    """
    H, W = img_shape
    Hp, Wp = padded_shape
    rows = torch.arange(Hp) - pad_north
    cols = torch.arange(Wp) - pad_west
    if mode == "earth":
        rows = rows % (2 * H)
        crossed = rows >= H
        rows = torch.where(crossed, 2 * H - 1 - rows, rows)
        shift = crossed.long() * (W // 2)
    else:
        period = max(2 * H - 2, 1)
        rows = rows % period
        rows = torch.where(rows >= H, period - rows, rows)
        shift = torch.zeros_like(rows)
    return (
        rows[:, None].expand(Hp, Wp),
        (cols[None, :] - shift[:, None]) % W,
    )


class SwinTransformerNet(nn.Module):
    """2D Swin U-Net with column interaction and optional AdaLN conditioning.

    The network pads the input to a multiple of ``2**num_levels * window_size
    * patch_size``, encodes it with a strided Conv2d to ``embed_dim`` channels
    on a token grid that is the padded pixel grid divided by ``patch_size``,
    applies a ``ChannelMixer``, then runs a U-Net of ``BasicLayer`` stages
    (depths ``[2, 6, 6, 2] * depth_multiplier``) with one channel-doubling
    downsample / upsample around the two deep stages and an optional skip
    connection, before projecting each token back to its ``patch_size`` block
    of pixels and decoding to ``out_chans`` and cropping to the original shape.

    With ``num_levels > 1``, ``num_levels - 1`` extra U-Net levels are inserted
    between the first stage and the bottleneck: each adds a dim-preserving
    ``PatchMerging`` plus encoder stage on the way down and a mirrored decoder
    stage plus dim-preserving ``PatchExpanding`` (with its own skip) on the way
    up. Because the inserted levels keep the channel dimension at
    ``embed_dim``, the bottleneck stages ``layer2``/``layer3`` keep exactly the
    same parameter count and per-token cost; they simply run on a token grid
    that is ``2**num_levels`` times coarser.

    With the defaults ``patch_size = (1, 1)`` and ``num_levels = 1`` there is
    one token per pixel, the patch embed is a plain 3x3 stride-1 convolution
    and no extra levels are built, so the module is identical to the
    pre-``patch_size`` network in structure, parameter shapes and numerics.

    Args:
        in_chans: Number of input channels.
        out_chans: Number of output channels.
        img_shape: ``(H, W)`` of the input data (before padding).
        embed_dim: Channel dimension of the first/last U-Net stage.
        depth_multiplier: Scales the per-stage depths ``[2, 6, 6, 2]``.
        num_heads: Attention heads for each of the four stages. Extra encoder
            stages (``num_levels > 1``) use ``num_heads[0]`` like the first
            stage and extra decoder stages ``num_heads[3]`` like the last.
        window_size: ``(ws_h, ws_w)`` attention window, measured in tokens.
        patch_size: ``(p_h, p_w)`` pixels per token. The token grid the U-Net
            runs on is the padded pixel grid divided by ``patch_size``, so this
            is the knob for running e.g. a 1-degree (180x360) model at the
            token cost of a 4-degree (45x90) one. Default ``(1, 1)`` is one
            token per pixel.
        num_levels: Number of U-Net downsampling levels above the bottleneck.
            Each level beyond the first inserts a dim-preserving merge plus
            encoder stage (and its mirror decoder stage, expand and skip) so
            the bottleneck stages run on a ``2**num_levels`` times coarser
            token grid at unchanged per-stage cost. The intended 1-degree
            recipe is ``patch_size=(2, 2)`` with ``num_levels=2``, which puts
            the bottleneck of a 180x360 model on the same token grid as a
            4-degree (45x90) model's (nominally 22x45, before padding up to
            whole attention windows). Default 1 is the single-level U-Net and
            builds no extra stages.
        mlp_ratio: Hidden-dim multiplier for block MLPs.
        drop_path_rate: Maximum stochastic-depth rate.
        use_skip: Whether to concatenate each encoder stage's output into the
            matching decoder stage.
        skip_projection: When True, project each
            concatenated ``2 * embed_dim`` skip back to ``embed_dim`` with a
            linear layer so the decoder stages run at ``embed_dim`` channels
            rather than ``2 * embed_dim``. The last decoder stage runs on the
            full-resolution grid, so this removes roughly a quarter of the
            network's FLOPs at the cost of decoder width. Requires
            ``use_skip``.
        context_config: Conditioning configuration.  In ``"adaln"`` mode,
            scalar and label conditioning are applied as independent additive
            AdaLN projections; ``None`` (or both 0) disables AdaLN.  In
            ``"cln"`` mode, ``embed_dim_noise`` drives per-block
            ``ConditionalLayerNorm``; ``embed_dim_scalar`` must be 0.
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
        conditioning: ``"adaln"`` (default) for native per-stage DiT AdaLN, or
            ``"cln"`` for ``ConditionalLayerNorm``-based noise conditioning.
        cpb_hidden_dim: Hidden dimension of the CPB MLPs.
        lat_coords: Optional ``(H,)`` latitudes in degrees of the input rows;
            enables cos-lat scaling of CPB longitude offsets. Each token row
            at every U-Net level uses the mean ``cos(lat)`` of the padded
            pixel rows it covers.
        padding_conf: Optional ``TensorPaddingConfig`` keyword arguments.
            Rows/columns added to reach the pad multiple
            (``2**num_levels * window_size * patch_size``) extend its
            south/east padding; without active padding they use ``"earth"``
            padding, which assumes a global grid. Padded latitudes are those
            of the data in each padded row.
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
        num_levels: int = 1,
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
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if skip_projection and not use_skip:
            raise ValueError("skip_projection=True requires use_skip=True")
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.img_shape = img_shape
        self.use_skip = use_skip
        self.skip_projection = skip_projection
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_levels = num_levels
        self.conditioning = conditioning

        ws_h, ws_w = window_size
        p_h, p_w = patch_size
        # The padded pixel grid must divide evenly into tokens, and the token
        # grid must be a multiple of 2**num_levels * window_size so the
        # coarsest (bottleneck) U-Net stages tile into whole attention windows.
        self.pad_mult = (
            p_h * ws_h * 2**num_levels,
            p_w * ws_w * 2**num_levels,
        )

        padding = TensorPaddingConfig(**(padding_conf or {"activate": False}))
        if padding.activate:
            mode, pad_lat, pad_lon = padding.mode, padding.pad_lat, padding.pad_lon
        else:
            mode, pad_lat, pad_lon = "earth", [0, 0], [0, 0]
        self.pad_offset = (pad_lat[0], pad_lon[0])
        H0 = img_shape[0] + pad_lat[0] + pad_lat[1]
        W0 = img_shape[1] + pad_lon[0] + pad_lon[1]
        Hp = math.ceil(H0 / self.pad_mult[0]) * self.pad_mult[0]
        Wp = math.ceil(W0 / self.pad_mult[1]) * self.pad_mult[1]
        self.padded_shape = (Hp, Wp)
        if mode == "earth" and Hp > img_shape[0] and img_shape[1] % 2 != 0:
            # Crossing a pole rotates longitude by W // 2, not 180 degrees.
            raise ValueError(
                "Earth padding across the poles requires an even number of "
                f"longitudes, got img_shape={tuple(img_shape)}; use "
                '"mirror" padding instead'
            )
        pad_rows, pad_cols = _padding_index(
            mode, img_shape, pad_lat[0], pad_lon[0], (Hp, Wp)
        )
        self.register_buffer("pad_rows", pad_rows, persistent=False)
        self.register_buffer("pad_cols", pad_cols, persistent=False)
        # Token grid the U-Net stages operate on.
        self.token_shape = (Hp // p_h, Wp // p_w)
        Ht, Wt = self.token_shape

        if context_config is not None:
            self.embed_dim_scalar = context_config.embed_dim_scalar
            self.embed_dim_labels = context_config.embed_dim_labels
            self.embed_dim_noise = context_config.embed_dim_noise
        else:
            self.embed_dim_scalar = 0
            self.embed_dim_labels = 0
            self.embed_dim_noise = 0

        # Mean cos(lat) of each token row at every U-Net level: entry k has
        # Ht // 2**k rows, obtained by averaging adjacent rows of entry k - 1.
        # Average cos, not latitude: merged rows can lie at opposite poles.
        cos_lat_levels: list[torch.Tensor | None]
        if lat_coords is not None:
            # The latitude of each padded row is that of the data in it.
            lat_pixel = lat_coords[pad_rows[:, 0]]  # (Hp,)
            cos_lat_level = torch.cos(torch.deg2rad(lat_pixel))
            # Each token row spans p_h pixel rows.
            cos_lat_level = cos_lat_level.reshape(Ht, p_h).mean(dim=1)  # (Ht,)
            cos_lat_levels = [cos_lat_level]
            for _ in range(num_levels):
                cos_lat_level = (cos_lat_level[::2] + cos_lat_level[1::2]) / 2
                cos_lat_levels.append(cos_lat_level)
        else:
            cos_lat_levels = [None] * (num_levels + 1)

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

        def _stage(
            dim: int,
            level: int,
            depth: int,
            heads: int,
            drop_path: list[float],
        ) -> BasicLayer:
            return BasicLayer(
                dim,
                (Ht // 2**level, Wt // 2**level),
                depth,
                heads,
                window_size,
                mlp_ratio,
                drop_path,
                embed_dim_scalar=self.embed_dim_scalar,
                embed_dim_labels=self.embed_dim_labels,
                mlp_layer=mlp_layer,
                conditioning=conditioning,
                context_config=context_config,
                cpb_hidden_dim=cpb_hidden_dim,
                cos_lat=cos_lat_levels[level],
            )

        # Channels after concatenating a level's encoder skip into its decoder.
        if self.skip_projection:
            decoder_dim = embed_dim
        else:
            decoder_dim = 2 * embed_dim if use_skip else embed_dim

        self.layer1 = _stage(embed_dim, 0, 2 * d, num_heads[0], dpr_shallow)
        # Extra U-Net levels. Entry i of each list belongs to level i + 1, so
        # ``extra_encoders[i]`` and ``extra_decoders[i]`` are the two stages
        # that run on the same token grid. All are dim-preserving (embed_dim
        # in, embed_dim out) so the bottleneck stages below keep their width.
        self.extra_merges = nn.ModuleList(
            [PatchMerging(embed_dim, out_dim=embed_dim) for _ in range(num_levels - 1)]
        )
        self.extra_encoders = nn.ModuleList(
            [
                _stage(embed_dim, level, 2 * d, num_heads[0], dpr_shallow)
                for level in range(1, num_levels)
            ]
        )
        self.downsample = PatchMerging(embed_dim)
        self.layer2 = _stage(2 * embed_dim, num_levels, 6 * d, num_heads[1], dpr_deep)
        self.layer3 = _stage(2 * embed_dim, num_levels, 6 * d, num_heads[2], dpr_deep)
        self.upsample = PatchExpanding(2 * embed_dim)  # -> embed_dim, 2x spatial
        self.extra_decoders = nn.ModuleList(
            [
                _stage(decoder_dim, level, 2 * d, num_heads[3], dpr_shallow)
                for level in range(1, num_levels)
            ]
        )
        self.extra_expands = nn.ModuleList(
            [
                PatchExpanding(decoder_dim, out_dim=embed_dim)
                for _ in range(num_levels - 1)
            ]
        )

        # One skip projection per level (entry k for level k), or none.
        self.skip_projs = nn.ModuleList(
            [
                nn.Linear(2 * embed_dim, embed_dim, bias=False)
                for _ in range(num_levels if self.skip_projection else 0)
            ]
        )
        self.layer4 = _stage(decoder_dim, 0, 2 * d, num_heads[3], dpr_shallow)
        # Unpatchify projection: each token predicts the embed_dim features of
        # all p_h * p_w pixels in its patch.
        self.final_linear = nn.Linear(decoder_dim, embed_dim * p_h * p_w, bias=False)
        self.decoder = nn.Conv2d(embed_dim, out_chans, kernel_size=3, padding=1)

    def _merge_skip(
        self, x: torch.Tensor, skip: torch.Tensor, level: int
    ) -> torch.Tensor:
        """Concatenate a level's encoder skip into ``x`` (and project it back
        to ``embed_dim`` with ``skip_projection``); identity without skips.
        """
        if not self.use_skip:
            return x
        x = torch.cat([x, skip], dim=-1)
        if self.skip_projection:
            x = self.skip_projs[level](x)
        return x

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad ``x`` from ``img_shape`` to ``padded_shape``."""
        if tuple(x.shape[-2:]) != tuple(self.img_shape):
            raise ValueError(
                f"Expected spatial shape {tuple(self.img_shape)}, "
                f"got {tuple(x.shape[-2:])}"
            )
        return x[..., self.pad_rows, self.pad_cols]

    def forward(self, x: torch.Tensor, context: Context | None = None) -> torch.Tensor:
        p_h, p_w = self.patch_size
        x = self._pad(x)

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
        # it without transposing activations. Entry k is the context for the
        # stages running at token resolution level k.
        ctx_levels: list[Context | None] = [context] * (self.num_levels + 1)
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
            noise = self._pad(context.noise)  # (B, embed_dim_noise, Hp, Wp)
            noise = noise.permute(0, 2, 3, 1)  # (B, Hp, Wp, embed_dim_noise)
            # Subsample from the pixel grid to the token grid, then halve the
            # resolution once per U-Net level.
            noise = noise[:, ::p_h, ::p_w, :]  # (B, Ht, Wt, embed_dim_noise)
            ctx_levels = [dataclasses.replace(context, noise=noise)]
            for _ in range(self.num_levels):
                noise = noise[:, ::2, ::2, :]
                ctx_levels.append(dataclasses.replace(context, noise=noise))

        x = self.layer1(x, cond_scalar, cond_labels, context=ctx_levels[0])
        # Encoder skips, indexed by level.
        skips = [x]
        for level, (merge, encoder) in enumerate(
            zip(self.extra_merges, self.extra_encoders), start=1
        ):
            x = merge(x)
            x = encoder(x, cond_scalar, cond_labels, context=ctx_levels[level])
            skips.append(x)
        x = self.downsample(x)
        x = self.layer2(x, cond_scalar, cond_labels, context=ctx_levels[-1])
        x = self.layer3(x, cond_scalar, cond_labels, context=ctx_levels[-1])
        x = self.upsample(x)
        for level in range(self.num_levels - 1, 0, -1):
            i = level - 1
            x = self._merge_skip(x, skips[level], level)
            x = self.extra_decoders[i](
                x, cond_scalar, cond_labels, context=ctx_levels[level]
            )
            x = self.extra_expands[i](x)
        x = self._merge_skip(x, skips[0], 0)
        x = self.layer4(x, cond_scalar, cond_labels, context=ctx_levels[0])

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
        p_lat, p_lon = self.pad_offset
        H, W = self.img_shape
        return x[..., p_lat : p_lat + H, p_lon : p_lon + W]
