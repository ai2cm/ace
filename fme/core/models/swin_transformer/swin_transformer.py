"""A 2D Swin U-Net backbone for ACE.

This is a 2D adaptation of ArchesWeather's 3D Swin U-Net to ACE's
``(B, C, H, W)`` interface, where all vertical levels are stacked into the
channel dimension. See ``swin_transformer.md`` for the design notes.
"""

import dataclasses
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from fme.core.models.conditional_sfno.layers import Context, ContextConfig

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
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        use_skip: bool = True,
        context_config: ContextConfig | None = None,
        mlp_layer: str = "mlp",
        conditioning: Literal["adaln", "cln"] = "adaln",
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

        ws_h, ws_w = window_size
        self.pad_mult = (ws_h * 2, ws_w * 2)
        H0, W0 = img_shape
        Hp = math.ceil(H0 / self.pad_mult[0]) * self.pad_mult[0]
        Wp = math.ceil(W0 / self.pad_mult[1]) * self.pad_mult[1]
        self.padded_shape = (Hp, Wp)

        if context_config is not None:
            self.embed_dim_scalar = context_config.embed_dim_scalar
            self.embed_dim_labels = context_config.embed_dim_labels
            self.embed_dim_noise = context_config.embed_dim_noise
        else:
            self.embed_dim_scalar = 0
            self.embed_dim_labels = 0
            self.embed_dim_noise = 0

        self.encoder = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.channel_mixer = ChannelMixer(embed_dim)

        d = depth_multiplier
        # DropPath schedule matching ArchesWeather: shallow stages (1 & 4)
        # share the first 2*d rates, deep stages (2 & 3) share the last 6*d.
        dpr = torch.linspace(0, drop_path_rate / d, 8 * d).tolist()
        dpr_shallow = dpr[: 2 * d]
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
        )
        self.final_linear = nn.Linear(decoder_dim, embed_dim, bias=False)
        self.decoder = nn.Conv2d(embed_dim, out_chans, kernel_size=3, padding=1)

    @staticmethod
    def _check(x: torch.Tensor, location: str) -> None:
        if not torch.isfinite(x).all():
            raise ValueError(
                f"NaN/inf after {location} "
                f"(max abs: {x.abs().max().item():.3e})"
            )

    def forward(self, x: torch.Tensor, context: Context | None = None) -> torch.Tensor:
        _, _, H, W = x.shape
        Hp, Wp = self.padded_shape
        pad_h = Hp - H
        pad_w = Wp - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.encoder(x)  # (B, embed_dim, Hp, Wp)
        self._check(x, "encoder")
        x = x.permute(0, 2, 3, 1)  # (B, Hp, Wp, embed_dim)
        x = self.channel_mixer(x)
        self._check(x, "channel_mixer")

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
                if context.labels is None:
                    raise ValueError("labels are required")
                cond_labels = context.labels

        # CLN conditioning: pad and subsample noise to match U-Net resolutions.
        ctx_full: Context | None = context
        ctx_half: Context | None = context
        if self.conditioning == "cln" and self.embed_dim_noise > 0:
            if context is None or context.noise is None:
                raise ValueError(
                    "context.noise is required for a cln-conditioned SwinTransformerNet"
                )
            noise = context.noise  # (B, embed_dim_noise, H, W)
            if pad_h > 0 or pad_w > 0:
                noise = F.pad(noise, (0, pad_w, 0, pad_h))
            noise_half = noise[..., ::2, ::2]
            ctx_full = dataclasses.replace(context, noise=noise)
            ctx_half = dataclasses.replace(context, noise=noise_half)

        x = self.layer1(x, cond_scalar, cond_labels, context=ctx_full)
        self._check(x, "layer1")
        skip = x
        x = self.downsample(x)
        self._check(x, "downsample")
        x = self.layer2(x, cond_scalar, cond_labels, context=ctx_half)
        self._check(x, "layer2")
        x = self.layer3(x, cond_scalar, cond_labels, context=ctx_half)
        self._check(x, "layer3")
        x = self.upsample(x)
        self._check(x, "upsample")
        if self.use_skip:
            x = torch.cat([x, skip], dim=-1)
            self._check(x, "skip_cat")
        x = self.layer4(x, cond_scalar, cond_labels, context=ctx_full)
        self._check(x, "layer4")

        x = self.final_linear(x)  # (B, Hp, Wp, embed_dim)
        self._check(x, "final_linear")
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, Hp, Wp)
        x = self.decoder(x)  # (B, out_chans, Hp, Wp)
        self._check(x, "decoder")
        x = x[..., :H, :W]
        return x
