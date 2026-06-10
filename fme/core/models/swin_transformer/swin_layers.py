"""Building blocks for the 2D Swin U-Net used in ACE.

These are a 2D adaptation of ArchesWeather's 3D Swin U-Net. See
``swin_transformer.py`` for how the pieces fit together. All transformer
blocks operate on tensors with a trailing channel dimension, i.e. shape
``(B, H, W, C)``.
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from fme.core.models.conditional_sfno.layers import (
    ConditionalLayerNorm,
    Context,
    ContextConfig,
    DropPath,
)


def window_partition_2d(x: torch.Tensor, ws_h: int, ws_w: int) -> torch.Tensor:
    """Partition ``(B, H, W, C)`` into windows ``(B * nW, ws_h, ws_w, C)``."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws_h, ws_h, W // ws_w, ws_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws_h, ws_w, C)
    return windows


def window_reverse_2d(
    windows: torch.Tensor, ws_h: int, ws_w: int, H: int, W: int
) -> torch.Tensor:
    """Inverse of ``window_partition_2d``: windows back to ``(B, H, W, C)``."""
    B = int(windows.shape[0] / (H * W / ws_h / ws_w))
    x = windows.view(B, H // ws_h, W // ws_w, ws_h, ws_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention2D(nn.Module):
    """Multi-head self-attention within 2D windows with continuous position bias.

    Uses a 2-layer MLP (CPB, Swin V2-style) over log-spaced coordinate offsets
    instead of a lookup-table RPB. When ``lat_mean`` is supplied at forward
    time, the longitude offsets are scaled by ``cos(lat)`` so that the bias
    reflects physical arc-length rather than pixel-index distance.

    Args:
        dim: Number of input channels.
        window_size: ``(ws_h, ws_w)`` height/width of the attention window.
        num_heads: Number of attention heads. Must divide ``dim``.
        cpb_hidden_dim: Hidden dimension of the CPB MLP.
        qkv_bias: Whether to add a learnable bias to query/key/value.
        attn_drop: Dropout rate on the attention matrix.
        proj_drop: Dropout rate on the output projection.
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        cpb_hidden_dim: int = 64,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.tau = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        ws_h, ws_w = window_size
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, cpb_hidden_dim),
            nn.ReLU(),
            nn.Linear(cpb_hidden_dim, num_heads),
        )
        nn.init.zeros_(self.cpb_mlp[-1].weight)
        nn.init.zeros_(self.cpb_mlp[-1].bias)

        coords_h = torch.arange(ws_h)
        coords_w = torch.arange(ws_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # (N, N, 2) → (N*N, 2); raw pixel-index offsets, no shift or log yet
        relative_coords_base = relative_coords.view(-1, 2).float()
        self.register_buffer("relative_coords_base", relative_coords_base)
        relative_coords_log = torch.sign(relative_coords_base) * torch.log(
            1.0 + relative_coords_base.abs()
        )
        self.register_buffer("relative_coords_log", relative_coords_log)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        lat_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply windowed attention.

        Args:
            x: Tokens of shape ``(num_windows * B, N, C)`` where
                ``N = ws_h * ws_w``.
            mask: Optional attention mask of shape ``(nW, N, N)`` for the
                shifted-window case.
            lat_mean: Optional ``(nW,)`` tensor of mean latitude in degrees
                for each spatial window. When provided, longitude offsets are
                scaled by ``cos(lat)`` to reflect physical arc-length.
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        norm_q = torch.norm(q, dim=-1, keepdim=True)
        norm_k = torch.norm(k, dim=-1, keepdim=True).transpose(-2, -1)
        attn = (q @ k.transpose(-2, -1)) / (norm_q * norm_k).clamp(min=1e-6)
        attn = attn / self.tau.clamp(min=0.01)

        if lat_mean is None:
            bias = 16.0 * torch.sigmoid(
                self.cpb_mlp(self.relative_coords_log)
            )  # (N*N, num_heads)
            bias = bias.permute(1, 0).reshape(self.num_heads, N, N)
            attn = attn + bias.unsqueeze(0)
        else:
            nW = lat_mean.shape[0]
            lat_rad = lat_mean * (math.pi / 180.0)  # (nW,)
            h_coords = self.relative_coords_base[:, 0]  # (N*N,)
            w_coords = self.relative_coords_base[:, 1].unsqueeze(0) * torch.cos(
                lat_rad
            ).unsqueeze(1)  # (nW, N*N)
            coords = torch.stack(
                [h_coords.unsqueeze(0).expand(nW, -1), w_coords], dim=-1
            )  # (nW, N*N, 2)
            coords_log = torch.sign(coords) * torch.log(1.0 + coords.abs())
            bias = 16.0 * torch.sigmoid(
                self.cpb_mlp(coords_log)
            )  # (nW, N*N, num_heads)
            bias = bias.permute(0, 2, 1).reshape(nW, self.num_heads, N, N)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + bias.unsqueeze(0)
            attn = attn.view(B_, self.num_heads, N, N)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ColumnMixer(nn.Module):
    """Pointwise ``Linear(dim, dim)`` applied at every spatial location.

    The 2D equivalent of ArchesWeather's cross-level attention (``axis_attn``).
    It has no normalization and no residual of its own; its output is folded
    into the window-attention output before the gated shortcut (see
    ``SwinTransformerBlock``).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ChannelMixer(nn.Module):
    """Pointwise ``Linear(dim, dim)`` plus residual, with no normalization.

    The 2D equivalent of ArchesWeather's ``LinVert`` preprocessing layer,
    applied once before all transformer blocks. The absence of a LayerNorm
    matches the original.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x)


class Mlp(nn.Module):
    """Standard two-layer MLP operating on the trailing channel dimension."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    """SwiGLU MLP (ArchesWeather's released-config default)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _build_mlp(mlp_layer: str, dim: int, hidden_features: int) -> nn.Module:
    if mlp_layer == "mlp":
        return Mlp(dim, hidden_features)
    elif mlp_layer == "swiglu":
        return SwiGLU(dim, hidden_features)
    raise ValueError(f"Unknown mlp_layer {mlp_layer!r}, expected 'mlp' or 'swiglu'")


# AdaLN conditioning is passed to a block as a 6-tuple of
# (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp). It is driven
# only by scalar/label embeddings, which produce (B, 1, 1, C) params that are
# broadcast (constant) over space; this path carries no per-pixel or noise term.
CondParams = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class SwinTransformerBlock(nn.Module):
    """A single 2D Swin transformer block with optional AdaLN or CLN conditioning.

    Two sub-blocks: window attention (with ColumnMixer folded into its output)
    followed by an MLP. In ``"adaln"`` mode, AdaLN scale/shift is applied after
    each norm and a gate scales each residual branch. In ``"cln"`` mode,
    ``ConditionalLayerNorm`` is used for both norms and a plain (ungated) residual
    is applied, matching the SFNO's noise-conditioned path.

    Args:
        dim: Number of channels.
        input_resolution: ``(H, W)`` of the (padded) feature map at this stage.
        num_heads: Number of attention heads.
        window_size: ``(ws_h, ws_w)`` attention window.
        shift_size: ``(sh, sw)`` cyclic shift; ``(0, 0)`` for a regular block.
        mlp_ratio: Hidden-dim multiplier for the MLP.
        drop_path: Stochastic-depth rate for this block.
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
        qkv_bias: Whether attention uses a qkv bias.
        conditioning: ``"adaln"`` (default) for native AdaLN, or ``"cln"`` for
            ``ConditionalLayerNorm``-based noise conditioning.
        context_config: Required when ``conditioning="cln"``; passed to each
            ``ConditionalLayerNorm``.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        mlp_layer: str = "mlp",
        qkv_bias: bool = True,
        conditioning: Literal["adaln", "cln"] = "adaln",
        context_config: ContextConfig | None = None,
        cpb_hidden_dim: int = 64,
        lat_coords: torch.Tensor | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.conditioning = conditioning

        if conditioning == "cln":
            if context_config is None:
                raise ValueError("context_config is required for cln conditioning")
            self.norm1: nn.Module = ConditionalLayerNorm(
                dim, input_resolution, context_config
            )
            self.norm2: nn.Module = ConditionalLayerNorm(
                dim, input_resolution, context_config
            )
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.attn = WindowAttention2D(
            dim,
            window_size,
            num_heads,
            cpb_hidden_dim=cpb_hidden_dim,
            qkv_bias=qkv_bias,
        )
        self.column_mixer = ColumnMixer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = _build_mlp(mlp_layer, dim, int(dim * mlp_ratio))

        self.register_buffer("attn_mask", self._build_mask(), persistent=False)
        self.register_buffer("lat_coords", lat_coords, persistent=False)

    def _build_mask(self) -> torch.Tensor | None:
        sh, sw = self.shift_size
        if sh == 0 and sw == 0:
            return None
        H, W = self.input_resolution
        ws_h, ws_w = self.window_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -ws_h), slice(-ws_h, -sh), slice(-sh, None))
        w_slices = (slice(0, -ws_w), slice(-ws_w, -sw), slice(-sw, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition_2d(img_mask, ws_h, ws_w).view(-1, ws_h * ws_w)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        cond_params: CondParams | None = None,
        context: Context | None = None,
    ) -> torch.Tensor:
        H, W = self.input_resolution
        ws_h, ws_w = self.window_size
        sh, sw = self.shift_size
        _, _, _, C = x.shape

        if self.lat_coords is not None:
            lat_shifted = (
                torch.roll(self.lat_coords, -sh) if sh != 0 else self.lat_coords
            )
            nH_win = H // ws_h
            nW_win = W // ws_w
            lat_mean_h = lat_shifted[:H].reshape(nH_win, ws_h).mean(1)  # (nH_win,)
            lat_mean: torch.Tensor | None = (
                lat_mean_h.unsqueeze(1).expand(-1, nW_win).reshape(-1)
            )
        else:
            lat_mean = None

        if self.conditioning == "cln":
            shortcut = x
            if sh > 0 or sw > 0:
                h = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
            else:
                h = x
            h_windows = window_partition_2d(h, ws_h, ws_w).view(-1, ws_h * ws_w, C)
            attn_windows = self.attn(h_windows, mask=self.attn_mask, lat_mean=lat_mean)
            attn_windows = attn_windows.view(-1, ws_h, ws_w, C)
            h = window_reverse_2d(attn_windows, ws_h, ws_w, H, W)
            if sh > 0 or sw > 0:
                h = torch.roll(h, shifts=(sh, sw), dims=(1, 2))
            # ColumnMixer folded in (no own residual).
            h = h + self.column_mixer(h)
            # CLN is channels-first; Swin is channels-last → transpose around norm.
            h_norm = self.norm1(h.permute(0, 3, 1, 2), context).permute(0, 2, 3, 1)
            x = shortcut + self.drop_path(h_norm)
            shortcut = x
            y_norm = self.norm2(self.mlp(x).permute(0, 3, 1, 2), context).permute(
                0, 2, 3, 1
            )
            x = shortcut + self.drop_path(y_norm)
        else:
            shortcut = x
            if cond_params is not None:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    cond_params
                )

            if sh > 0 or sw > 0:
                h = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
            else:
                h = x
            h_windows = window_partition_2d(h, ws_h, ws_w).view(-1, ws_h * ws_w, C)
            attn_windows = self.attn(h_windows, mask=self.attn_mask, lat_mean=lat_mean)
            attn_windows = attn_windows.view(-1, ws_h, ws_w, C)
            h = window_reverse_2d(attn_windows, ws_h, ws_w, H, W)
            if sh > 0 or sw > 0:
                h = torch.roll(h, shifts=(sh, sw), dims=(1, 2))

            # ColumnMixer folded into the window-attention output (no own residual).
            h = h + self.column_mixer(h)

            if cond_params is not None:
                h_norm = self.norm1(h) * (1 + scale_msa) + shift_msa
                x = shortcut + gate_msa * self.drop_path(h_norm)
                shortcut = x
                h_norm = self.norm2(self.mlp(x)) * (1 + scale_mlp) + shift_mlp
                x = shortcut + gate_mlp * self.drop_path(h_norm)
            else:
                x = shortcut + self.drop_path(self.norm1(h))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class PatchMerging(nn.Module):
    """Downsample 2x: concat 2x2 patches, normalize, then project ``4C -> 2C``.

    Norm precedes the linear, matching ArchesWeather's ``DownSample``.
    Operates on ``(B, H, W, C)`` with ``H`` and ``W`` even.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    """Upsample 2x, mapping ``dim -> dim // 2`` channels.

    ``Linear(dim, 2*dim)`` -> pixel-shuffle 2x -> ``LayerNorm`` ->
    ``Linear``, matching ArchesWeather's ``UpSample`` (two linears + norm).
    Operates on ``(B, H, W, C)``.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"PatchExpanding dim ({dim}) must be even")
        out_dim = dim // 2
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.linear = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)  # (B, H, W, 2C)
        x = x.permute(0, 3, 1, 2)  # (B, 2C, H, W)
        x = F.pixel_shuffle(x, 2)  # (B, C/2, 2H, 2W)
        x = x.permute(0, 2, 3, 1)  # (B, 2H, 2W, C/2)
        x = self.norm(x)
        x = self.linear(x)
        return x


class BasicLayer(nn.Module):
    """A stack of ``SwinTransformerBlock``s with AdaLN or CLN conditioning.

    In ``"adaln"`` mode, blocks alternate regular/shifted windows by index and
    the layer owns separate ``SiLU -> Linear`` projections for scalar and label
    conditioning, each producing ``6*dim`` AdaLN parameters.  Their outputs are
    summed additively before being split into the 6-tuple passed to each block.

    In ``"cln"`` mode, each block uses ``ConditionalLayerNorm`` (which owns its
    own per-pixel noise convs) and no AdaLN projections are built.

    Args:
        dim: Number of channels.
        input_resolution: ``(H, W)`` of the (padded) feature map.
        depth: Number of blocks.
        num_heads: Number of attention heads.
        window_size: ``(ws_h, ws_w)`` attention window.
        mlp_ratio: Hidden-dim multiplier for each block's MLP.
        drop_path: Per-block stochastic-depth rates (length ``depth``).
        embed_dim_scalar: Scalar conditioning dimension; 0 disables that path
            (only used in ``"adaln"`` mode).
        embed_dim_labels: Label conditioning dimension; 0 disables that path
            (only used in ``"adaln"`` mode).
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
        conditioning: ``"adaln"`` (default) or ``"cln"``.
        context_config: Required when ``conditioning="cln"``; forwarded to each
            block's ``ConditionalLayerNorm``.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: tuple[int, int],
        mlp_ratio: float,
        drop_path: list[float],
        embed_dim_scalar: int,
        embed_dim_labels: int,
        mlp_layer: str,
        conditioning: Literal["adaln", "cln"] = "adaln",
        context_config: ContextConfig | None = None,
        cpb_hidden_dim: int = 64,
        lat_coords: torch.Tensor | None = None,
    ):
        super().__init__()
        if len(drop_path) != depth:
            raise ValueError(
                f"drop_path has length {len(drop_path)}, expected depth {depth}"
            )
        self.conditioning = conditioning
        ws_h, ws_w = window_size
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=((0, 0) if i % 2 == 0 else (ws_h // 2, ws_w // 2)),
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    mlp_layer=mlp_layer,
                    conditioning=conditioning,
                    context_config=context_config,
                    cpb_hidden_dim=cpb_hidden_dim,
                    lat_coords=lat_coords,
                )
                for i in range(depth)
            ]
        )
        # AdaLN projections: DiT-style zero-init so blocks start as identity.
        # Only built in "adaln" mode.
        if conditioning == "adaln" and embed_dim_scalar > 0:
            self.adaln_scalar: nn.Module | None = nn.Sequential(
                nn.SiLU(), nn.Linear(embed_dim_scalar, 6 * dim)
            )
            nn.init.zeros_(self.adaln_scalar[1].weight)
            nn.init.zeros_(self.adaln_scalar[1].bias)
        else:
            self.adaln_scalar = None
        if conditioning == "adaln" and embed_dim_labels > 0:
            self.adaln_labels: nn.Module | None = nn.Sequential(
                nn.SiLU(), nn.Linear(embed_dim_labels, 6 * dim)
            )
            nn.init.zeros_(self.adaln_labels[1].weight)
            nn.init.zeros_(self.adaln_labels[1].bias)
        else:
            self.adaln_labels = None

    def forward(
        self,
        x: torch.Tensor,
        cond_scalar: torch.Tensor | None = None,
        cond_labels: torch.Tensor | None = None,
        context: Context | None = None,
    ) -> torch.Tensor:
        if self.conditioning == "cln":
            for blk in self.blocks:
                x = blk(x, context=context)
            return x

        cond_params: CondParams | None = None
        raw: torch.Tensor | None = None
        if self.adaln_scalar is not None:
            if cond_scalar is None:
                raise ValueError(
                    "cond_scalar must be provided for a scalar-conditioned BasicLayer"
                )
            raw = self.adaln_scalar(cond_scalar)
        if self.adaln_labels is not None:
            if cond_labels is None:
                raise ValueError(
                    "cond_labels must be provided for a label-conditioned BasicLayer"
                )
            labels_out = self.adaln_labels(cond_labels)
            raw = labels_out if raw is None else raw + labels_out
        if raw is not None:
            params = raw.chunk(6, dim=-1)
            cond_params = tuple(  # type: ignore[assignment]
                p.unsqueeze(1).unsqueeze(1) for p in params
            )
        for blk in self.blocks:
            x = blk(x, cond_params)
        return x
