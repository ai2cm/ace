"""Building blocks for the 2D Swin U-Net used in ACE.

These are a 2D adaptation of ArchesWeather's 3D Swin U-Net. See
``swin_transformer.py`` and the design notes in ``swin_transformer.md`` for
how the pieces fit together. All transformer blocks operate on tensors with
a trailing channel dimension, i.e. shape ``(B, H, W, C)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fme.core.models.conditional_sfno.initialization import trunc_normal_
from fme.core.models.conditional_sfno.layers import DropPath


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
    """Multi-head self-attention within 2D windows with relative position bias.

    Args:
        dim: Number of input channels.
        window_size: ``(ws_h, ws_w)`` height/width of the attention window.
        num_heads: Number of attention heads. Must divide ``dim``.
        qkv_bias: Whether to add a learnable bias to query/key/value.
        attn_drop: Dropout rate on the attention matrix.
        proj_drop: Dropout rate on the output projection.
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
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
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        ws_h, ws_w = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws_h - 1) * (2 * ws_w - 1), num_heads)
        )
        coords_h = torch.arange(ws_h)
        coords_w = torch.arange(ws_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += ws_h - 1
        relative_coords[:, :, 1] += ws_w - 1
        relative_coords[:, :, 0] *= 2 * ws_w - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply windowed attention.

        Args:
            x: Tokens of shape ``(num_windows * B, N, C)`` where
                ``N = ws_h * ws_w``.
            mask: Optional attention mask of shape ``(nW, N, N)`` for the
                shifted-window case.
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

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
    """A single 2D Swin transformer block with optional AdaLN conditioning.

    Two gated sub-blocks: window attention (with ColumnMixer folded into its
    output) followed by an MLP. When conditioning is active, AdaLN scale/shift
    is applied after each norm and a gate scales each residual branch, matching
    ArchesWeather. ColumnMixer has no conditioning and no separate residual.

    Args:
        dim: Number of channels.
        input_resolution: ``(H, W)`` of the (padded) feature map at this stage.
        num_heads: Number of attention heads.
        window_size: ``(ws_h, ws_w)`` attention window.
        shift_size: ``(sh, sw)`` cyclic shift; ``(0, 0)`` for a regular block.
        mlp_ratio: Hidden-dim multiplier for the MLP.
        drop_path: Stochastic-depth rate for this block.
        conditioned: Whether AdaLN conditioning parameters will be supplied.
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
        qkv_bias: Whether attention uses a qkv bias.
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
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention2D(dim, window_size, num_heads, qkv_bias=qkv_bias)
        self.column_mixer = ColumnMixer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _build_mlp(mlp_layer, dim, int(dim * mlp_ratio))

        self.register_buffer("attn_mask", self._build_mask(), persistent=False)

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
        self, x: torch.Tensor, cond_params: CondParams | None = None
    ) -> torch.Tensor:
        H, W = self.input_resolution
        ws_h, ws_w = self.window_size
        sh, sw = self.shift_size
        _, _, _, C = x.shape

        shortcut = x
        x = self.norm1(x)
        if cond_params is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = cond_params
            x = x * (1 + scale_msa) + shift_msa

        if sh > 0 or sw > 0:
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
        x_windows = window_partition_2d(x, ws_h, ws_w).view(-1, ws_h * ws_w, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, ws_h, ws_w, C)
        x = window_reverse_2d(attn_windows, ws_h, ws_w, H, W)
        if sh > 0 or sw > 0:
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))

        # ColumnMixer folded into the window-attention output (no own residual).
        x = x + self.column_mixer(x)

        if cond_params is not None:
            x = shortcut + gate_msa * self.drop_path(x)
            x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
            x = x + self.drop_path(gate_mlp * self.mlp(x_norm))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
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
    """A stack of ``SwinTransformerBlock``s sharing independent AdaLN projections.

    Blocks alternate regular/shifted windows by index. When conditioning is
    active, the layer owns separate ``SiLU -> Linear`` projections for scalar
    and label conditioning, each producing ``6*dim`` AdaLN parameters. Their
    outputs are summed additively before being split into the 6-tuple passed to
    each block, matching the SFNO's independent-additive FiLM pattern.

    Args:
        dim: Number of channels.
        input_resolution: ``(H, W)`` of the (padded) feature map.
        depth: Number of blocks.
        num_heads: Number of attention heads.
        window_size: ``(ws_h, ws_w)`` attention window.
        mlp_ratio: Hidden-dim multiplier for each block's MLP.
        drop_path: Per-block stochastic-depth rates (length ``depth``).
        embed_dim_scalar: Scalar conditioning dimension; 0 disables that path.
        embed_dim_labels: Label conditioning dimension; 0 disables that path.
        mlp_layer: ``"mlp"`` or ``"swiglu"``.
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
    ):
        super().__init__()
        if len(drop_path) != depth:
            raise ValueError(
                f"drop_path has length {len(drop_path)}, expected depth {depth}"
            )
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
                )
                for i in range(depth)
            ]
        )
        # DiT-style identity init: zero the projections so blocks start as
        # identity and conditioning is learned from zero. Both projections are
        # summed additively, so zero-init on each → zero sum → identity.
        if embed_dim_scalar > 0:
            self.adaln_scalar: nn.Module | None = nn.Sequential(
                nn.SiLU(), nn.Linear(embed_dim_scalar, 6 * dim)
            )
            nn.init.zeros_(self.adaln_scalar[1].weight)
            nn.init.zeros_(self.adaln_scalar[1].bias)
        else:
            self.adaln_scalar = None
        if embed_dim_labels > 0:
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
    ) -> torch.Tensor:
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
