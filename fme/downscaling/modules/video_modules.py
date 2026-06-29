"""Network building blocks for video (temporal) diffusion downscaling.

Compact 5-D ``(B, C, T, H, W)`` backbone with cBottle-style calendar embedding
and temporal attention, wrapped in EDM preconditioning (Karras 2022).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fme.downscaling.modules.physicsnemo_unets_v2.layers import (
    FourierEmbedding,
    PositionalEmbedding,
)


class NoiseEmbedding(nn.Module):
    """Positional (DDPM++) or Fourier (NCSN++) embedding of the log-noise level,
    mapped through an MLP -- matches HiRO/SongUNet noise conditioning.
    """

    def __init__(self, dim: int, embedding_type: str = "positional"):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("NoiseEmbedding dim must be even.")
        if embedding_type == "fourier":
            self.embed: nn.Module = FourierEmbedding(dim)
        elif embedding_type == "positional":
            self.embed = PositionalEmbedding(dim)
        else:
            raise ValueError(f"unknown noise embedding_type {embedding_type!r}")
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(c_noise.float()))


class FrequencyEmbedding(nn.Module):
    """Periodic sinusoidal encoding on the circle for x in [0, 1)."""

    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.arange(1, self.num_freqs + 1, device=x.device, dtype=x.dtype)
        ang = 2 * math.pi * x.unsqueeze(-1) * k
        return torch.cat([ang.cos(), ang.sin()], dim=-1)


class CalendarEmbedding(nn.Module):
    """cBottle-style absolute-time embedding (diurnal + seasonal).

    ``second_of_day`` is shifted by longitude into local solar time so the
    diurnal phase is correct per pixel.
    """

    def __init__(self, num_freqs: int = 4):
        super().__init__()
        self.embed = FrequencyEmbedding(num_freqs)
        self.out_channels = 4 * num_freqs

    def forward(
        self,
        day_of_year: torch.Tensor,
        second_of_day: torch.Tensor,
        lon: torch.Tensor,
        height: int,
    ) -> torch.Tensor:
        lon = lon.to(second_of_day.device, second_of_day.dtype)
        local = (second_of_day[:, :, None] + lon[None, None, :] * 86400.0 / 360.0)
        local = local % 86400.0
        diurnal = self.embed(local / 86400.0)
        diurnal = diurnal.permute(0, 3, 1, 2)
        diurnal = diurnal.unsqueeze(3).expand(-1, -1, -1, height, -1)

        seasonal = self.embed((day_of_year / 365.25) % 1.0)
        seasonal = seasonal.permute(0, 2, 1)[:, :, :, None, None]
        seasonal = seasonal.expand(-1, -1, -1, height, diurnal.shape[-1])
        return torch.cat([diurnal, seasonal], dim=1)


def _groups(channels: int) -> int:
    g = min(32, channels)
    while channels % g != 0:
        g -= 1
    return g


class PeriodicConv3d(nn.Module):
    """Per-frame spatial Conv3d, periodic in longitude (W) and zero-padded in
    latitude (H) for global lat/lon fields.
    """

    def __init__(self, in_ch, out_ch, kernel=(1, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        self.ph = kernel[1] // 2
        self.pw = kernel[2] // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pw:
            x = torch.cat([x[..., -self.pw :], x, x[..., : self.pw]], dim=-1)
        if self.ph:
            x = F.pad(x, (0, 0, self.ph, self.ph))
        return self.conv(x)


class ResBlock(nn.Module):
    """Per-frame spatial residual block with adaptive GroupNorm (AdaGN)
    noise-level conditioning and ``1/sqrt(2)`` skip scaling (cBottle/EDM style).
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, skip_scale=2**-0.5):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = PeriodicConv3d(in_ch, out_ch)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = PeriodicConv3d(out_ch, out_ch)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.skip_scale = skip_scale

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb)[:, :, None, None, None].chunk(2, dim=1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift))
        return (self.skip(x) + h) * self.skip_scale


class TemporalAttention(nn.Module):
    """Self-attention along the time axis with learned relative-position bias.

    Each pixel attends across all ``T`` frames; output projection is
    zero-initialized so the block starts as near-identity.
    """

    def __init__(self, channels: int, n_heads: int, seq_length: int):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError("channels must be divisible by n_heads.")
        self.n_heads = n_heads
        self.seq_length = seq_length
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.relative_embedding = nn.Parameter(torch.zeros(n_heads, 2 * seq_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        if T > self.seq_length:
            raise ValueError(
                f"clip length {T} exceeds configured seq_length {self.seq_length}."
            )
        cdim = C // self.n_heads
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        def reshape(t):
            t = t.reshape(B, self.n_heads, cdim, T, H, W)
            return t.permute(0, 4, 5, 1, 3, 2).reshape(B * H * W, self.n_heads, T, cdim)

        q, k, v = reshape(q), reshape(k), reshape(v)
        attn = torch.einsum("nhqc,nhkc->nhqk", q, k) / math.sqrt(cdim)
        i = torch.arange(T, device=x.device)
        pairwise = (i[:, None] - i[None, :]) + self.seq_length - 1
        bias = self.relative_embedding[:, pairwise]
        attn = (attn + bias[None]).softmax(dim=-1)
        out = torch.einsum("nhqk,nhkc->nhqc", attn, v)
        out = out.reshape(B, H, W, self.n_heads, T, cdim)
        out = out.permute(0, 3, 5, 4, 1, 2).reshape(B, C, T, H, W)
        return x + self.proj(out)


class SpatialAttention(nn.Module):
    """Multi-head self-attention over the spatial (H*W) plane, per frame.

    Cost is O((H*W)^2), so use at coarser levels only; output projection is
    zero-initialized so the block starts as near-identity.
    """

    def __init__(self, channels: int, n_heads: int):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError("channels must be divisible by n_heads.")
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        cdim = C // self.n_heads
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        def reshape(t):
            t = t.reshape(B, self.n_heads, cdim, T, H * W)
            return t.permute(0, 3, 1, 4, 2).reshape(B * T, self.n_heads, H * W, cdim)

        q, k, v = reshape(q), reshape(k), reshape(v)
        attn = torch.einsum("nhqc,nhkc->nhqk", q, k) / math.sqrt(cdim)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("nhqk,nhkc->nhqc", attn, v)
        out = out.reshape(B, T, self.n_heads, H, W, cdim)
        out = out.permute(0, 2, 5, 1, 3, 4).reshape(B, C, T, H, W)
        return x + self.proj(out)


class _BlockAttn(nn.Module):
    """Optional spatial and/or temporal self-attention for a U-Net block.

    Temporal attention is cheap and runs at every level so time is mixed
    throughout; spatial attention is restricted to coarse levels.
    """

    def __init__(
        self,
        channels: int,
        n_heads: int,
        seq_length: int,
        spatial: bool,
        temporal: bool,
    ):
        super().__init__()
        self.spatial = SpatialAttention(channels, n_heads) if spatial else None
        self.temporal = (
            TemporalAttention(channels, n_heads, seq_length) if temporal else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial is not None:
            x = self.spatial(x)
        if self.temporal is not None:
            x = self.temporal(x)
        return x


class FIRBlur(nn.Module):
    """Depthwise binomial (FIR) low-pass applied per frame, periodic in longitude.

    Anti-aliases before down-sampling and smooths after up-sampling. The kernel
    is a fixed constant buffer, so it is identical on every rank (DDP-safe).
    """

    def __init__(self):
        super().__init__()
        f = torch.tensor([1.0, 2.0, 1.0])
        k = torch.outer(f, f)
        self.register_buffer("kernel", (k / k.sum())[None, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        w = self.kernel.expand(c, 1, 1, 3, 3)
        x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
        x = F.pad(x, (0, 0, 1, 1))
        return F.conv3d(x, w, groups=c)


def _resize_spatial(x: torch.Tensor, size, blur: nn.Module | None = None) -> torch.Tensor:
    """Resize the (H, W) of a (B, C, T, H, W) tensor to ``size`` (keeps T), with
    an optional FIR low-pass applied only when the size actually changes.
    """
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    x = F.interpolate(x, size=(x.shape[2], size[0], size[1]), mode="nearest")
    return blur(x) if blur is not None else x


class VideoUNet(nn.Module):
    """Multi-scale video denoiser: per-frame conv U-Net with spatial + temporal
    self-attention (cBottle-style). Down/up-sampling changes only (H, W); the
    time axis is preserved and handled by ``TemporalAttention``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_length: int,
        model_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 2),
        num_blocks: int = 2,
        n_heads: int = 4,
        attention_levels: tuple[int, ...] = (1, 2),
        temporal_attention_levels: tuple[int, ...] | None = None,
        num_freqs: int = 4,
        noise_embedding_type: str = "positional",
    ):
        super().__init__()
        self.levels = len(channel_mult)
        self.num_blocks = num_blocks
        emb_dim = model_channels
        self.noise_embed = NoiseEmbedding(model_channels, noise_embedding_type)
        self.blur = FIRBlur()
        self.calendar = CalendarEmbedding(num_freqs)
        chs = [model_channels * m for m in channel_mult]
        attn_spatial = set(attention_levels)
        attn_temporal = (
            set(temporal_attention_levels)
            if temporal_attention_levels is not None
            else set(range(self.levels))
        )

        self.in_conv = PeriodicConv3d(in_channels + self.calendar.out_channels, chs[0])

        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_ch = [chs[0]]
        ch = chs[0]
        for level in range(self.levels):
            for _ in range(num_blocks):
                self.enc_blocks.append(ResBlock(ch, chs[level], emb_dim))
                ch = chs[level]
                self.enc_attn.append(
                    _BlockAttn(
                        ch, n_heads, seq_length,
                        level in attn_spatial, level in attn_temporal,
                    )
                )
                skip_ch.append(ch)
            if level < self.levels - 1:
                self.downsamples.append(
                    PeriodicConv3d(ch, ch, (1, 3, 3), stride=(1, 2, 2))
                )
                skip_ch.append(ch)

        self.mid_block1 = ResBlock(ch, ch, emb_dim)
        self.mid_attn = _BlockAttn(ch, n_heads, seq_length, True, True)
        self.mid_block2 = ResBlock(ch, ch, emb_dim)

        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        for level in reversed(range(self.levels)):
            for _ in range(num_blocks + 1):
                sch = skip_ch.pop()
                self.dec_blocks.append(ResBlock(ch + sch, chs[level], emb_dim))
                ch = chs[level]
                self.dec_attn.append(
                    _BlockAttn(
                        ch, n_heads, seq_length,
                        level in attn_spatial, level in attn_temporal,
                    )
                )

        self.out_norm = nn.GroupNorm(_groups(ch), ch)
        self.out_conv = PeriodicConv3d(ch, out_channels)
        nn.init.zeros_(self.out_conv.conv.weight)
        nn.init.zeros_(self.out_conv.conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        c_noise: torch.Tensor,
        day_of_year: torch.Tensor,
        second_of_day: torch.Tensor,
        lon: torch.Tensor,
    ) -> torch.Tensor:
        cal = self.calendar(day_of_year, second_of_day, lon, x.shape[-2])
        emb = self.noise_embed(c_noise)
        h = self.in_conv(torch.cat([x, cal], dim=1))

        skips = [h]
        idx = 0
        for level in range(self.levels):
            for _ in range(self.num_blocks):
                h = self.enc_attn[idx](self.enc_blocks[idx](h, emb))
                skips.append(h)
                idx += 1
            if level < self.levels - 1:
                h = self.downsamples[level](self.blur(h))
                skips.append(h)

        h = self.mid_block2(self.mid_attn(self.mid_block1(h, emb)), emb)

        for idx in range(len(self.dec_blocks)):
            skip = skips.pop()
            h = torch.cat([_resize_spatial(h, skip.shape[-2:], self.blur), skip], dim=1)
            h = self.dec_attn[idx](self.dec_blocks[idx](h, emb))

        return self.out_conv(F.silu(self.out_norm(h)))


class VideoEDMPrecond(nn.Module):
    """EDM preconditioning (Karras 2022) for the 5-D video tensor."""

    def __init__(self, model: VideoUNet, sigma_data: float = 1.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None,
        sigma: torch.Tensor,
        day_of_year: torch.Tensor,
        second_of_day: torch.Tensor,
        lon: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)
        if sigma.dim() == 0:
            sigma = sigma.reshape(1, 1, 1, 1, 1)
        elif sigma.dim() == 1:
            if sigma.shape[0] == x.shape[0]:
                sigma = sigma.reshape(-1, 1, 1, 1, 1)
            elif sigma.shape[0] == x.shape[1]:
                sigma = sigma.reshape(1, -1, 1, 1, 1)
            else:
                sigma = sigma.reshape(-1, 1, 1, 1, 1)
        elif sigma.dim() == 2:
            sigma = sigma.reshape(sigma.shape[0], sigma.shape[1], 1, 1, 1)
        else:
            sigma = sigma.reshape(*sigma.shape[:2], 1, 1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        if sigma.shape[1] == 1:
            c_noise = (sigma.log() / 4).flatten()
        else:
            c_noise = sigma.log().mean(dim=1).flatten() / 4

        arg = c_in * x
        sigma_features = (sigma.log() / 4).expand(
            x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        )
        if condition is not None:
            arg = torch.cat([arg, condition, sigma_features], dim=1)
        else:
            arg = torch.cat([arg, sigma_features], dim=1)
        f_x = self.model(arg, c_noise, day_of_year, second_of_day, lon)
        return c_skip * x + c_out * f_x
