"""Network building blocks for video (temporal) diffusion downscaling.

This is a compact backbone that implements the two cBottle physical-time
pathways (see ``cBottle/PHYSICAL_TIMESTEP_REPORT.md``):

* ``CalendarEmbedding`` -- absolute wall-clock time (diurnal + seasonal),
  longitude-shifted to local solar time, concatenated to the input channels.
* ``TemporalAttention`` -- self-attention along the frame (``T``) axis with a
  learned per-head relative-position bias.

Tensors here are 5-D ``(B, C, T, H, W)``. Spatial convolutions use a
``(1, 3, 3)`` kernel so they act per-frame (no implicit time mixing); all genuine
temporal reasoning is delegated to ``TemporalAttention``. The backbone is
intentionally small/single-scale -- it is meant to be correct and swappable, not
state-of-the-art -- and is wrapped in EDM preconditioning (Karras 2022).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseEmbedding(nn.Module):
    """Sinusoidal embedding of the (log) noise level -> MLP."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("NoiseEmbedding dim must be even.")
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=c_noise.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = c_noise.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


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

    Produces ``(B, 4*num_freqs, T, H, W)`` features that are concatenated to the
    input channels. ``second_of_day`` is shifted by longitude into local solar
    time so the diurnal phase is correct per pixel.
    """

    def __init__(self, num_freqs: int = 4):
        super().__init__()
        self.embed = FrequencyEmbedding(num_freqs)
        self.out_channels = 4 * num_freqs

    def forward(
        self,
        day_of_year: torch.Tensor,  # (B, T)
        second_of_day: torch.Tensor,  # (B, T)
        lon: torch.Tensor,  # (W,) degrees
        height: int,
    ) -> torch.Tensor:
        lon = lon.to(second_of_day.device, second_of_day.dtype)
        # local solar time per (frame, longitude)
        local = (second_of_day[:, :, None] + lon[None, None, :] * 86400.0 / 360.0)
        local = local % 86400.0  # (B, T, W)
        diurnal = self.embed(local / 86400.0)  # (B, T, W, 2n)
        diurnal = diurnal.permute(0, 3, 1, 2)  # (B, 2n, T, W)
        diurnal = diurnal.unsqueeze(3).expand(-1, -1, -1, height, -1)

        seasonal = self.embed((day_of_year / 365.25) % 1.0)  # (B, T, 2n)
        seasonal = seasonal.permute(0, 2, 1)[:, :, :, None, None]
        seasonal = seasonal.expand(-1, -1, -1, height, diurnal.shape[-1])
        return torch.cat([diurnal, seasonal], dim=1)  # (B, 4n, T, H, W)


class ResBlock(nn.Module):
    """Per-frame spatial residual block with noise-level conditioning."""

    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        groups = min(32, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))
        self.emb = nn.Linear(emb_dim, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TemporalAttention(nn.Module):
    """Self-attention along the time axis with learned relative-position bias.

    Each spatial pixel attends across all ``T`` frames; the per-head bias depends
    only on the signed frame offset (cBottle ``TemporalAttention``). The output
    projection is zero-initialized so the block starts as near-identity.
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
        bias = self.relative_embedding[:, pairwise]  # (n_heads, T, T)
        attn = (attn + bias[None]).softmax(dim=-1)
        out = torch.einsum("nhqk,nhkc->nhqc", attn, v)
        out = out.reshape(B, H, W, self.n_heads, T, cdim)
        out = out.permute(0, 3, 5, 4, 1, 2).reshape(B, C, T, H, W)
        return x + self.proj(out)


class VideoUNet(nn.Module):
    """Compact single-scale video denoiser network.

    Input is the (preconditioned) noisy clip concatenated with conditioning
    channels; ``CalendarEmbedding`` features are concatenated internally.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_length: int,
        model_channels: int = 64,
        n_heads: int = 4,
        num_freqs: int = 4,
    ):
        super().__init__()
        self.noise_embed = NoiseEmbedding(model_channels)
        self.calendar = CalendarEmbedding(num_freqs)
        self.in_conv = nn.Conv3d(
            in_channels + self.calendar.out_channels,
            model_channels,
            (1, 3, 3),
            padding=(0, 1, 1),
        )
        self.block1 = ResBlock(model_channels, model_channels)
        self.temporal_attention = TemporalAttention(model_channels, n_heads, seq_length)
        self.block2 = ResBlock(model_channels, model_channels)
        self.out_norm = nn.GroupNorm(min(32, model_channels), model_channels)
        self.out_conv = nn.Conv3d(
            model_channels, out_channels, (1, 3, 3), padding=(0, 1, 1)
        )
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,  # (B, C_in, T, H, W)
        c_noise: torch.Tensor,  # (B,)
        day_of_year: torch.Tensor,  # (B, T)
        second_of_day: torch.Tensor,  # (B, T)
        lon: torch.Tensor,  # (W,)
    ) -> torch.Tensor:
        height = x.shape[-2]
        calendar = self.calendar(day_of_year, second_of_day, lon, height)
        h = torch.cat([x, calendar], dim=1)
        emb = self.noise_embed(c_noise)
        h = self.in_conv(h)
        h = self.block1(h, emb)
        h = self.temporal_attention(h)
        h = self.block2(h, emb)
        return self.out_conv(F.silu(self.out_norm(h)))


class VideoEDMPrecond(nn.Module):
    """EDM preconditioning (Karras 2022) for the 5-D video tensor."""

    def __init__(self, model: VideoUNet, sigma_data: float = 1.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(
        self,
        x: torch.Tensor,  # (B, C_out, T, H, W) noisy residual
        condition: torch.Tensor | None,  # (B, C_cond, T, H, W)
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
