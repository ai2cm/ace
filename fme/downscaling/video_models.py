"""Endpoint-conditioned video interpolation diffusion (temporal-only).

Diffuses the residual over a temporal linear interpolation of observed endpoints;
only interior frames are denoised. Operates on the fine clip only.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.packer import Packer
from fme.core.rand import randn_like
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import PairedVideoBatchData
from fme.downscaling.models import ModelOutputs
from fme.downscaling.modules.video_modules import VideoEDMPrecond, VideoUNet
from fme.downscaling.noise import (
    LogNormalNoiseDistribution,
    LogUniformNoiseDistribution,
    NoiseDistribution,
    brownian_bridge_mixing_matrix,
)
from fme.downscaling.requirements import DataRequirements

CHANNEL_AXIS = 1


def _interior_mask(n_times: int, device: torch.device) -> torch.Tensor:
    """(1, 1, T, 1, 1) mask: 1 on interior frames, 0 on observed endpoints."""
    mask = torch.ones(n_times, device=device)
    mask[0] = 0.0
    mask[-1] = 0.0
    return mask.reshape(1, 1, n_times, 1, 1)


def _linear_interp_endpoints(field: torch.Tensor) -> torch.Tensor:
    """Temporal linear interpolation of the two endpoints along the time axis."""
    n_times = field.shape[-3]
    shape = [1] * field.dim()
    shape[-3] = n_times
    w = torch.linspace(0.0, 1.0, n_times, device=field.device).reshape(shape)
    x0 = field[..., 0:1, :, :]
    xT = field[..., n_times - 1 : n_times, :, :]
    return (1 - w) * x0 + w * xT


@dataclasses.dataclass
class VideoDiffusionModelConfig:
    """Configuration for the temporal-interpolation video diffusion model."""

    out_names: list[str]
    n_timesteps: int
    normalization: NormalizationConfig | None = None
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    churn: float = 0.0
    num_diffusion_generation_steps: int = 18
    model_channels: int = 64
    n_heads: int = 4
    num_freqs: int = 4
    # log-noise embedding for the simple backbone: "positional" or "fourier".
    noise_embedding_type: str = "positional"
    # Multi-scale U-Net: one channel multiplier per resolution level (0 = finest).
    channel_mult: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 2])
    num_blocks: int = 2
    # spatial attention levels; temporal attention defaults to all levels (None).
    attention_levels: list[int] = dataclasses.field(default_factory=lambda: [1, 2])
    temporal_attention_levels: list[int] | None = None
    # backbone: "simple" (periodic VideoUNet) or "songunet" (SongUNetv2).
    backbone: str = "simple"
    img_resolution: list[int] | None = None  # [H, W], required for songunet
    attn_resolutions: list[int] | None = None
    training_noise_distribution: (
        LogNormalNoiseDistribution | LogUniformNoiseDistribution | None
    ) = None
    training_noise_distributions: (
        dict[str, LogNormalNoiseDistribution | LogUniformNoiseDistribution] | None
    ) = None
    sigma_min_by_channel: dict[str, float] | None = None
    sigma_max_by_channel: dict[str, float] | None = None
    loss_weight_exponent: float = 1.0
    # Channels modeled in log space via log1p(x*scale); maps channel to scale.
    log_transform_channels: dict[str, float] | None = None
    # Temporal correlation of the residual noise: "independent" (per-frame white
    # noise, default) or "brownian_bridge" (endpoint-pinned time-correlated noise).
    temporal_noise_correlation: str = "independent"

    def __post_init__(self):
        if self.n_timesteps < 3:
            raise ValueError(
                "Video interpolation needs at least 3 frames (2 endpoints + 1 "
                f"interior), got n_timesteps={self.n_timesteps}."
            )
        for field_name in (
            "training_noise_distributions",
            "sigma_min_by_channel",
            "sigma_max_by_channel",
        ):
            values = getattr(self, field_name)
            if values is None:
                continue
            unknown = set(values) - set(self.out_names)
            if unknown:
                raise ValueError(
                    f"{field_name} contains channels not in out_names: "
                    f"{sorted(unknown)}"
                )
        if self.training_noise_distributions is not None:
            missing = set(self.out_names) - set(self.training_noise_distributions)
            if missing:
                raise ValueError(
                    "training_noise_distributions must specify every output "
                    f"channel; missing {sorted(missing)}"
                )
        if (self.sigma_min_by_channel is None) != (self.sigma_max_by_channel is None):
            raise ValueError(
                "sigma_min_by_channel and sigma_max_by_channel must be specified "
                "together."
            )
        if (
            self.training_noise_distribution is not None
            and self.training_noise_distributions is not None
        ):
            raise ValueError(
                "Specify only one of training_noise_distribution or "
                "training_noise_distributions."
            )
        unknown_log = set(self.log_transform_channels or {}) - set(self.out_names)
        if unknown_log:
            raise ValueError(
                "log_transform_channels contains channels not in out_names: "
                f"{sorted(unknown_log)}"
            )
        if any(lvl not in range(len(self.channel_mult)) for lvl in self.attention_levels):
            raise ValueError(
                f"attention_levels {self.attention_levels} must index into "
                f"channel_mult (0..{len(self.channel_mult) - 1})."
            )
        for m in self.channel_mult:
            if (self.model_channels * m) % self.n_heads != 0:
                raise ValueError(
                    f"model_channels*{m}={self.model_channels * m} not divisible "
                    f"by n_heads={self.n_heads}."
                )
        if self.temporal_noise_correlation not in ("independent", "brownian_bridge"):
            raise ValueError(
                "temporal_noise_correlation must be 'independent' or "
                f"'brownian_bridge', got {self.temporal_noise_correlation}."
            )
        if self.backbone not in ("simple", "songunet"):
            raise ValueError(f"backbone must be 'simple' or 'songunet', got {self.backbone}.")
        if self.noise_embedding_type not in ("positional", "fourier"):
            raise ValueError(
                "noise_embedding_type must be 'positional' or 'fourier', got "
                f"{self.noise_embedding_type}."
            )
        if self.backbone == "songunet" and self.img_resolution is None:
            raise ValueError("songunet backbone requires img_resolution [H, W].")

    @property
    def noise_distribution(self) -> NoiseDistribution:
        if self.training_noise_distribution is not None:
            return self.training_noise_distribution
        return LogNormalNoiseDistribution(p_mean=-1.2, p_std=1.2)

    def sample_training_noise(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.training_noise_distributions is None:
            return self.noise_distribution.sample(batch_size, device)
        sigma_by_channel = [
            self.training_noise_distributions[name].sample(batch_size, device)
            for name in self.out_names
        ]
        return torch.cat(sigma_by_channel, dim=1)

    def generation_sigma_bounds(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_min = torch.tensor(
            [
                self.sigma_min
                if self.sigma_min_by_channel is None
                else self.sigma_min_by_channel.get(name, self.sigma_min)
                for name in self.out_names
            ],
            dtype=torch.float32,
            device=device,
        )
        sigma_max = torch.tensor(
            [
                self.sigma_max
                if self.sigma_max_by_channel is None
                else self.sigma_max_by_channel.get(name, self.sigma_max)
                for name in self.out_names
            ],
            dtype=torch.float32,
            device=device,
        )
        return sigma_min, sigma_max

    @property
    def data_requirements(self) -> "DataRequirements":
        # Temporal-only: fine == coarse; clip length comes from the data config.
        return DataRequirements(
            fine_names=self.out_names,
            coarse_names=self.out_names,
            n_timesteps=1,
            use_fine_topography=False,
        )

    def build(
        self, normalizer: StandardNormalizer | None = None
    ) -> "VideoDiffusionModel":
        if normalizer is None:
            if self.normalization is None:
                raise ValueError(
                    "Either `normalization` config or a prebuilt `normalizer` "
                    "must be provided."
                )
            normalizer = self.normalization.build(self.out_names)

        n_channels = len(self.out_names)
        # noisy residual (C) + endpoint values (C) + mask (1) + log-sigma (C)
        in_channels = 3 * n_channels + 1
        if self.backbone == "songunet":
            from fme.downscaling.modules.video_song_unet import VideoSongUNet

            default_attn = self.img_resolution[0] >> (len(self.channel_mult) - 1)
            net = VideoSongUNet(
                in_channels=in_channels,
                out_channels=n_channels,
                img_resolution=self.img_resolution,
                seq_length=self.n_timesteps,
                model_channels=self.model_channels,
                channel_mult=tuple(self.channel_mult),
                num_blocks=self.num_blocks,
                n_heads=self.n_heads,
                attn_resolutions=tuple(self.attn_resolutions or [default_attn]),
                num_freqs=self.num_freqs,
            )
        else:
            net = VideoUNet(
                in_channels=in_channels,
                out_channels=n_channels,
                seq_length=self.n_timesteps,
                model_channels=self.model_channels,
                channel_mult=tuple(self.channel_mult),
                num_blocks=self.num_blocks,
                n_heads=self.n_heads,
                attention_levels=tuple(self.attention_levels),
                temporal_attention_levels=(
                    None
                    if self.temporal_attention_levels is None
                    else tuple(self.temporal_attention_levels)
                ),
                num_freqs=self.num_freqs,
                noise_embedding_type=self.noise_embedding_type,
            )
        module = VideoEDMPrecond(net, sigma_data=1.0)
        return VideoDiffusionModel(self, module, normalizer, self.out_names)


class VideoDiffusionModel:
    def __init__(
        self,
        config: VideoDiffusionModelConfig,
        module: torch.nn.Module,
        normalizer: StandardNormalizer,
        out_names: list[str],
    ):
        self.config = config
        self.sigma_data = 1.0
        dist = Distributed.get_instance()
        self.module = dist.wrap_module(module.to(get_device()))
        self.normalizer = normalizer
        self.out_names = out_names
        self.packer = Packer(out_names)
        self.n_timesteps = config.n_timesteps
        self.log_transform_channels = dict(config.log_transform_channels or {})
        if config.temporal_noise_correlation == "brownian_bridge":
            self._noise_mixing: torch.Tensor | None = brownian_bridge_mixing_matrix(
                config.n_timesteps
            ).to(get_device())
        else:
            self._noise_mixing = None

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([self.module])

    def _sample_residual_noise(self, like: torch.Tensor) -> torch.Tensor:
        """White noise shaped like ``like`` (B, C, T, H, W), temporally correlated
        with the Brownian-bridge kernel when configured, else independent."""
        noise = randn_like(like)
        if self._noise_mixing is None:
            return noise
        mixing = self._noise_mixing.to(device=noise.device, dtype=noise.dtype)
        return torch.einsum("ti,bcihw->bcthw", mixing, noise)

    def _pack_normalized(self, data: TensorMapping) -> torch.Tensor:
        selected = {
            k: torch.log1p(data[k].clamp(min=0.0) * scale)
            if (scale := self.log_transform_channels.get(k))
            else data[k]
            for k in self.out_names
        }
        normalized = self.normalizer.normalize(selected)
        return self.packer.pack(normalized, axis=CHANNEL_AXIS)

    def _denormalize_invert(self, packed: torch.Tensor) -> TensorDict:
        """Denormalize and invert any log1p transform back to physical units."""
        data = self.normalizer.denormalize(
            self.packer.unpack(packed, axis=CHANNEL_AXIS)
        )
        return {
            k: (torch.expm1(v) / scale).clamp(min=0.0)
            if (scale := self.log_transform_channels.get(k))
            else v
            for k, v in data.items()
        }

    def _conditioning(
        self, clip: torch.Tensor, interior_mask: torch.Tensor
    ) -> torch.Tensor:
        """Observed-endpoint values + a binary observed mask channel."""
        observed = 1.0 - interior_mask
        observed_values = clip * observed
        mask_channel = observed.expand(clip.shape[0], 1, -1, clip.shape[-2], clip.shape[-1])
        return torch.cat([observed_values, mask_channel], dim=CHANNEL_AXIS)

    def _calendar_inputs(self, batch_fine):
        lon = batch_fine.latlon_coordinates.lon
        if lon.dim() == 2:  # all members identical, use first
            lon = lon[0]
        return (
            batch_fine.day_of_year.to(get_device()),
            batch_fine.second_of_day.to(get_device()),
            lon.to(get_device()),
        )

    def train_on_batch(self, batch: PairedVideoBatchData, optimizer) -> ModelOutputs:
        fine = batch.fine
        clip = self._pack_normalized(fine.data)
        batch_size, _, n_times, _, _ = clip.shape
        baseline = _linear_interp_endpoints(clip)
        residual = clip - baseline  # ~0 at endpoints by construction

        interior = _interior_mask(n_times, clip.device)
        condition = self._conditioning(clip, interior)
        day_of_year, second_of_day, lon = self._calendar_inputs(fine)

        sigma = self.config.sample_training_noise(batch_size, clip.device)
        sigma = sigma.reshape(batch_size, -1, 1, 1, 1)
        # Gaussian noise on interior frames only
        noised = residual + self._sample_residual_noise(residual) * sigma * interior

        denoised = self.module(
            noised, condition, sigma, day_of_year, second_of_day, lon
        )

        weight = (
            (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        ) ** self.config.loss_weight_exponent
        sq_err = (denoised - residual) ** 2 * interior
        n_interior_elems = interior.expand_as(sq_err).sum()
        loss = (weight * sq_err).sum() / n_interior_elems

        optimizer.accumulate_loss(loss)
        optimizer.step_weights()

        with torch.no_grad():
            weighted_sq_err = sq_err * weight
            per_sample_denominator = (
                interior.expand(batch_size, 1, n_times, *clip.shape[-2:])
                .sum(dim=(-3, -2, -1))
                .clamp(min=1)
            )
            # pin observed endpoints (residual is 0 there)
            full_norm = baseline + denoised * interior
            prediction = self._denormalize_invert(full_norm)
            channel_losses = {
                name: weighted_sq_err[:, i].sum() / per_sample_denominator.sum()
                for i, name in enumerate(self.out_names)
            }
            per_sample_channel_loss = {
                name: weighted_sq_err[:, i].sum(dim=(-3, -2, -1))
                / per_sample_denominator.flatten()
                for i, name in enumerate(self.out_names)
            }
        target = {k: fine.data[k] for k in self.out_names}
        return ModelOutputs(
            prediction=prediction,
            target=target,
            loss=loss,
            channel_losses=channel_losses,
            sigma=(
                sigma.flatten()
                if sigma.shape[1] == 1
                else sigma.squeeze(-1).squeeze(-1).squeeze(-1)
            ),
            per_sample_channel_loss=per_sample_channel_loss,
        )

    @torch.no_grad()
    def generate(self, batch: PairedVideoBatchData, n_samples: int = 1) -> TensorDict:
        fine = batch.fine
        clip = self._pack_normalized(fine.data)
        batch_size, n_channels, n_times, height, width = clip.shape
        baseline = _linear_interp_endpoints(clip)
        interior = _interior_mask(n_times, clip.device)
        condition = self._conditioning(clip, interior)
        day_of_year, second_of_day, lon = self._calendar_inputs(fine)

        def repeat(t):
            return t.repeat_interleave(n_samples, dim=0)

        condition = repeat(condition)
        baseline = repeat(baseline)
        day_of_year = repeat(day_of_year)
        second_of_day = repeat(second_of_day)
        interior_b = interior.expand(batch_size * n_samples, n_channels, n_times, 1, 1)

        latents = self._sample_residual_noise(
            torch.empty(
                batch_size * n_samples, n_channels, n_times, height, width,
                device=clip.device,
            )
        )
        sigma_min, sigma_max = self.config.generation_sigma_bounds(clip.device)
        residual = _video_edm_sample(
            self.module,
            latents,
            condition,
            interior_b,
            day_of_year,
            second_of_day,
            lon,
            num_steps=self.config.num_diffusion_generation_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            s_churn=self.config.churn,
            noise_mixing=self._noise_mixing,
        )
        full_norm = baseline + residual
        generated = self._denormalize_invert(full_norm)
        # (B*n, T, H, W) -> (B, n, T, H, W)
        return {
            k: v.reshape(batch_size, n_samples, *v.shape[1:])
            for k, v in generated.items()
        }

    def get_state(self) -> Mapping[str, Any]:
        return {
            "config": dataclasses.asdict(self.config),
            "module": self.module.state_dict(),
        }


def _video_edm_sample(
    net: torch.nn.Module,
    latents: torch.Tensor,
    condition: torch.Tensor,
    interior_mask: torch.Tensor,
    day_of_year: torch.Tensor,
    second_of_day: torch.Tensor,
    lon: torch.Tensor,
    num_steps: int,
    sigma_min: float | torch.Tensor,
    sigma_max: float | torch.Tensor,
    rho: float = 7.0,
    s_churn: float = 0.0,
    noise_mixing: torch.Tensor | None = None,
) -> torch.Tensor:
    """EDM stochastic sampler that keeps the observed endpoints pinned (re-zeroed
    after every update). When ``noise_mixing`` is given, the stochastic churn noise
    is temporally correlated with that mixing matrix (matching the training noise)."""
    compute_dtype = torch.float32 if latents.device.type == "mps" else torch.float64
    sigma_min_t = torch.as_tensor(
        sigma_min, dtype=compute_dtype, device=latents.device
    ).reshape(1, -1, 1, 1, 1)
    sigma_max_t = torch.as_tensor(
        sigma_max, dtype=compute_dtype, device=latents.device
    ).reshape(1, -1, 1, 1, 1)
    step = torch.arange(
        num_steps, dtype=compute_dtype, device=latents.device
    ).reshape(-1, 1, 1, 1, 1)
    t = (
        sigma_max_t ** (1 / rho)
        + step
        / (num_steps - 1)
        * (sigma_min_t ** (1 / rho) - sigma_max_t ** (1 / rho))
    ) ** rho
    t = torch.cat([t, torch.zeros_like(t[:1])])
    mask = interior_mask.to(compute_dtype)

    def denoise(x, sigma):
        # broadcast per-channel sigma to (B, C) for the precond
        sigma_bc = sigma.reshape(1, -1).expand(x.shape[0], -1).to(torch.float32)
        out = net(
            x.to(torch.float32),
            condition,
            sigma_bc,
            day_of_year,
            second_of_day,
            lon,
        )
        return out.to(compute_dtype)

    def churn_noise(x):
        noise = torch.randn_like(x)
        if noise_mixing is None:
            return noise
        mixing = noise_mixing.to(device=noise.device, dtype=noise.dtype)
        return torch.einsum("ti,bcihw->bcthw", mixing, noise)

    x = latents.to(compute_dtype) * t[0] * mask
    for i, (t_cur, t_next) in enumerate(zip(t[:-1], t[1:])):
        gamma = s_churn / num_steps
        t_hat = t_cur + gamma * t_cur
        x_hat = x + (t_hat**2 - t_cur**2).clamp(min=0).sqrt() * churn_noise(x)
        x_hat = x_hat * mask
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x = (x_hat + (t_next - t_hat) * d_cur) * mask
        if i < num_steps - 1:
            d_prime = (x - denoise(x, t_next)) / t_next
            x = (x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)) * mask
    return x.to(latents.dtype)
