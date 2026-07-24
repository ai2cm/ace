"""Endpoint-conditioned video interpolation diffusion, with optional spatial
downscaling.

Diffuses the residual over a per-frame baseline. By default
(``coarse_normalization`` unset) the baseline is the temporal linear
interpolation of the observed fine endpoints, ``batch.coarse`` is ignored,
and only interior frames are denoised -- the two endpoints are pinned exactly
to their true fine values (temporal-only, prior behavior).

When ``coarse_normalization`` is set, the model also conditions on a
coarser-resolution clip of the same frames (``batch.coarse``), and the
baseline becomes that clip bicubic-upsampled to the fine grid:
- ``endpoints_observed=True`` (default): bias-corrected so it still exactly
  reproduces the true fine endpoints -- combining CorrDiff-style spatial
  residual diffusion with the endpoint-conditioned temporal residual.
- ``endpoints_observed=False``: pure coarse-to-fine mode. No frame has
  fine-resolution truth available (e.g. deploying on a continuously-running
  coarse emulator with no sparse high-res anchors): the baseline is exactly
  ``upsample(coarse)`` with no fine value entering it, nothing is pinned, and
  every frame -- including the former "endpoints" -- is diffused.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.packer import Packer
from fme.core.rand import randn_like
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import (
    BatchedLatLonCoordinates,
    PairedVideoBatchData,
    VideoBatchData,
    adjust_fine_coord_range,
)
from fme.downscaling.metrics_and_maths import interpolate
from fme.downscaling.models import ModelOutputs
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.video_modules import VideoEDMPrecond, VideoUNet
from fme.downscaling.noise import (
    LogNormalNoiseDistribution,
    LogUniformNoiseDistribution,
    NoiseDistribution,
    brownian_bridge_mixing_matrix,
    condition_with_noise_for_training,
    ou_mixing_matrix,
    rbf_mixing_matrix,
    uniform_frame_times,
)
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.samplers import stochastic_sampler as edm_sampler

CHANNEL_AXIS = 1


def _interior_mask(
    n_times: int, device: torch.device, endpoints_observed: bool = True
) -> torch.Tensor:
    """(1, 1, T, 1, 1) mask: 1 on generated frames, 0 on pinned observed
    endpoints. With ``endpoints_observed=False`` no frame is pinned (all
    ones) -- every frame is generated, matching the pure coarse-to-fine mode.
    """
    mask = torch.ones(n_times, device=device)
    if endpoints_observed:
        mask[0] = 0.0
        mask[-1] = 0.0
    return mask.reshape(1, 1, n_times, 1, 1)


def _linear_interp_endpoints(
    field: torch.Tensor, tau: torch.Tensor | None = None
) -> torch.Tensor:
    """Temporal linear interpolation of the two endpoints along the time axis.

    ``tau`` gives the normalized time of each frame in ``[0, 1]`` (endpoints at 0
    and 1). When omitted the frames are assumed uniformly spaced, reproducing the
    ``linspace`` weights; passing the true ``tau`` lets the baseline stay correct
    for a non-uniform subset of frames.
    """
    n_times = field.shape[-3]
    shape = [1] * field.dim()
    shape[-3] = n_times
    if tau is None:
        w = torch.linspace(0.0, 1.0, n_times, device=field.device)
    else:
        w = tau.to(device=field.device, dtype=field.dtype)
    w = w.reshape(shape)
    x0 = field[..., 0:1, :, :]
    xT = field[..., n_times - 1 : n_times, :, :]
    return (1 - w) * x0 + w * xT


def _upsample_coarse_clip(
    coarse: torch.Tensor, fine_hw: tuple[int, int]
) -> torch.Tensor:
    """Bicubic-upsample a ``(B, C, T, Hc, Wc)`` coarse clip to the fine ``(H, W)``,
    frame by frame (reuses the same 2D interpolation as the spatial-only
    ``DiffusionModel`` in ``models.py``, so the two spatial-downscaling paths
    match exactly).
    """
    batch, channels, n_times, height, width = coarse.shape
    if (height, width) == tuple(fine_hw):
        return coarse
    fine_h, fine_w = fine_hw
    if (
        fine_h % height != 0
        or fine_w % width != 0
        or fine_h // height != fine_w // width
    ):
        raise ValueError(
            f"Fine shape {fine_hw} must be an integer multiple of the coarse "
            f"shape {(height, width)} with equal lat/lon scale factor."
        )
    scale_factor = fine_h // height
    # coarse is (B, C, T, H, W): C and T are not adjacent in memory, so a plain
    # reshape to (B*T, C, H, W) would interleave channels from different
    # timesteps instead of grouping each frame's own channels. Move T next to
    # B first so the merge is meaningful, then move it back afterwards.
    flat = coarse.permute(0, 2, 1, 3, 4).reshape(
        batch * n_times, channels, height, width
    )
    upsampled = interpolate(flat, scale_factor)
    return upsampled.reshape(batch, n_times, channels, fine_h, fine_w).permute(
        0, 2, 1, 3, 4
    )


def _fold_endpoint_frames(clip: torch.Tensor) -> torch.Tensor:
    """``(B, C, T, H, W) -> (B*2, C, H, W)``: the two endpoint frames (index 0
    and -1), folded into the batch dim for a 2D per-frame network call. T and
    C are not adjacent in memory (same reasoning as ``_upsample_coarse_clip``),
    so permute before reshape rather than a naive ``.reshape``.
    """
    batch, channels, _, height, width = clip.shape
    endpoints = clip[:, :, [0, -1]]  # (B, C, 2, H, W)
    return endpoints.permute(0, 2, 1, 3, 4).reshape(batch * 2, channels, height, width)


def _unfold_endpoint_frames(folded: torch.Tensor, batch: int) -> torch.Tensor:
    """Inverse of ``_fold_endpoint_frames``: ``(B*2, C, H, W) -> (B, C, 2, H, W)``."""
    _, channels, height, width = folded.shape
    return folded.reshape(batch, 2, channels, height, width).permute(0, 2, 1, 3, 4)


@dataclasses.dataclass
class EndpointSuperResolutionConfig:
    """Stage A of the two-stage spatiotemporal model: a HiRO-style 2D
    residual-diffusion super-resolution network that generates fine-resolution
    endpoint estimates from the coarse endpoint frames alone. Its output feeds
    into the video-diffusion (stage B) endpoint pinning machinery in place of
    real fine-resolution ground truth -- see ``endpoints_observed``'s
    docstring for why. Reuses the same 2D EDM registry, preconditioning, and
    sampler as the spatial-only ``DiffusionModel`` in ``models.py``.

    Parameters:
        module: 2D network registry selector (``unet_diffusion_song`` or
            ``unet_diffusion_song_v2``, same as ``DiffusionModelConfig.module``).
        loss: Loss configuration for stage A's own denoising loss.
        training_noise_distribution: Noise distribution for stage A's
            training-time noise conditioning.
        coarse_shape: ``[lat, lon]`` of the FINE-resolution patch/domain this
            model will train on (i.e. the same H, W the video model itself
            sees -- NOT the coarse-resolution shape, despite the field name
            matching ``DiffusionModuleRegistrySelector.build``'s
            ``coarse_shape`` argument). Passed through with
            ``downscale_factor=1`` since stage A's input is already the
            coarse endpoint upsampled to fine resolution (reusing stage B's
            own ``coarse_upsampled``, not re-interpolating). Required because
            the underlying SongUNet backbone bakes a fixed-size positional
            embedding buffer into its weights at construction time
            (``physicsnemo_unets_v1/unets.py``) -- keep this in sync with
            whatever fine patch size ``coarse_patch_extent_lat/lon``
            (scaled by the fine/coarse downscale factor) actually produces.
        sigma_min: Min noise level for stage A generation.
        sigma_max: Max noise level for stage A generation.
        churn: Stochasticity during stage A generation.
        num_diffusion_generation_steps: Number of diffusion steps for stage A
            generation (independent of stage B's own schedule).
        loss_weight: Multiplies stage A's loss before summing into the
            combined total loss.
    """

    module: DiffusionModuleRegistrySelector
    loss: LossConfig
    training_noise_distribution: (
        LogNormalNoiseDistribution | LogUniformNoiseDistribution
    )
    coarse_shape: list[int]
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    churn: float = 0.0
    num_diffusion_generation_steps: int = 18
    loss_weight: float = 1.0

    def __post_init__(self):
        if len(self.coarse_shape) != 2:
            raise ValueError(
                f"coarse_shape must be [lat, lon], got {self.coarse_shape}."
            )
        if self.loss_weight < 0.0:
            raise ValueError(f"loss_weight must be >= 0, got {self.loss_weight}.")

    def build(self, n_channels: int) -> "torch.nn.Module":
        # sigma_data hardcoded to 1.0: standard-score-normalized data, matching
        # DiffusionModelConfig's own convention (see models.py).
        return self.module.build(
            n_in_channels=n_channels,
            n_out_channels=n_channels,
            coarse_shape=tuple(self.coarse_shape),
            downscale_factor=1,
            sigma_data=1.0,
        )


@dataclasses.dataclass
class VideoDiffusionModelConfig:
    """Configuration for the temporal-interpolation video diffusion model."""

    out_names: list[str]
    n_timesteps: int
    normalization: NormalizationConfig | None = None
    # Optional spatial downscaling. When set, the model additionally
    # conditions on a coarser-resolution clip of the same out_names/frames
    # (``batch.coarse``, bicubic-upsampled per frame to the fine grid -- see
    # ``_upsample_coarse_clip``) and predicts the residual against that
    # upsampled coarse field (bias-corrected by the endpoint residual, see
    # ``VideoDiffusionModel.train_on_batch``) instead of the pure temporal
    # baseline. None (default) reproduces the exact prior temporal-only
    # behavior, where ``batch.coarse`` is ignored entirely.
    coarse_normalization: NormalizationConfig | None = None
    # Whether the two endpoint frames are real fine-resolution ground truth
    # (default, matches all prior behavior): they're pinned exactly in the
    # output and their true values are fed into the conditioning/baseline.
    # Set to False for pure coarse-to-fine downscaling where NO frame ever
    # has fine-resolution truth available (e.g. deploying on a continuously-
    # running coarse emulator with no sparse high-res anchors) -- every frame
    # is then diffused against a pure ``upsample(coarse)`` baseline, with no
    # fine value of any kind entering the conditioning. Requires
    # ``coarse_normalization`` to be set (it's the only conditioning source
    # left) and is incompatible with subset_augmentation_prob/
    # marginal_consistency_weight > 0 or a brownian_bridge noise kernel (all
    # three assume a fixed pinned endpoint, which no longer exists).
    endpoints_observed: bool = True
    # Two-stage mode: when set, ``endpoints_observed`` must be True and
    # ``coarse_normalization`` must be set. Instead of pinning to real fine
    # ground truth (which isn't available in the pure coarse-to-fine
    # deployment scenario), a HiRO-style 2D super-resolution sub-model
    # (stage A, see EndpointSuperResolutionConfig) generates a fine-resolution
    # estimate of the two endpoints from the coarse endpoint frames alone,
    # and THAT estimate is pinned/fed into the video-diffusion (stage B)
    # conditioning -- restoring a meaningful pinned endpoint (so
    # brownian_bridge noise, subset training, and marginal-consistency loss
    # are valid again) without depending on real fine truth as input. Stage
    # A is trained by its own denoising loss (supervised against true fine
    # endpoints) with NO gradient from stage B's loss flowing back into it
    # (detached at the handoff) -- see VideoDiffusionModel._endpoint_super_resolution.
    endpoint_super_resolution: EndpointSuperResolutionConfig | None = None
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
    # Per-channel EDM sigma_data (std of the diffused residual). Channels left
    # unset default to 1.0. Setting it to the measured residual std per channel
    # fixes the preconditioning/loss weighting for residual diffusion.
    sigma_data_by_channel: dict[str, float] | None = None
    loss_weight_exponent: float = 1.0
    # Channels modeled in log space via log1p(x*scale); maps channel to scale.
    log_transform_channels: dict[str, float] | None = None
    # Temporal correlation of the residual noise: "independent" (per-frame white
    # noise, default), "brownian_bridge" (endpoint-pinned time-correlated noise,
    # same kernel for every channel), or "per_channel" (a different kernel per
    # channel, via per_channel_noise_kernel/per_channel_kernel_length_scale below
    # -- e.g. matching each channel to whichever of bridge/OU/RBF best fits its
    # real temporal correlation, see toy/process_residual_bridge_report.md).
    temporal_noise_correlation: str = "independent"
    # Required iff temporal_noise_correlation == "per_channel": maps every
    # out_names entry to "independent", "brownian_bridge", "ou", or "rbf".
    per_channel_noise_kernel: dict[str, str] | None = None
    # Required iff per_channel_noise_kernel has any "ou"/"rbf" entries: maps
    # those channels to their kernel length scale, in the same normalized
    # [0, 1]-over-the-full-window units as uniform_frame_times (i.e. hours /
    # total_window_hours, NOT raw hours -- e.g. an empirically-fit 12.9h OU
    # length scale over a 24h window is 12.9/24 = 0.5375 here).
    per_channel_kernel_length_scale: dict[str, float] | None = None
    # Fraction of training batches trained on a random subset of interior frames
    # (the two endpoints are always kept) instead of the full uniform grid, so the
    # model learns to answer variable query sets and stays consistent across them.
    # 0.0 (default) trains only on the full grid -- exact prior behavior.
    subset_augmentation_prob: float = 0.0
    # Minimum number of interior frames to keep when a batch is subsetted.
    subset_min_interior: int = 1
    # Weight of the marginal-consistency loss (video PMD L_marg). When > 0, each
    # training step runs a second pass on a random strict subset of the (possibly
    # augmentation-subset) interior frames -- sharing the first pass's noised
    # inputs on the shared frames -- and penalizes disagreement between the first
    # pass's prediction restricted to the subset and the subset-native prediction.
    # May be combined with subset_augmentation_prob (holds subset exposure fixed
    # while toggling the loss). 0.0 (default) disables it (single pass, exact
    # prior behavior).
    marginal_consistency_weight: float = 0.0

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
            "sigma_data_by_channel",
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
        if any(
            lvl not in range(len(self.channel_mult)) for lvl in self.attention_levels
        ):
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
        if self.temporal_noise_correlation not in (
            "independent", "brownian_bridge", "per_channel"
        ):
            raise ValueError(
                "temporal_noise_correlation must be 'independent', "
                f"'brownian_bridge', or 'per_channel', got "
                f"{self.temporal_noise_correlation}."
            )
        if self.temporal_noise_correlation == "per_channel":
            if self.per_channel_noise_kernel is None:
                raise ValueError(
                    "temporal_noise_correlation == 'per_channel' requires "
                    "per_channel_noise_kernel."
                )
            missing = set(self.out_names) - set(self.per_channel_noise_kernel)
            extra = set(self.per_channel_noise_kernel) - set(self.out_names)
            if missing or extra:
                raise ValueError(
                    "per_channel_noise_kernel must specify exactly out_names: "
                    f"missing {sorted(missing)}, unexpected {sorted(extra)}."
                )
            bad_kernels = {
                k for k in self.per_channel_noise_kernel.values()
                if k not in ("independent", "brownian_bridge", "ou", "rbf")
            }
            if bad_kernels:
                raise ValueError(
                    "per_channel_noise_kernel values must be 'independent', "
                    f"'brownian_bridge', 'ou', or 'rbf', got {sorted(bad_kernels)}."
                )
            needs_length_scale = {
                name for name, k in self.per_channel_noise_kernel.items()
                if k in ("ou", "rbf")
            }
            have_length_scale = set(self.per_channel_kernel_length_scale or {})
            missing_ell = needs_length_scale - have_length_scale
            if missing_ell:
                raise ValueError(
                    "per_channel_kernel_length_scale missing entries for "
                    f"ou/rbf channels: {sorted(missing_ell)}."
                )
            bad_ell = {
                name: ell for name, ell in (self.per_channel_kernel_length_scale or {}).items()
                if ell <= 0.0
            }
            if bad_ell:
                raise ValueError(
                    f"per_channel_kernel_length_scale must be > 0, got {bad_ell}."
                )
        elif self.per_channel_noise_kernel is not None:
            raise ValueError(
                "per_channel_noise_kernel is only used when "
                "temporal_noise_correlation == 'per_channel'."
            )
        if not 0.0 <= self.subset_augmentation_prob <= 1.0:
            raise ValueError(
                "subset_augmentation_prob must be in [0, 1], got "
                f"{self.subset_augmentation_prob}."
            )
        max_interior = self.n_timesteps - 2
        if not 1 <= self.subset_min_interior <= max_interior:
            raise ValueError(
                f"subset_min_interior must be in [1, {max_interior}] "
                f"(n_timesteps - 2), got {self.subset_min_interior}."
            )
        if self.marginal_consistency_weight < 0.0:
            raise ValueError(
                "marginal_consistency_weight must be >= 0, got "
                f"{self.marginal_consistency_weight}."
            )
        if self.marginal_consistency_weight > 0.0:
            # The full grid must admit a strict interior subset (keep in
            # [subset_min_interior, n_interior - 1]), so n_interior >=
            # subset_min_interior + 1. May be combined with subset_augmentation:
            # when an augmented batch is too small to form a strict subset the
            # consistency pass is skipped for that batch (see train_on_batch).
            if self.n_timesteps - 2 < self.subset_min_interior + 1:
                raise ValueError(
                    "marginal_consistency_weight > 0 needs n_timesteps >= "
                    f"subset_min_interior + 3 (got n_timesteps={self.n_timesteps}, "
                    f"subset_min_interior={self.subset_min_interior})."
                )
        if not self.endpoints_observed:
            if self.coarse_normalization is None:
                raise ValueError(
                    "endpoints_observed=False requires coarse_normalization to "
                    "be set -- it's the only conditioning source left once no "
                    "frame has fine-resolution truth."
                )
            if self.subset_augmentation_prob > 0.0:
                raise ValueError(
                    "endpoints_observed=False is incompatible with "
                    "subset_augmentation_prob > 0: subset training assumes a "
                    "fixed pinned endpoint, which doesn't exist here."
                )
            if self.marginal_consistency_weight > 0.0:
                raise ValueError(
                    "endpoints_observed=False is incompatible with "
                    "marginal_consistency_weight > 0: the consistency pass "
                    "assumes a fixed pinned endpoint, which doesn't exist here."
                )
            bridge_kernels = (
                {self.temporal_noise_correlation}
                if self.temporal_noise_correlation != "per_channel"
                else set((self.per_channel_noise_kernel or {}).values())
            )
            if "brownian_bridge" in bridge_kernels:
                raise ValueError(
                    "endpoints_observed=False is incompatible with "
                    "brownian_bridge noise (per-channel or shared): its "
                    "endpoint-pinned covariance assumes a fixed pinned "
                    "endpoint, which doesn't exist here."
                )
        if self.endpoint_super_resolution is not None:
            if not self.endpoints_observed:
                raise ValueError(
                    "endpoint_super_resolution requires endpoints_observed=True "
                    "-- it generates the pinned endpoint value, it doesn't "
                    "remove pinning."
                )
            if self.coarse_normalization is None:
                raise ValueError(
                    "endpoint_super_resolution requires coarse_normalization "
                    "to be set -- stage A super-resolves the coarse endpoint "
                    "frames."
                )
        if self.backbone not in ("simple", "songunet"):
            raise ValueError(
                f"backbone must be 'simple' or 'songunet', got {self.backbone}."
            )
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

    def sigma_data_tensor(self, device: torch.device) -> torch.Tensor:
        """Per-channel sigma_data (default 1.0), ordered by out_names."""
        return torch.tensor(
            [
                1.0
                if self.sigma_data_by_channel is None
                else self.sigma_data_by_channel.get(name, 1.0)
                for name in self.out_names
            ],
            dtype=torch.float32,
            device=device,
        )

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
        # fine and coarse share out_names; coarse may be a lower spatial
        # resolution of the same variables (see coarse_normalization).
        # Clip length comes from the data config.
        return DataRequirements(
            fine_names=self.out_names,
            coarse_names=self.out_names,
            n_timesteps=1,
            use_fine_topography=False,
        )

    def build(
        self,
        normalizer: StandardNormalizer | None = None,
        coarse_normalizer: StandardNormalizer | None = None,
        full_fine_coords: LatLonCoordinates | None = None,
        downscale_factor: int | None = None,
    ) -> "VideoDiffusionModel":
        """
        Args:
            normalizer: Prebuilt normalizer; built from ``normalization`` if
                not given.
            coarse_normalizer: Prebuilt coarse normalizer; built from
                ``coarse_normalization`` if not given.
            full_fine_coords: The full fine-resolution domain coordinates
                (e.g. ``PairedVideoGriddedData.fine_coords``). Optional --
                only needed for ``VideoDiffusionModel.generate_on_batch_no_target``,
                which derives the output grid for a coarse-only batch from
                this plus ``downscale_factor`` (mirrors
                ``DiffusionModelConfig.build``'s same-named argument).
            downscale_factor: The coarse-to-fine downscale factor (e.g.
                ``PairedVideoGriddedData.downscale_factor``). Same
                optionality/purpose as ``full_fine_coords``.
        """
        if normalizer is None:
            if self.normalization is None:
                raise ValueError(
                    "Either `normalization` config or a prebuilt `normalizer` "
                    "must be provided."
                )
            normalizer = self.normalization.build(self.out_names)
        if coarse_normalizer is None and self.coarse_normalization is not None:
            coarse_normalizer = self.coarse_normalization.build(self.out_names)

        n_channels = len(self.out_names)
        # noisy residual (C) + log-sigma (C), plus conditioning:
        #  - endpoint values (C) + observed mask (1), only if endpoints_observed
        #  - upsampled coarse clip (C), only if spatial downscaling is enabled
        # endpoints_observed=False requires coarse_normalizer set (validated in
        # __post_init__), so in_channels is always well-defined.
        in_channels = 2 * n_channels
        if self.endpoints_observed:
            in_channels += n_channels + 1
        if coarse_normalizer is not None:
            in_channels += n_channels
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
        module = VideoEDMPrecond(net, sigma_data=self.sigma_data_tensor(get_device()))
        endpoint_sr_module = None
        if self.endpoint_super_resolution is not None:
            endpoint_sr_module = self.endpoint_super_resolution.build(n_channels)
        return VideoDiffusionModel(
            self,
            module,
            normalizer,
            self.out_names,
            coarse_normalizer=coarse_normalizer,
            endpoint_sr_module=endpoint_sr_module,
            full_fine_coords=full_fine_coords,
            downscale_factor=downscale_factor,
        )


def _channel_mixing_matrix(
    tau: torch.Tensor, kernel: str, length_scale: float | None
) -> torch.Tensor:
    """``(T, T)`` mixing matrix for one channel's chosen kernel -- the
    per-channel building block for ``temporal_noise_correlation ==
    'per_channel'``. ``"independent"`` embeds an identity in the interior
    block (endpoints still zero), giving unmixed white noise for that
    channel while still stacking cleanly into a per-channel tensor.
    """
    if kernel == "independent":
        n_timesteps = tau.shape[0]
        n_interior = n_timesteps - 2
        mixing = torch.zeros(
            n_timesteps, n_timesteps, dtype=torch.float32, device=tau.device
        )
        mixing[1 : 1 + n_interior, 1 : 1 + n_interior] = torch.eye(
            n_interior, device=tau.device
        )
        return mixing
    if kernel == "brownian_bridge":
        return brownian_bridge_mixing_matrix(tau)
    if kernel == "ou":
        assert length_scale is not None
        return ou_mixing_matrix(tau, length_scale)
    if kernel == "rbf":
        assert length_scale is not None
        return rbf_mixing_matrix(tau, length_scale)
    raise ValueError(f"Unknown per-channel noise kernel {kernel!r}")


def _per_channel_mixing_tensor(
    tau: torch.Tensor,
    out_names: list[str],
    kernel_map: dict[str, str],
    length_scale_map: dict[str, float] | None,
) -> torch.Tensor:
    """``(C, T, T)`` stacked per-channel mixing matrices, ordered by
    ``out_names`` -- each channel's own kernel/length-scale choice."""
    mats = [
        _channel_mixing_matrix(
            tau, kernel_map[name], (length_scale_map or {}).get(name)
        )
        for name in out_names
    ]
    return torch.stack(mats, dim=0)


class VideoDiffusionModel:
    def __init__(
        self,
        config: VideoDiffusionModelConfig,
        module: torch.nn.Module,
        normalizer: StandardNormalizer,
        out_names: list[str],
        coarse_normalizer: StandardNormalizer | None = None,
        endpoint_sr_module: torch.nn.Module | None = None,
        full_fine_coords: LatLonCoordinates | None = None,
        downscale_factor: int | None = None,
    ):
        self.config = config
        # (1, C, 1, 1, 1) so it broadcasts against the per-channel sigma tensor.
        self.sigma_data = config.sigma_data_tensor(get_device()).reshape(1, -1, 1, 1, 1)
        dist = Distributed.get_instance()
        self.module = dist.wrap_module(module.to(get_device()))
        self.normalizer = normalizer
        # Set iff config.coarse_normalization is set: enables the spatial
        # (coarse-conditioned) path in train_on_batch/generate. None keeps the
        # original temporal-only behavior, ignoring batch.coarse entirely.
        self.coarse_normalizer = coarse_normalizer
        # False: no frame is pinned to fine-resolution truth -- every frame is
        # diffused against a pure upsample(coarse) baseline (see
        # VideoDiffusionModelConfig.endpoints_observed's docstring).
        self.endpoints_observed = config.endpoints_observed
        # Set iff config.endpoint_super_resolution is set: stage A, a 2D EDM
        # network that generates the pinned endpoint value instead of reading
        # it from batch.fine -- see _endpoint_super_resolution.
        self.endpoint_sr_config = config.endpoint_super_resolution
        self.endpoint_sr_module = (
            dist.wrap_module(endpoint_sr_module.to(get_device()))
            if endpoint_sr_module is not None
            else None
        )
        self.endpoint_sr_loss = (
            self.endpoint_sr_config.loss.build(gridded_operations=None)
            if self.endpoint_sr_config is not None
            else None
        )
        # Canonical fine-resolution output grid + coarse-to-fine ratio; both
        # optional (None unless the caller supplied them at build time), only
        # required by generate_on_batch_no_target.
        self.full_fine_coords = (
            full_fine_coords.to(get_device()) if full_fine_coords is not None else None
        )
        self.downscale_factor = downscale_factor
        self.out_names = out_names
        self.packer = Packer(out_names)
        self.n_timesteps = config.n_timesteps
        self.log_transform_channels = dict(config.log_transform_channels or {})
        # normalized full-grid frame times (endpoints at 0/1) used to derive the
        # baseline weights and bridge kernel for whatever frame subset is in play.
        self._full_tau = uniform_frame_times(config.n_timesteps)
        self._marginal_consistency_weight = config.marginal_consistency_weight
        self._bridge_noise = config.temporal_noise_correlation == "brownian_bridge"
        self._per_channel_noise = config.temporal_noise_correlation == "per_channel"
        if self._bridge_noise:
            self._noise_mixing: torch.Tensor | None = brownian_bridge_mixing_matrix(
                self._full_tau
            ).to(get_device())
        elif self._per_channel_noise:
            assert config.per_channel_noise_kernel is not None  # validated in __post_init__
            self._noise_mixing = _per_channel_mixing_tensor(
                self._full_tau,
                out_names,
                config.per_channel_noise_kernel,
                config.per_channel_kernel_length_scale,
            ).to(get_device())
        else:
            self._noise_mixing = None

    @property
    def modules(self) -> torch.nn.ModuleList:
        modules = [self.module]
        if self.endpoint_sr_module is not None:
            modules.append(self.endpoint_sr_module)
        return torch.nn.ModuleList(modules)

    def _sample_residual_noise(
        self, like: torch.Tensor, mixing: torch.Tensor | None = None
    ) -> torch.Tensor:
        """White noise shaped like ``like`` (B, C, T, H, W), temporally correlated
        with ``mixing`` when given (bridge/OU/RBF kernel, shared or per-channel),
        else independent.

        ``mixing`` defaults to the full-grid mixing tensor; pass a subset matrix
        for a subset of frames, or ``None`` stays independent (mixing is ``None``
        in independent mode regardless). ``mixing`` is either ``(T, T)`` (same
        kernel for every channel) or ``(C, T, T)`` (per-channel kernels).
        """
        noise = randn_like(like)
        if mixing is None:
            mixing = self._noise_mixing
        if mixing is None:
            return noise
        mixing = mixing.to(device=noise.device, dtype=noise.dtype)
        if mixing.ndim == 3:
            return torch.einsum("cti,bcihw->bcthw", mixing, noise)
        return torch.einsum("ti,bcihw->bcthw", mixing, noise)

    def _tau_for_indices(self, idx: torch.Tensor | None) -> torch.Tensor | None:
        """Normalized frame times for a subset of the full grid (``None`` = full
        grid, where the ``linspace`` default in the baseline already applies).
        """
        if idx is None:
            return None
        return self._full_tau.to(idx.device).index_select(0, idx)

    def _mixing_for_indices(self, idx: torch.Tensor | None) -> torch.Tensor | None:
        """Mixing tensor for a subset of frames -- the full-window kernel(s)
        restricted to ``idx`` (re-derived at just those frame times, which
        equals the true marginal -- see brownian_bridge_mixing_matrix's
        docstring). ``None`` for independent noise or the full grid.
        """
        if not (self._bridge_noise or self._per_channel_noise):
            return None
        if idx is None:
            return self._noise_mixing
        tau = self._full_tau.index_select(0, idx.cpu())
        if self._per_channel_noise:
            assert self.config.per_channel_noise_kernel is not None
            return _per_channel_mixing_tensor(
                tau,
                self.out_names,
                self.config.per_channel_noise_kernel,
                self.config.per_channel_kernel_length_scale,
            ).to(get_device())
        return brownian_bridge_mixing_matrix(tau).to(get_device())

    def _synced_generator(self, device: torch.device) -> torch.Generator:
        """A CPU RNG seeded identically on every rank (drawn on rank 0 and
        broadcast) so all data-/model-parallel ranks pick the same frame subset
        and therefore agree on the temporal shape of the batch.
        """
        dist = Distributed.get_instance()
        seed = torch.randint(0, 2**31 - 1, (1,), device=device)
        if dist.rank != 0:
            seed.zero_()
        seed = int(dist.reduce_sum(seed).item())
        return torch.Generator().manual_seed(seed)

    @staticmethod
    def _interior_subset_indices(
        n_keep: int, n_times: int, gen: torch.Generator, device: torch.device
    ) -> torch.Tensor:
        """Sorted frame indices keeping both endpoints plus ``n_keep`` random
        interior frames.
        """
        n_interior = n_times - 2
        interior = torch.sort(torch.randperm(n_interior, generator=gen)[:n_keep]).values
        idx = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),
                interior + 1,
                torch.full((1,), n_times - 1, dtype=torch.long),
            ]
        )
        return idx.to(device)

    def _sample_training_subset_indices(
        self, n_times: int, device: torch.device
    ) -> torch.Tensor | None:
        """Randomly pick frame indices (always keeping the two endpoints) for
        subset-augmented training, or ``None`` to train on the full grid.
        """
        if self.config.subset_augmentation_prob <= 0.0:
            return None
        gen = self._synced_generator(device)
        if float(torch.rand((), generator=gen)) >= self.config.subset_augmentation_prob:
            return None
        n_interior = n_times - 2
        n_keep = int(
            torch.randint(
                self.config.subset_min_interior, n_interior + 1, (1,), generator=gen
            )
        )
        if n_keep >= n_interior:
            return None  # kept everything -> full grid
        return self._interior_subset_indices(n_keep, n_times, gen, device)

    def _sample_consistency_subset_indices(
        self, n_times: int, device: torch.device
    ) -> torch.Tensor:
        """A random *strict* interior subset (endpoints kept, at least one interior
        frame dropped) for the marginal-consistency second pass.
        """
        gen = self._synced_generator(device)
        n_interior = n_times - 2
        # keep in [subset_min_interior, n_interior - 1]: never the full interior,
        # so the two passes always differ in their query set.
        n_keep = int(
            torch.randint(
                self.config.subset_min_interior, n_interior, (1,), generator=gen
            )
        )
        return self._interior_subset_indices(n_keep, n_times, gen, device)

    @staticmethod
    def _validate_frames(
        frames: "list[int] | None", n_times: int, device: torch.device
    ) -> torch.Tensor | None:
        """Validate a requested frame subset for ``generate`` and return it as a
        LongTensor of indices into the full grid, or ``None`` for the full grid.
        """
        if frames is None:
            return None
        idx = torch.as_tensor(list(frames), dtype=torch.long)
        if idx.ndim != 1 or idx.numel() < 3:
            raise ValueError(
                "frames must list at least 3 frame indices (2 endpoints + "
                f"interior), got {list(frames)}."
            )
        if int(idx[0]) != 0 or int(idx[-1]) != n_times - 1:
            raise ValueError(
                f"frames must start at 0 and end at {n_times - 1} (the observed "
                f"endpoints), got {list(frames)}."
            )
        if not bool(torch.all(idx[1:] > idx[:-1])):
            raise ValueError(f"frames must be strictly increasing, got {list(frames)}.")
        return idx.to(device)

    def _pack_normalized(
        self, data: TensorMapping, normalizer: StandardNormalizer | None = None
    ) -> torch.Tensor:
        selected = {
            k: torch.log1p(data[k].clamp(min=0.0) * scale)
            if (scale := self.log_transform_channels.get(k))
            else data[k]
            for k in self.out_names
        }
        normalized = (normalizer or self.normalizer).normalize(selected)
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
        self,
        clip: torch.Tensor,
        interior_mask: torch.Tensor,
        coarse_upsampled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Observed-endpoint values + a binary observed mask channel (only
        when ``self.endpoints_observed``; no fine value of any kind enters
        the conditioning otherwise), plus the upsampled coarse clip (all
        frames) when spatial downscaling is enabled.
        """
        parts = []
        if self.endpoints_observed:
            observed = 1.0 - interior_mask
            observed_values = clip * observed
            mask_channel = observed.expand(
                clip.shape[0], 1, -1, clip.shape[-2], clip.shape[-1]
            )
            parts.extend([observed_values, mask_channel])
        if coarse_upsampled is not None:
            parts.append(coarse_upsampled)
        return torch.cat(parts, dim=CHANNEL_AXIS)

    def _calendar_inputs(self, batch_fine):
        lon = batch_fine.latlon_coordinates.lon
        if lon.dim() == 2:  # all members identical, use first
            lon = lon[0]
        return (
            batch_fine.day_of_year.to(get_device()),
            batch_fine.second_of_day.to(get_device()),
            lon.to(get_device()),
        )

    def _upsample_coarse_for_batch(
        self, batch: PairedVideoBatchData, fine_hw: tuple[int, int]
    ) -> torch.Tensor | None:
        """Normalized, fine-resolution-upsampled coarse clip (all frames), or
        ``None`` when spatial downscaling is disabled (``coarse_normalizer``
        unset). Full ``n_timesteps`` length; caller subsets to match ``clip``.
        """
        if self.coarse_normalizer is None:
            return None
        coarse_clip = self._pack_normalized(batch.coarse.data, self.coarse_normalizer)
        return _upsample_coarse_clip(coarse_clip, fine_hw)

    def _spatiotemporal_baseline(
        self,
        clip: torch.Tensor,
        coarse_upsampled: torch.Tensor | None,
        tau: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-frame baseline used to form the diffused residual.

        Three cases:
        - No spatial downscaling: the pure temporal linear interpolation of
          the observed endpoints (prior behavior).
        - Spatial downscaling + endpoints_observed: the upsampled coarse
          field at every frame, bias-corrected by linearly interpolating the
          *endpoint* residual (fine - upsampled coarse) across the clip -- so
          it still exactly reproduces the true fine endpoints (required for
          pinning), while using the real coarse observation (not a temporal
          guess) at interior frames.
        - Spatial downscaling, endpoints_observed=False: pure
          upsample(coarse) at every frame, with NO use of ``clip`` (the fine
          truth) at all -- there's nothing to bias-correct against or pin to.
        """
        if coarse_upsampled is None:
            return _linear_interp_endpoints(clip, tau)
        if not self.endpoints_observed:
            return coarse_upsampled
        return coarse_upsampled + _linear_interp_endpoints(
            clip - coarse_upsampled, tau
        )

    def _endpoint_super_resolution(
        self,
        clip: torch.Tensor,
        coarse_upsampled: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stage A: HiRO-style 2D residual-diffusion super-resolution of the
        two endpoint frames from the coarse conditioning alone.

        Returns ``(anchor_clip, loss_a)``: ``anchor_clip`` is ``clip`` with
        the two endpoint frame slots (index 0 and -1) replaced by stage A's
        generated fine-resolution estimate -- everything downstream
        (``_conditioning``, ``_spatiotemporal_baseline``) is called with
        ``anchor_clip`` in place of ``clip``, so the existing
        ``endpoints_observed=True`` pinning machinery is unchanged, it's just
        pinned to a generated value instead of real fine truth. ``loss_a`` is
        stage A's own denoising loss (supervised against the TRUE fine
        endpoints in ``clip``) when ``training=True``, else ``None``.

        The generated estimate is detached before being folded into
        ``anchor_clip`` (regardless of ``training``): stage B's loss does not
        backprop into stage A's weights, matching the "one model call, two
        independently-supervised losses" design -- see
        ``VideoDiffusionModelConfig.endpoint_super_resolution``'s docstring.
        """
        assert self.endpoint_sr_module is not None
        assert self.endpoint_sr_config is not None
        batch = clip.shape[0]
        coarse_endpoints = _fold_endpoint_frames(coarse_upsampled)  # (B*2, C, H, W)

        loss_a: torch.Tensor | None = None
        if training:
            true_fine_endpoints = _fold_endpoint_frames(clip)
            target_residual = true_fine_endpoints - coarse_endpoints
            conditioned = condition_with_noise_for_training(
                target_residual,
                self.endpoint_sr_config.training_noise_distribution,
                sigma_data=1.0,
            )
            denoised_residual = self.endpoint_sr_module(
                conditioned.latents, coarse_endpoints, conditioned.sigma
            )
            assert self.endpoint_sr_loss is not None
            [loss_component] = self.endpoint_sr_loss(
                denoised_residual, target_residual
            )
            loss_a = torch.mean(conditioned.weight * loss_component.loss)
            sr_residual = denoised_residual.detach()
        else:
            latents = torch.randn_like(coarse_endpoints)
            sr_residual, _ = edm_sampler(
                self.endpoint_sr_module,
                latents,
                coarse_endpoints,
                num_steps=self.endpoint_sr_config.num_diffusion_generation_steps,
                sigma_min=self.endpoint_sr_config.sigma_min,
                sigma_max=self.endpoint_sr_config.sigma_max,
                S_churn=self.endpoint_sr_config.churn,
            )

        sr_endpoints = _unfold_endpoint_frames(coarse_endpoints + sr_residual, batch)
        anchor_clip = clip.clone()
        anchor_clip[:, :, [0, -1]] = sr_endpoints
        return anchor_clip, loss_a

    def train_on_batch(self, batch: PairedVideoBatchData, optimizer) -> ModelOutputs:
        fine = batch.fine
        clip = self._pack_normalized(fine.data)
        day_of_year, second_of_day, lon = self._calendar_inputs(fine)
        coarse_upsampled = self._upsample_coarse_for_batch(batch, clip.shape[-2:])

        # Optionally train on a random subset of interior frames (endpoints kept)
        # so the model learns to answer variable query sets; the bridge noise and
        # baseline follow the subset's true times, i.e. the full-window marginal.
        idx = self._sample_training_subset_indices(clip.shape[2], clip.device)
        if idx is not None:
            clip = clip.index_select(2, idx)
            day_of_year = day_of_year.index_select(1, idx)
            second_of_day = second_of_day.index_select(1, idx)
            if coarse_upsampled is not None:
                coarse_upsampled = coarse_upsampled.index_select(2, idx)
        batch_size, _, n_times, _, _ = clip.shape
        tau = self._tau_for_indices(idx)

        # Stage A (optional): generate the pinned endpoint value instead of
        # reading it from clip. anchor_clip == clip except at the two
        # endpoint slots; residual below still uses true clip (GT
        # supervision for stage B is unaffected).
        endpoint_sr_loss: torch.Tensor | None = None
        if self.endpoint_sr_module is not None:
            assert coarse_upsampled is not None  # validated in __post_init__
            anchor_clip, endpoint_sr_loss = self._endpoint_super_resolution(
                clip, coarse_upsampled, training=True
            )
        else:
            anchor_clip = clip

        baseline = self._spatiotemporal_baseline(anchor_clip, coarse_upsampled, tau)
        residual = clip - baseline  # ~0 at endpoints by construction (if observed)

        interior = _interior_mask(n_times, clip.device, self.endpoints_observed)
        condition = self._conditioning(anchor_clip, interior, coarse_upsampled)
        mixing = self._mixing_for_indices(idx)

        sigma = self.config.sample_training_noise(batch_size, clip.device)
        sigma = sigma.reshape(batch_size, -1, 1, 1, 1)
        # Gaussian noise on interior frames only
        noise = self._sample_residual_noise(residual, mixing)
        noised = residual + noise * sigma * interior

        # ``idx`` (or None for the full grid) is exactly the frames' true grid
        # positions, so the temporal attention's relative bias reflects real
        # spacing rather than packed-contiguous order.
        denoised = self.module(
            noised, condition, sigma, day_of_year, second_of_day, lon, frame_index=idx
        )

        weight = (
            (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        ) ** self.config.loss_weight_exponent
        sq_err = (denoised - residual) ** 2 * interior
        n_interior_elems = interior.expand_as(sq_err).sum()
        loss = (weight * sq_err).sum() / n_interior_elems

        # Marginal-consistency loss: a second pass on a random strict subset of
        # the (full) interior frames, sharing the SAME noised inputs, sigma, and
        # conditioning on the shared frames (obtained by slicing the full pass, so
        # the only difference is the query set). We add the subset's own diffusion
        # loss plus a penalty tying the full-pass prediction, restricted to the
        # subset, to the subset-native prediction on the shared interior frames.
        marginal_loss: torch.Tensor | None = None
        total_loss = loss
        # A strict interior subset needs at least subset_min_interior + 1 interior
        # frames; an augmentation-shrunk batch may fall below that, so skip the
        # consistency pass for it rather than fail.
        can_subset = n_times - 2 >= self.config.subset_min_interior + 1
        if self._marginal_consistency_weight > 0.0 and can_subset:
            sub = self._sample_consistency_subset_indices(n_times, clip.device)
            interior_s = _interior_mask(int(sub.numel()), clip.device)
            # True grid positions of the consistency frames: map the subset picks
            # through the current frames' own grid positions (``idx`` when the
            # batch was already augmented, else the full grid).
            base_index = (
                idx if idx is not None else torch.arange(n_times, device=clip.device)
            )
            sub_frame_index = base_index.index_select(0, sub)
            denoised_s = self.module(
                noised.index_select(2, sub),
                condition.index_select(2, sub),
                sigma,
                day_of_year.index_select(1, sub),
                second_of_day.index_select(1, sub),
                lon,
                frame_index=sub_frame_index,
            )
            residual_s = residual.index_select(2, sub)
            n_interior_s = interior_s.expand_as(residual_s).sum()
            sq_err_s = (denoised_s - residual_s) ** 2 * interior_s
            dsm_sub = (weight * sq_err_s).sum() / n_interior_s
            # full-pass prediction restricted to the subset vs subset-native one
            diff = (denoised.index_select(2, sub) - denoised_s) ** 2 * interior_s
            marginal_loss = diff.sum() / n_interior_s
            total_loss = (
                loss + dsm_sub + self._marginal_consistency_weight * marginal_loss
            )

        if endpoint_sr_loss is not None:
            assert self.endpoint_sr_config is not None
            total_loss = (
                total_loss + self.endpoint_sr_config.loss_weight * endpoint_sr_loss
            )

        optimizer.accumulate_loss(total_loss)
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
        # keep target aligned with the (possibly subset) prediction frames
        target = {
            k: fine.data[k] if idx is None else fine.data[k].index_select(1, idx)
            for k in self.out_names
        }
        return ModelOutputs(
            prediction=prediction,
            target=target,
            loss=total_loss,
            marginal_consistency_loss=(
                None if marginal_loss is None else marginal_loss.detach()
            ),
            channel_losses=channel_losses,
            sigma=(
                sigma.flatten()
                if sigma.shape[1] == 1
                else sigma.squeeze(-1).squeeze(-1).squeeze(-1)
            ),
            per_sample_channel_loss=per_sample_channel_loss,
            endpoint_sr_loss=(
                None if endpoint_sr_loss is None else endpoint_sr_loss.detach()
            ),
        )

    @torch.no_grad()
    def generate(
        self,
        batch: PairedVideoBatchData,
        n_samples: int = 1,
        frames: list[int] | None = None,
    ) -> TensorDict:
        """Generate the interior frames conditioned on the observed endpoints
        (or, with ``self.endpoints_observed=False``, generate every frame
        from the coarse conditioning alone -- see the module docstring).

        ``frames`` optionally restricts generation to a subset of frame indices
        into the full ``n_timesteps`` grid; it must start at 0 and end at
        ``n_timesteps - 1`` (the query grid's boundary indices, regardless of
        whether those frames are pinned) and be strictly increasing. The
        returned clips then contain only those frames. The baseline and the
        bridge noise use the subset's true times, so a subset draws from the
        exact marginal of the full-window process. Defaults to the full grid.
        """
        fine = batch.fine
        clip = self._pack_normalized(fine.data)
        day_of_year, second_of_day, lon = self._calendar_inputs(fine)
        coarse_upsampled = self._upsample_coarse_for_batch(batch, clip.shape[-2:])
        idx = self._validate_frames(frames, clip.shape[2], clip.device)
        if idx is not None:
            clip = clip.index_select(2, idx)
            day_of_year = day_of_year.index_select(1, idx)
            second_of_day = second_of_day.index_select(1, idx)
            if coarse_upsampled is not None:
                coarse_upsampled = coarse_upsampled.index_select(2, idx)
        batch_size, n_channels, n_times, height, width = clip.shape
        tau = self._tau_for_indices(idx)

        # Stage A (optional): generate the pinned endpoint value instead of
        # reading it from clip. Computed once (not per ensemble sample) and
        # shared across all n_samples, same as a real observed endpoint
        # would be -- the repeat() calls below broadcast it to the ensemble.
        if self.endpoint_sr_module is not None:
            assert coarse_upsampled is not None  # validated in __post_init__
            anchor_clip, _ = self._endpoint_super_resolution(
                clip, coarse_upsampled, training=False
            )
        else:
            anchor_clip = clip

        baseline = self._spatiotemporal_baseline(anchor_clip, coarse_upsampled, tau)
        interior = _interior_mask(n_times, clip.device, self.endpoints_observed)
        condition = self._conditioning(anchor_clip, interior, coarse_upsampled)
        mixing = self._mixing_for_indices(idx)

        def repeat(t):
            return t.repeat_interleave(n_samples, dim=0)

        condition = repeat(condition)
        baseline = repeat(baseline)
        day_of_year = repeat(day_of_year)
        second_of_day = repeat(second_of_day)
        interior_b = interior.expand(batch_size * n_samples, n_channels, n_times, 1, 1)

        latents = self._sample_residual_noise(
            torch.empty(
                batch_size * n_samples,
                n_channels,
                n_times,
                height,
                width,
                device=clip.device,
            ),
            mixing,
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
            noise_mixing=mixing,
            frame_index=idx,
        )
        full_norm = baseline + residual
        generated = self._denormalize_invert(full_norm)
        # (B*n, T, H, W) -> (B, n, T, H, W)
        return {
            k: v.reshape(batch_size, n_samples, *v.shape[1:])
            for k, v in generated.items()
        }

    def get_fine_coords_for_batch(self, coarse: VideoBatchData) -> LatLonCoordinates:
        """Fine-resolution coordinates matching the spatial extent of a
        coarse-only batch, derived from ``self.full_fine_coords`` -- the
        video analog of ``DiffusionModel.get_fine_coords_for_batch`` in
        ``models.py`` (same ``adjust_fine_coord_range`` utility).
        """
        if self.full_fine_coords is None or self.downscale_factor is None:
            raise ValueError(
                "get_fine_coords_for_batch requires the model to have been "
                "built with full_fine_coords and downscale_factor -- see "
                "VideoDiffusionModelConfig.build."
            )
        coarse_lat = coarse.latlon_coordinates.lat[0]
        coarse_lon = coarse.latlon_coordinates.lon[0]
        fine_lat_interval = adjust_fine_coord_range(
            coarse.lat_interval,
            full_coarse_coord=coarse_lat,
            full_fine_coord=self.full_fine_coords.lat,
            downscale_factor=self.downscale_factor,
        )
        fine_lon_interval = adjust_fine_coord_range(
            coarse.lon_interval,
            full_coarse_coord=coarse_lon,
            full_fine_coord=self.full_fine_coords.lon,
            downscale_factor=self.downscale_factor,
        )
        return LatLonCoordinates(
            lat=fine_lat_interval.subset_of(self.full_fine_coords.lat),
            lon=fine_lon_interval.subset_of(self.full_fine_coords.lon),
        )

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        coarse: VideoBatchData,
        n_samples: int = 1,
        frames: list[int] | None = None,
    ) -> TensorDict:
        """Generate from coarse data ALONE -- no fine-resolution truth of any
        kind, not even for the "endpoint" frames (the video analog of
        ``DiffusionModel.generate_on_batch_no_target`` in ``models.py``).

        Only valid when no real fine-resolution value is ever required as
        input: ``endpoints_observed=False`` (pure coarse-to-fine), or
        ``endpoint_super_resolution`` set (stage A generates the endpoint
        estimate instead). With plain ``endpoints_observed=True`` and no
        stage A, the model fundamentally needs real fine endpoint values as
        conditioning, which this method -- by construction -- doesn't have.

        Implementation note: rather than duplicating ``generate``'s logic,
        this builds a zero-filled placeholder ``fine`` ``VideoBatchData`` with
        the correct shape/coordinates/calendar metadata (derived from
        ``coarse`` plus ``get_fine_coords_for_batch``) and delegates to
        ``generate``. This is safe because in both supported modes,
        ``generate`` never actually reads the fine data's *values* at any
        frame it isn't given as real input: interior frames are masked to
        zero before entering the conditioning tensor and are never read by
        the baseline; endpoint frames are either not read at all
        (endpoints_observed=False) or overwritten by stage A before anything
        reads them (endpoint_super_resolution).
        """
        if self.endpoints_observed and self.endpoint_sr_module is None:
            raise ValueError(
                "generate_on_batch_no_target requires endpoints_observed=False "
                "or endpoint_super_resolution to be set -- with plain "
                "endpoints_observed=True the model needs real fine-resolution "
                "endpoint values as input, which this method doesn't have."
            )
        fine_coords = self.get_fine_coords_for_batch(coarse)
        fine_hw = (len(fine_coords.lat), len(fine_coords.lon))
        example = next(iter(coarse.data.values()))
        batch_size, n_times = example.shape[0], example.shape[1]
        dummy_fine_data = {
            name: torch.zeros(
                batch_size,
                n_times,
                *fine_hw,
                dtype=example.dtype,
                device=example.device,
            )
            for name in self.out_names
        }
        dummy_fine = VideoBatchData(
            data=dummy_fine_data,
            time=coarse.time,
            latlon_coordinates=BatchedLatLonCoordinates(
                lat=fine_coords.lat.unsqueeze(0).expand(batch_size, -1).clone(),
                lon=fine_coords.lon.unsqueeze(0).expand(batch_size, -1).clone(),
            ),
            day_of_year=coarse.day_of_year,
            second_of_day=coarse.second_of_day,
        )
        pseudo_batch = PairedVideoBatchData(fine=dummy_fine, coarse=coarse)
        return self.generate(pseudo_batch, n_samples=n_samples, frames=frames)

    def get_state(self) -> Mapping[str, Any]:
        state: dict[str, Any] = {
            "config": dataclasses.asdict(self.config),
            "module": self.module.state_dict(),
        }
        if self.endpoint_sr_module is not None:
            state["endpoint_sr_module"] = self.endpoint_sr_module.state_dict()
        return state


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
    frame_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """EDM stochastic sampler that keeps the observed endpoints pinned (re-zeroed
    after every update). When ``noise_mixing`` is given, the stochastic churn noise
    is temporally correlated with that mixing matrix (matching the training noise).
    """
    compute_dtype = torch.float32 if latents.device.type == "mps" else torch.float64
    sigma_min_t = torch.as_tensor(
        sigma_min, dtype=compute_dtype, device=latents.device
    ).reshape(1, -1, 1, 1, 1)
    sigma_max_t = torch.as_tensor(
        sigma_max, dtype=compute_dtype, device=latents.device
    ).reshape(1, -1, 1, 1, 1)
    step = torch.arange(num_steps, dtype=compute_dtype, device=latents.device).reshape(
        -1, 1, 1, 1, 1
    )
    t = (
        sigma_max_t ** (1 / rho)
        + step / (num_steps - 1) * (sigma_min_t ** (1 / rho) - sigma_max_t ** (1 / rho))
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
            frame_index=frame_index,
        )
        return out.to(compute_dtype)

    def churn_noise(x):
        noise = torch.randn_like(x)
        if noise_mixing is None:
            return noise
        mixing = noise_mixing.to(device=noise.device, dtype=noise.dtype)
        if mixing.ndim == 3:
            return torch.einsum("cti,bcihw->bcthw", mixing, noise)
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
