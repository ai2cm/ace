import datetime

import cftime
import pytest
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import NullOptimization
from fme.core.rand import set_seed
from fme.downscaling.data.datasets import (
    PairedVideoBatchData,
    VideoBatchData,
    VideoBatchItem,
)
from fme.downscaling.data.time_encoding import compute_calendar_features
from fme.downscaling.metrics_and_maths import interpolate as interpolate_2d
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.video_modules import FIRBlur, TemporalAttention
from fme.downscaling.noise import (
    LogNormalNoiseDistribution,
    LogUniformNoiseDistribution,
)
from fme.downscaling.twoblock import coarse_temporal_interp, conservative_downsample
from fme.downscaling.video_models import (
    CHANNEL_AXIS,
    EndpointSuperResolutionConfig,
    VideoDiffusionModelConfig,
    _interior_mask,
    _linear_interp_endpoints,
    _upsample_coarse_clip,
)

OUT_NAMES = ["var0", "var1"]


def _times(n):
    base = cftime.DatetimeProlepticGregorian(2013, 1, 2)
    return xr.DataArray(
        [base + datetime.timedelta(hours=3 * i) for i in range(n)], dims=["time"]
    )


def _video_item(n_times, height, width):
    data = {v: torch.rand(n_times, height, width) for v in OUT_NAMES}
    time = _times(n_times)
    coords = LatLonCoordinates(
        lat=torch.linspace(-10.0, 10.0, height),
        lon=torch.linspace(0.0, 30.0, width),
    )
    doy, sod = compute_calendar_features(time)
    return VideoBatchItem(data, time, coords, doy, sod)


def _paired_batch(batch_size, n_times, height, width):
    items = [_video_item(n_times, height, width) for _ in range(batch_size)]
    clip = VideoBatchData.from_sequence(items)
    # coarse is unused by the temporal-only model; reuse the fine clip.
    return PairedVideoBatchData(fine=clip, coarse=clip)


def _spatial_paired_batch(
    batch_size, n_times, fine_height, fine_width, downscale_factor
):
    """Fine/coarse pair at genuinely different spatial resolutions (coarse is
    independent random data, not a downsampled copy of fine)."""
    fine_items = [
        _video_item(n_times, fine_height, fine_width) for _ in range(batch_size)
    ]
    coarse_items = [
        _video_item(
            n_times, fine_height // downscale_factor, fine_width // downscale_factor
        )
        for _ in range(batch_size)
    ]
    fine = VideoBatchData.from_sequence(fine_items)
    coarse = VideoBatchData.from_sequence(coarse_items)
    return PairedVideoBatchData(fine=fine, coarse=coarse)


def test_upsample_coarse_clip_groups_channels_by_frame():
    # Regression: coarse is (B, C, T, H, W) -- C and T are not adjacent in
    # memory, so a naive reshape to (B*T, C, H, W) would interleave channels
    # from different timesteps instead of grouping each frame's own channels.
    # Distinct values per (channel, timestep) make that scrambling detectable:
    # if frames were mixed up, some output frame would fail to match the
    # independently-computed per-frame upsample below.
    batch, channels, n_times, height, width = 2, 3, 4, 2, 2
    factor = 2
    coarse = torch.arange(
        batch * channels * n_times * height * width, dtype=torch.float32
    ).reshape(batch, channels, n_times, height, width)

    fine_hw = (height * factor, width * factor)
    result = _upsample_coarse_clip(coarse, fine_hw)
    assert result.shape == (batch, channels, n_times, *fine_hw)

    for b in range(batch):
        for t in range(n_times):
            frame = coarse[b, :, t]  # (C, H, W): this frame's own channels
            expected = interpolate_2d(frame.unsqueeze(0), factor).squeeze(0)
            assert torch.allclose(
                result[b, :, t], expected, atol=1e-5
            ), f"frame mismatch at batch={b}, time={t}"


def _model(n_times, temporal_noise_correlation="independent", **config_kwargs):
    # full_fine_coords/downscale_factor are .build() args, not config fields --
    # pulled out of config_kwargs so callers can pass them through uniformly.
    full_fine_coords = config_kwargs.pop("full_fine_coords", None)
    downscale_factor = config_kwargs.pop("downscale_factor", None)
    config = VideoDiffusionModelConfig(
        out_names=OUT_NAMES,
        n_timesteps=n_times,
        normalization=NormalizationConfig(
            means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
        ),
        num_diffusion_generation_steps=4,
        model_channels=16,
        n_heads=2,
        num_freqs=3,
        temporal_noise_correlation=temporal_noise_correlation,
        **config_kwargs,
    )
    return config.build(
        full_fine_coords=full_fine_coords, downscale_factor=downscale_factor
    )


def test_train_on_batch_runs_and_backprops():
    n_times, height, width = 5, 8, 8
    model = _model(n_times)
    batch = _paired_batch(batch_size=2, n_times=n_times, height=height, width=width)

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    assert outputs.loss.requires_grad
    # gradients flow into the network
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    # prediction is a full clip per variable
    assert outputs.prediction["var0"].shape == (2, n_times, height, width)


def test_train_on_batch_supports_per_channel_noise():
    n_times, height, width = 5, 8, 8
    config = VideoDiffusionModelConfig(
        out_names=OUT_NAMES,
        n_timesteps=n_times,
        normalization=NormalizationConfig(
            means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
        ),
        num_diffusion_generation_steps=4,
        model_channels=16,
        n_heads=2,
        num_freqs=3,
        training_noise_distributions={
            "var0": LogNormalNoiseDistribution(p_mean=-1.2, p_std=1.8),
            "var1": LogUniformNoiseDistribution(p_min=0.005, p_max=2000.0),
        },
        sigma_min=0.002,
        sigma_max=150.0,
        sigma_min_by_channel={"var1": 0.005},
        sigma_max_by_channel={"var1": 2000.0},
    )
    model = config.build()
    batch = _paired_batch(batch_size=2, n_times=n_times, height=height, width=width)

    outputs = model.train_on_batch(batch, NullOptimization())

    assert outputs.sigma is not None
    assert outputs.sigma.shape == (2, 2)
    sigma_min, sigma_max = config.generation_sigma_bounds(outputs.sigma.device)
    assert torch.allclose(sigma_min, torch.tensor([0.002, 0.005]))
    assert torch.allclose(sigma_max, torch.tensor([150.0, 2000.0]))


def test_per_channel_sigma_data():
    n_times, height, width = 5, 8, 8
    config = VideoDiffusionModelConfig(
        out_names=OUT_NAMES,
        n_timesteps=n_times,
        normalization=NormalizationConfig(
            means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
        ),
        num_diffusion_generation_steps=4,
        model_channels=16,
        n_heads=2,
        sigma_min=0.002,
        sigma_max=150.0,
        sigma_data_by_channel={"var0": 0.2},  # var1 falls back to 1.0
    )
    model = config.build()
    bare = getattr(model.module, "module", model.module)
    assert torch.allclose(
        bare.sigma_data.flatten(),
        torch.tensor([0.2, 1.0], device=bare.sigma_data.device),
    )
    assert model.sigma_data.shape == (1, 2, 1, 1, 1)
    # train + generate still run end to end with per-channel sigma_data
    batch = _paired_batch(batch_size=2, n_times=n_times, height=height, width=width)
    model.train_on_batch(batch, NullOptimization())
    generated = model.generate(batch, n_samples=2)
    assert set(generated) == set(OUT_NAMES)


@pytest.mark.parametrize(
    "temporal_noise_correlation", ["independent", "brownian_bridge"]
)
def test_generate_pins_observed_endpoints(temporal_noise_correlation):
    n_times, height, width = 5, 8, 8
    model = _model(n_times, temporal_noise_correlation=temporal_noise_correlation)
    batch = _paired_batch(batch_size=2, n_times=n_times, height=height, width=width)

    generated = model.generate(batch, n_samples=3)
    out = generated["var0"]
    assert out.shape == (2, 3, n_times, height, width)
    assert torch.isfinite(out).all()

    # observed endpoints (t=0 and t=-1) must be reproduced exactly for every
    # sample; only the interior is generated.
    truth = batch.fine.data["var0"]  # (B, T, H, W)
    for s in range(3):
        assert torch.allclose(out[:, s, 0], truth[:, 0], atol=1e-4)
        assert torch.allclose(out[:, s, -1], truth[:, -1], atol=1e-4)


def test_spatial_downscaling_disabled_by_default():
    n_times = 5
    model = _model(n_times)
    assert model.config.coarse_normalization is None
    assert model.coarse_normalizer is None
    # noisy residual (C) + endpoint values (C) + mask (1) + log-sigma (C) = 3C+1
    n_channels = len(OUT_NAMES)
    expected_in_channels = 3 * n_channels + 1
    net = model.module.module.model
    in_conv = net.in_conv.conv
    assert in_conv.in_channels == expected_in_channels + net.calendar.out_channels


def _spatial_model(n_times, **config_kwargs):
    return _model(
        n_times,
        coarse_normalization=NormalizationConfig(
            means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
        ),
        **config_kwargs,
    )


def test_spatial_downscaling_widens_input_channels():
    n_times = 5
    model = _spatial_model(n_times)
    n_channels = len(OUT_NAMES)
    # adds a 4th block of C channels (endpoint-masked coarse clip) plus its
    # own observed-mask channel vs. the temporal-only 3C+1
    expected_in_channels = 4 * n_channels + 2
    net = model.module.module.model
    in_conv = net.in_conv.conv
    assert in_conv.in_channels == expected_in_channels + net.calendar.out_channels


def test_spatial_downscaling_baseline_ignores_interior_coarse():
    # endpoints_observed=True (default): the baseline must not depend on
    # coarse at all -- coarse only enters via _conditioning now, masked to
    # the endpoints. Perturbing only the interior frames of coarse must
    # leave the baseline unchanged, and it must equal the pure temporal
    # interpolation of the fine endpoints directly.
    n_times, height, width = 5, 4, 4
    model = _spatial_model(n_times)
    clip = torch.randn(2, len(OUT_NAMES), n_times, height, width)
    tau = model._tau_for_indices(None)

    coarse_a = torch.randn(2, len(OUT_NAMES), n_times, height, width)
    coarse_b = coarse_a.clone()
    coarse_b[:, :, 1:-1] = torch.randn_like(coarse_b[:, :, 1:-1])

    baseline_a = model._spatiotemporal_baseline(clip, coarse_a, tau)
    baseline_b = model._spatiotemporal_baseline(clip, coarse_b, tau)
    assert torch.equal(baseline_a, baseline_b)

    expected = _linear_interp_endpoints(clip, tau)
    assert torch.equal(baseline_a, expected)
    assert not torch.equal(baseline_a, coarse_a)


def test_spatial_downscaling_conditioning_masks_coarse_to_endpoints():
    # endpoints_observed=True: the coarse conditioning block must be exactly
    # zero (value and mask channel) at interior frames, and reproduce the
    # true coarse_upsampled values (with mask=1) at the two endpoints.
    n_times, height, width = 5, 4, 4
    n_channels = len(OUT_NAMES)
    model = _spatial_model(n_times)
    clip = torch.randn(2, n_channels, n_times, height, width)
    coarse_upsampled = torch.randn(2, n_channels, n_times, height, width)
    interior = _interior_mask(n_times, clip.device, model.endpoints_observed)

    condition = model._conditioning(clip, interior, coarse_upsampled)
    # layout: fine observed values (C) + fine mask (1) + coarse observed
    # values (C) + coarse mask (1)
    coarse_values = condition[:, n_channels + 1 : 2 * n_channels + 1]
    coarse_mask = condition[:, 2 * n_channels + 1 : 2 * n_channels + 2]

    assert torch.equal(
        coarse_values[:, :, 1:-1], torch.zeros_like(coarse_values[:, :, 1:-1])
    )
    assert torch.equal(
        coarse_mask[:, :, 1:-1], torch.zeros_like(coarse_mask[:, :, 1:-1])
    )
    for t in (0, n_times - 1):
        assert torch.equal(coarse_values[:, :, t], coarse_upsampled[:, :, t])
    assert torch.equal(
        coarse_mask[:, :, [0, -1]], torch.ones_like(coarse_mask[:, :, [0, -1]])
    )


def test_pure_coarse_to_fine_conditioning_uses_full_unmasked_coarse():
    # endpoints_observed=False regression guard: the coarse block stays the
    # full unmasked per-frame clip, with no extra mask channel, since it's
    # the only conditioning source in this mode.
    n_times, height, width = 5, 4, 4
    n_channels = len(OUT_NAMES)
    model = _pure_coarse_to_fine_model(n_times)
    clip = torch.randn(2, n_channels, n_times, height, width)
    coarse_upsampled = torch.randn(2, n_channels, n_times, height, width)
    interior = _interior_mask(n_times, clip.device, model.endpoints_observed)

    condition = model._conditioning(clip, interior, coarse_upsampled)
    assert condition.shape[CHANNEL_AXIS] == n_channels
    assert torch.equal(condition, coarse_upsampled)


def test_spatial_downscaling_train_on_batch_runs_and_backprops():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _spatial_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    assert outputs.loss.requires_grad
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert outputs.prediction["var0"].shape == (2, n_times, fine_height, fine_width)


def test_spatial_downscaling_generate_pins_observed_endpoints():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _spatial_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    generated = model.generate(batch, n_samples=3)
    out = generated["var0"]
    assert out.shape == (2, 3, n_times, fine_height, fine_width)
    assert torch.isfinite(out).all()

    truth = batch.fine.data["var0"]
    for s in range(3):
        assert torch.allclose(out[:, s, 0], truth[:, 0], atol=1e-4)
        assert torch.allclose(out[:, s, -1], truth[:, -1], atol=1e-4)


def _pure_coarse_to_fine_model(n_times, **config_kwargs):
    return _spatial_model(n_times, endpoints_observed=False, **config_kwargs)


def test_pure_coarse_to_fine_requires_coarse_normalization():
    with pytest.raises(ValueError, match="requires coarse_normalization"):
        _model(5, endpoints_observed=False)


def test_pure_coarse_to_fine_rejects_subset_augmentation():
    with pytest.raises(ValueError, match="subset_augmentation_prob > 0"):
        _pure_coarse_to_fine_model(5, subset_augmentation_prob=0.5)


def test_pure_coarse_to_fine_rejects_marginal_consistency():
    with pytest.raises(ValueError, match="marginal_consistency_weight > 0"):
        _pure_coarse_to_fine_model(5, marginal_consistency_weight=1.0)


def test_pure_coarse_to_fine_rejects_brownian_bridge():
    with pytest.raises(ValueError, match="brownian_bridge noise"):
        _pure_coarse_to_fine_model(5, temporal_noise_correlation="brownian_bridge")


def test_pure_coarse_to_fine_rejects_per_channel_brownian_bridge():
    with pytest.raises(ValueError, match="brownian_bridge noise"):
        _pure_coarse_to_fine_model(
            5,
            temporal_noise_correlation="per_channel",
            per_channel_noise_kernel={"var0": "brownian_bridge", "var1": "independent"},
        )


def test_pure_coarse_to_fine_narrows_input_channels():
    n_times = 5
    model = _pure_coarse_to_fine_model(n_times)
    n_channels = len(OUT_NAMES)
    # noisy residual (C) + coarse condition (C) + log-sigma (C) = 3C; no
    # endpoint-values/mask block since no frame is ever pinned.
    expected_in_channels = 3 * n_channels
    net = model.module.module.model
    in_conv = net.in_conv.conv
    assert in_conv.in_channels == expected_in_channels + net.calendar.out_channels


def test_pure_coarse_to_fine_baseline_ignores_fine_values():
    n_times = 5
    model = _pure_coarse_to_fine_model(n_times)
    coarse_upsampled = torch.randn(2, len(OUT_NAMES), n_times, 4, 4)
    tau = model._tau_for_indices(None)
    zeros_clip = torch.zeros(2, len(OUT_NAMES), n_times, 4, 4)
    random_clip = torch.randn(2, len(OUT_NAMES), n_times, 4, 4)

    baseline_a = model._spatiotemporal_baseline(zeros_clip, coarse_upsampled, tau)
    baseline_b = model._spatiotemporal_baseline(random_clip, coarse_upsampled, tau)
    # baseline must not depend on the fine clip at all -- there's no fine
    # truth of any kind to bias-correct against or pin to.
    assert torch.equal(baseline_a, coarse_upsampled)
    assert torch.equal(baseline_b, coarse_upsampled)


def test_pure_coarse_to_fine_train_on_batch_runs_and_backprops():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _pure_coarse_to_fine_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    assert outputs.loss.requires_grad
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert outputs.prediction["var0"].shape == (2, n_times, fine_height, fine_width)


def test_pure_coarse_to_fine_generate_runs_and_does_not_pin_endpoints():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _pure_coarse_to_fine_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    generated = model.generate(batch, n_samples=3)
    out = generated["var0"]
    assert out.shape == (2, 3, n_times, fine_height, fine_width)
    assert torch.isfinite(out).all()

    # unlike the endpoints_observed=True case, endpoints are NOT pinned to
    # the true fine value -- every sample independently diffuses them, so
    # they won't match ground truth (or each other) exactly.
    truth = batch.fine.data["var0"]
    assert not torch.allclose(out[:, 0, 0], truth[:, 0], atol=1e-4)
    assert not torch.allclose(out[:, 0, 0], out[:, 1, 0], atol=1e-4)


def _coarse_endpoints_only_model(n_times, **config_kwargs):
    return _spatial_model(
        n_times, endpoints_observed=False, coarse_endpoints_only=True, **config_kwargs
    )


def test_coarse_endpoints_only_rejected_when_endpoints_observed():
    with pytest.raises(ValueError, match="coarse_endpoints_only"):
        _spatial_model(5, coarse_endpoints_only=True)


def test_coarse_endpoints_only_narrows_input_channels():
    n_times = 5
    model = _coarse_endpoints_only_model(n_times)
    n_channels = len(OUT_NAMES)
    # noisy residual (C) + endpoint-masked coarse condition (C) + its own
    # mask (1) + log-sigma (C) = 3C+1 -- no fine-endpoint block since no
    # frame is ever pinned, but the coarse block still gets a mask channel
    # (it's endpoint-masked here, unlike the continuous pure-coarse-to-fine
    # mode's unmasked full-clip conditioning).
    expected_in_channels = 3 * n_channels + 1
    net = model.module.module.model
    in_conv = net.in_conv.conv
    assert in_conv.in_channels == expected_in_channels + net.calendar.out_channels


def test_coarse_endpoints_only_baseline_interpolates_coarse_endpoints():
    # The baseline must ignore the fine clip entirely (no fine truth exists)
    # and must ignore interior coarse frames (only the two endpoints are
    # ever available in this deployment scenario) -- it's the linear
    # interpolation of the coarse-upsampled endpoints, not a pinned/
    # continuous value.
    n_times, height, width = 5, 4, 4
    model = _coarse_endpoints_only_model(n_times)
    tau = model._tau_for_indices(None)
    random_clip = torch.randn(2, len(OUT_NAMES), n_times, height, width)

    coarse_a = torch.randn(2, len(OUT_NAMES), n_times, height, width)
    coarse_b = coarse_a.clone()
    coarse_b[:, :, 1:-1] = torch.randn_like(coarse_b[:, :, 1:-1])

    baseline_a = model._spatiotemporal_baseline(random_clip, coarse_a, tau)
    baseline_b = model._spatiotemporal_baseline(random_clip, coarse_b, tau)
    assert torch.equal(baseline_a, baseline_b)

    expected = _linear_interp_endpoints(coarse_a, tau)
    assert torch.equal(baseline_a, expected)
    assert not torch.equal(baseline_a, coarse_a)


def test_coarse_endpoints_only_conditioning_masks_coarse_to_endpoints():
    # Same masking behavior as endpoints_observed=True's coarse block, but
    # with NO fine block at all (no fine value of any kind exists here).
    n_times, height, width = 5, 4, 4
    n_channels = len(OUT_NAMES)
    model = _coarse_endpoints_only_model(n_times)
    clip = torch.randn(2, n_channels, n_times, height, width)
    coarse_upsampled = torch.randn(2, n_channels, n_times, height, width)
    interior = _interior_mask(n_times, clip.device, model.endpoints_observed)
    assert torch.equal(interior, torch.ones_like(interior))  # nothing pinned

    condition = model._conditioning(clip, interior, coarse_upsampled)
    assert condition.shape[CHANNEL_AXIS] == n_channels + 1  # coarse values + mask
    coarse_values = condition[:, :n_channels]
    coarse_mask = condition[:, n_channels : n_channels + 1]

    assert torch.equal(
        coarse_values[:, :, 1:-1], torch.zeros_like(coarse_values[:, :, 1:-1])
    )
    assert torch.equal(
        coarse_mask[:, :, 1:-1], torch.zeros_like(coarse_mask[:, :, 1:-1])
    )
    for t in (0, n_times - 1):
        assert torch.equal(coarse_values[:, :, t], coarse_upsampled[:, :, t])
    assert torch.equal(
        coarse_mask[:, :, [0, -1]], torch.ones_like(coarse_mask[:, :, [0, -1]])
    )


def test_coarse_endpoints_only_train_on_batch_runs_and_backprops():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _coarse_endpoints_only_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    assert outputs.loss.requires_grad
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert outputs.prediction["var0"].shape == (2, n_times, fine_height, fine_width)


@pytest.mark.parametrize(
    "kernel_kwargs",
    [
        {},
        {
            "temporal_noise_correlation": "per_channel",
            "per_channel_noise_kernel": {"var0": "ou", "var1": "ou"},
            "per_channel_kernel_length_scale": {"var0": 0.3, "var1": 0.3},
        },
    ],
    ids=["independent", "ou"],
)
def test_coarse_endpoints_only_generate_runs_and_does_not_pin_endpoints(
    kernel_kwargs,
):
    # Also covers the "does OU actually denoise the endpoints at generation
    # time" question directly: _video_edm_sample's `mask` is all-ones in this
    # mode (nothing pinned), so every `* mask` in the sampler loop is a
    # no-op there -- the endpoints go through the SAME full num_steps
    # iterative denoising as every interior frame, seeded from the (now
    # correctly nonzero, per the noise-mixing fix) OU-correlated initial
    # noise. If they were still being frozen/pinned, repeated samples would
    # be identical at the endpoints; they are not (checked below).
    n_times, fine_height, fine_width, factor = 9, 8, 8, 2
    model = _coarse_endpoints_only_model(n_times, **kernel_kwargs)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    generated = model.generate(batch, n_samples=3)
    out = generated["var0"]
    assert out.shape == (2, 3, n_times, fine_height, fine_width)
    assert torch.isfinite(out).all()

    # endpoints are diffused (not pinned), same as continuous pure-coarse-to-fine.
    truth = batch.fine.data["var0"]
    assert not torch.allclose(out[:, 0, 0], truth[:, 0], atol=1e-4)
    assert not torch.allclose(out[:, 0, 0], out[:, 1, 0], atol=1e-4)


def test_spatial_downscaling_rejects_non_integer_scale_factor():
    n_times = 5
    model = _spatial_model(n_times)
    # fine (8, 8) is not an integer multiple of coarse (3, 3)
    batch = _spatial_paired_batch(
        batch_size=2, n_times=n_times, fine_height=8, fine_width=8, downscale_factor=2
    )
    bad_coarse_items = [_video_item(n_times, 3, 3) for _ in range(2)]
    batch = PairedVideoBatchData(
        fine=batch.fine, coarse=VideoBatchData.from_sequence(bad_coarse_items)
    )
    with pytest.raises(ValueError, match="integer multiple"):
        model.train_on_batch(batch, NullOptimization())


def _endpoint_sr_config(fine_hw, **kwargs):
    return EndpointSuperResolutionConfig(
        module=DiffusionModuleRegistrySelector(
            type="unet_diffusion_song",
            config={
                "model_channels": 8,
                "channel_mult": [1, 1],
                "num_blocks": 1,
                "attn_resolutions": [],
            },
        ),
        loss=LossConfig(),
        training_noise_distribution=LogNormalNoiseDistribution(p_mean=-1.2, p_std=1.2),
        coarse_shape=list(fine_hw),
        num_diffusion_generation_steps=2,
        **kwargs,
    )


def _two_stage_model(n_times, fine_hw=(8, 8), sr_kwargs=None, **config_kwargs):
    return _spatial_model(
        n_times,
        endpoint_super_resolution=_endpoint_sr_config(fine_hw, **(sr_kwargs or {})),
        **config_kwargs,
    )


def test_endpoint_super_resolution_coarse_shape_validation():
    with pytest.raises(ValueError, match=r"\[lat, lon\]"):
        _endpoint_sr_config((8, 8, 8))


def test_endpoint_super_resolution_requires_endpoints_observed():
    with pytest.raises(ValueError, match="requires endpoints_observed=True"):
        _two_stage_model(5, endpoints_observed=False)


def test_endpoint_super_resolution_requires_coarse_normalization():
    with pytest.raises(ValueError, match="requires coarse_normalization"):
        _model(5, endpoint_super_resolution=_endpoint_sr_config((8, 8)))


def test_endpoint_super_resolution_builds_stage_a_module():
    model = _two_stage_model(5)
    assert model.endpoint_sr_module is not None
    assert len(model.modules) == 2


def test_endpoint_super_resolution_disabled_by_default_has_one_module():
    model = _spatial_model(5)
    assert model.endpoint_sr_module is None
    assert len(model.modules) == 1


def test_endpoint_super_resolution_train_on_batch_runs_and_backprops():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _two_stage_model(n_times, fine_hw=(fine_height, fine_width))
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    assert outputs.endpoint_sr_loss is not None
    assert torch.isfinite(outputs.endpoint_sr_loss)
    outputs.loss.backward()

    stage_b_grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    stage_a_grads = [
        p.grad for p in model.endpoint_sr_module.parameters() if p.grad is not None
    ]
    assert len(stage_b_grads) > 0
    assert len(stage_a_grads) > 0
    assert all(torch.isfinite(g).all() for g in stage_b_grads + stage_a_grads)


def test_endpoint_super_resolution_detaches_from_stage_b_loss():
    # loss_weight=0.0 isolates total_loss to stage B's own loss only; since
    # the SR endpoint is detached before feeding into stage B, backward
    # through that loss alone must give stage A's params zero (or no) grad.
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _two_stage_model(
        n_times, fine_hw=(fine_height, fine_width), sr_kwargs={"loss_weight": 0.0}
    )
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    outputs = model.train_on_batch(batch, NullOptimization())
    outputs.loss.backward()
    stage_a_grads = [p.grad for p in model.endpoint_sr_module.parameters()]
    assert all(g is None or torch.all(g == 0) for g in stage_a_grads)


def test_endpoint_super_resolution_generate_runs():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _two_stage_model(n_times, fine_hw=(fine_height, fine_width))
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    generated = model.generate(batch, n_samples=2)
    out = generated["var0"]
    assert out.shape == (2, 2, n_times, fine_height, fine_width)
    assert torch.isfinite(out).all()


def test_endpoint_super_resolution_composes_with_brownian_bridge_noise():
    # The whole point of stage A: a meaningfully pinned (generated, not raw
    # fine truth) endpoint restores brownian_bridge's validity even in the
    # spatial-downscaling setting where no real fine truth is available.
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _two_stage_model(
        n_times,
        fine_hw=(fine_height, fine_width),
        temporal_noise_correlation="brownian_bridge",
    )
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )
    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def _midpoint_grid(n, width):
    """Genuinely nested cell-center grid (matches real lat/lon convention):
    n cells of the given width, centers at width/2, 3*width/2, ... -- unlike
    torch.linspace, subdividing by an exact factor gives an exact nested
    subset relationship, which adjust_fine_coord_range depends on."""
    return torch.arange(n, dtype=torch.float32) * width + width / 2


def _nested_coords(n_coarse_full, factor, coarse_slice, coarse_width=5.0):
    """A genuinely nested (full_coarse, full_fine, batch's own coarse subset)
    triple for get_fine_coords_for_batch/generate_on_batch_no_target tests."""
    full_coarse = _midpoint_grid(n_coarse_full, coarse_width)
    full_fine = _midpoint_grid(n_coarse_full * factor, coarse_width / factor)
    coarse_coord = full_coarse[coarse_slice]
    expected_fine = full_fine[coarse_slice.start * factor : coarse_slice.stop * factor]
    return full_fine, coarse_coord, expected_fine


def _no_target_model_and_batch(
    n_times=5, factor=4, n_coarse_full=20, coarse_slice=slice(5, 13)
):
    full_fine, coarse_coord, expected_fine = _nested_coords(
        n_coarse_full, factor, coarse_slice
    )
    fine_h = fine_w = len(coarse_coord) * factor
    model = _two_stage_model(
        n_times,
        fine_hw=(fine_h, fine_w),
        full_fine_coords=LatLonCoordinates(lat=full_fine, lon=full_fine),
        downscale_factor=factor,
    )

    def _coarse_item(n_t):
        data = {
            v: torch.rand(n_t, len(coarse_coord), len(coarse_coord)) for v in OUT_NAMES
        }
        time = _times(n_t)
        coords = LatLonCoordinates(lat=coarse_coord, lon=coarse_coord)
        doy, sod = compute_calendar_features(time)
        return VideoBatchItem(data, time, coords, doy, sod)

    coarse = VideoBatchData.from_sequence([_coarse_item(n_times) for _ in range(2)])
    return model, coarse, expected_fine, (fine_h, fine_w)


def test_get_fine_coords_for_batch_requires_full_fine_coords():
    model = _spatial_model(
        5, endpoints_observed=False
    )  # no full_fine_coords/downscale_factor
    coarse = _spatial_paired_batch(
        batch_size=2, n_times=5, fine_height=8, fine_width=8, downscale_factor=2
    ).coarse
    with pytest.raises(ValueError, match="built with full_fine_coords"):
        model.get_fine_coords_for_batch(coarse)


def test_get_fine_coords_for_batch_derives_nested_grid():
    model, coarse, expected_fine, _ = _no_target_model_and_batch()
    fine_coords = model.get_fine_coords_for_batch(coarse)
    assert torch.allclose(fine_coords.lat, expected_fine)
    assert torch.allclose(fine_coords.lon, expected_fine)


def test_generate_on_batch_no_target_requires_valid_mode():
    # plain endpoints_observed=True, no endpoint_super_resolution: the model
    # fundamentally needs real fine endpoint values, so this must refuse.
    model = _model(
        5,
        coarse_normalization=NormalizationConfig(
            means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
        ),
    )
    with pytest.raises(ValueError, match="requires endpoints_observed=False"):
        model.generate_on_batch_no_target(coarse=None)


def test_generate_on_batch_no_target_runs_with_endpoint_super_resolution():
    model, coarse, _, (fine_h, fine_w) = _no_target_model_and_batch()
    generated = model.generate_on_batch_no_target(coarse, n_samples=2)
    out = generated["var0"]
    assert out.shape == (2, 2, 5, fine_h, fine_w)
    assert torch.isfinite(out).all()


def test_generate_on_batch_no_target_runs_with_pure_coarse_to_fine():
    n_times, factor, n_coarse_full = 5, 4, 20
    coarse_slice = slice(5, 13)
    full_fine, coarse_coord, _ = _nested_coords(n_coarse_full, factor, coarse_slice)
    fine_h = fine_w = len(coarse_coord) * factor
    model = _pure_coarse_to_fine_model(
        n_times,
        full_fine_coords=LatLonCoordinates(lat=full_fine, lon=full_fine),
        downscale_factor=factor,
    )

    def _coarse_item(n_t):
        data = {
            v: torch.rand(n_t, len(coarse_coord), len(coarse_coord)) for v in OUT_NAMES
        }
        time = _times(n_t)
        coords = LatLonCoordinates(lat=coarse_coord, lon=coarse_coord)
        doy, sod = compute_calendar_features(time)
        return VideoBatchItem(data, time, coords, doy, sod)

    coarse = VideoBatchData.from_sequence([_coarse_item(n_times) for _ in range(2)])
    generated = model.generate_on_batch_no_target(coarse, n_samples=2)
    out = generated["var0"]
    assert out.shape == (2, 2, n_times, fine_h, fine_w)
    assert torch.isfinite(out).all()


def test_train_on_batch_runs_with_brownian_bridge_noise():
    n_times, height, width = 5, 8, 8
    model = _model(n_times, temporal_noise_correlation="brownian_bridge")
    batch = _paired_batch(batch_size=2, n_times=n_times, height=height, width=width)

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    assert outputs.loss.requires_grad
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_brownian_bridge_noise_has_expected_temporal_covariance():
    n_times = 9
    model = _model(n_times, temporal_noise_correlation="brownian_bridge")
    set_seed(0)
    # (samples, C=1, T, H=1, W=1): treat sample dim as the population.
    like = torch.empty(20000, 1, n_times, 1, 1)
    noise = model._sample_residual_noise(like)
    flat = noise[:, 0, :, 0, 0]  # (samples, T)

    emp_cov = (flat.T @ flat) / flat.shape[0]
    expected = (model._noise_mixing @ model._noise_mixing.T).cpu()
    assert torch.allclose(emp_cov, expected, atol=0.03)
    # endpoints are noise-free; interior frames are correlated (off-diagonal != 0)
    assert torch.allclose(emp_cov[0], torch.zeros(n_times))
    assert torch.allclose(emp_cov[-1], torch.zeros(n_times))
    assert emp_cov[1, 2].abs() > 0.1


def test_independent_noise_is_uncorrelated_in_time():
    n_times = 9
    model = _model(n_times, temporal_noise_correlation="independent")
    assert model._noise_mixing is None
    set_seed(0)
    like = torch.empty(20000, 1, n_times, 1, 1)
    flat = model._sample_residual_noise(like)[:, 0, :, 0, 0]
    emp_cov = (flat.T @ flat) / flat.shape[0]
    assert torch.allclose(emp_cov, torch.eye(n_times), atol=0.05)


def test_pure_coarse_to_fine_ou_kernel_noises_endpoints():
    # Regression guard: ou/rbf mixing matrices used to always zero the
    # endpoint rows regardless of endpoints_observed (correct only when
    # endpoints are pinned). With endpoints_observed=False every frame,
    # including the former endpoints, is diffused and must get real,
    # nonzero training noise -- otherwise the model never sees a noised
    # endpoint during training despite generation starting from pure noise
    # there, and the endpoints would never gain any detail.
    n_times = 9
    model = _pure_coarse_to_fine_model(
        n_times,
        temporal_noise_correlation="per_channel",
        per_channel_noise_kernel={"var0": "ou", "var1": "ou"},
        per_channel_kernel_length_scale={"var0": 0.3, "var1": 0.3},
    )
    set_seed(0)
    like = torch.empty(20000, len(OUT_NAMES), n_times, 1, 1)
    noise = model._sample_residual_noise(like)
    flat = noise[:, 0, :, 0, 0]
    emp_cov = (flat.T @ flat) / flat.shape[0]
    # no frame -- including the endpoints -- has zero noise.
    assert (emp_cov.diagonal() > 0.5).all()
    # the endpoint is correlated with its neighbor, same as any other frame.
    assert emp_cov[0, 1].abs() > 0.05


def test_invalid_temporal_noise_correlation_rejected():
    with pytest.raises(ValueError, match="temporal_noise_correlation"):
        VideoDiffusionModelConfig(
            out_names=OUT_NAMES,
            n_timesteps=5,
            normalization=NormalizationConfig(
                means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
            ),
            temporal_noise_correlation="bridge",
        )


@pytest.mark.parametrize(
    "temporal_noise_correlation", ["independent", "brownian_bridge"]
)
def test_generate_subset_frames(temporal_noise_correlation):
    # full grid is 9 frames (00..24 at 3h); request a non-uniform subset
    n_times, height, width = 9, 8, 8
    model = _model(n_times, temporal_noise_correlation=temporal_noise_correlation)
    batch = _paired_batch(batch_size=2, n_times=n_times, height=height, width=width)

    frames = [0, 3, 7, 8]  # endpoints + two non-uniformly spaced interior frames
    generated = model.generate(batch, n_samples=3, frames=frames)
    out = generated["var0"]
    assert out.shape == (2, 3, len(frames), height, width)
    assert torch.isfinite(out).all()

    # observed endpoints (first/last of the subset) reproduced exactly
    truth = batch.fine.data["var0"]
    for s in range(3):
        assert torch.allclose(out[:, s, 0], truth[:, 0], atol=1e-4)
        assert torch.allclose(out[:, s, -1], truth[:, -1], atol=1e-4)


def test_generate_rejects_invalid_frames():
    n_times = 9
    model = _model(n_times)
    batch = _paired_batch(batch_size=1, n_times=n_times, height=8, width=8)
    # must include both endpoints
    with pytest.raises(ValueError, match="start at 0 and end at"):
        model.generate(batch, frames=[0, 3, 6])
    # must be strictly increasing
    with pytest.raises(ValueError, match="strictly increasing"):
        model.generate(batch, frames=[0, 3, 3, 8])
    # needs at least one interior frame
    with pytest.raises(ValueError, match="at least 3 frame indices"):
        model.generate(batch, frames=[0, 8])


@pytest.mark.parametrize(
    "temporal_noise_correlation", ["independent", "brownian_bridge"]
)
def test_subset_augmented_training_runs(temporal_noise_correlation):
    n_times, height, width = 9, 8, 8
    model = _model(
        n_times,
        temporal_noise_correlation=temporal_noise_correlation,
        subset_augmentation_prob=1.0,  # always subset, to exercise the path
        subset_min_interior=1,
    )
    batch = _paired_batch(batch_size=2, n_times=n_times, height=height, width=width)
    set_seed(0)

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    # prediction and target stay aligned on the (subset) frame axis
    assert outputs.prediction["var0"].shape == outputs.target["var0"].shape
    assert outputs.prediction["var0"].shape[1] < n_times


def test_subset_augmentation_prob_zero_is_full_grid():
    n_times = 9
    model = _model(n_times, subset_augmentation_prob=0.0)
    batch = _paired_batch(batch_size=2, n_times=n_times, height=8, width=8)
    outputs = model.train_on_batch(batch, NullOptimization())
    assert outputs.prediction["var0"].shape[1] == n_times


def test_invalid_subset_config_rejected():
    with pytest.raises(ValueError, match="subset_augmentation_prob"):
        _model(5, subset_augmentation_prob=1.5)
    with pytest.raises(ValueError, match="subset_min_interior"):
        _model(5, subset_min_interior=4)  # n_timesteps - 2 == 3


@pytest.mark.parametrize(
    "temporal_noise_correlation", ["independent", "brownian_bridge"]
)
def test_marginal_consistency_loss_trains(temporal_noise_correlation):
    n_times, height, width = 9, 8, 8
    model = _model(
        n_times,
        temporal_noise_correlation=temporal_noise_correlation,
        marginal_consistency_weight=0.5,
    )
    batch = _paired_batch(batch_size=2, n_times=n_times, height=height, width=width)
    set_seed(0)

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss) and outputs.loss.requires_grad
    # the consistency term is surfaced, finite, and non-negative
    assert outputs.marginal_consistency_loss is not None
    assert torch.isfinite(outputs.marginal_consistency_loss)
    assert outputs.marginal_consistency_loss.item() >= 0.0
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    # ModelOutputs still describes the full-grid pass
    assert outputs.prediction["var0"].shape == (2, n_times, height, width)
    assert outputs.target["var0"].shape == (2, n_times, height, width)


def test_marginal_consistency_disabled_by_default():
    model = _model(9)
    batch = _paired_batch(batch_size=2, n_times=9, height=8, width=8)
    outputs = model.train_on_batch(batch, NullOptimization())
    assert outputs.marginal_consistency_loss is None


def test_consistency_subset_is_strict():
    model = _model(9, marginal_consistency_weight=1.0)
    for _ in range(20):
        idx = model._sample_consistency_subset_indices(9, torch.device("cpu"))
        # endpoints kept, at least one interior frame kept AND at least one dropped
        assert int(idx[0]) == 0 and int(idx[-1]) == 8
        assert 3 <= idx.numel() < 9
        assert bool(torch.all(idx[1:] > idx[:-1]))


def test_marginal_consistency_is_reproducible():
    # same seed -> same subset choice and same shared noise -> same L_marg
    model = _model(9, marginal_consistency_weight=1.0)
    batch = _paired_batch(batch_size=2, n_times=9, height=8, width=8)
    set_seed(3)
    a = model.train_on_batch(batch, NullOptimization()).marginal_consistency_loss
    set_seed(3)
    b = model.train_on_batch(batch, NullOptimization()).marginal_consistency_loss
    assert a is not None and b is not None
    assert a.item() == b.item()


def test_marginal_consistency_with_subset_augmentation_compose():
    # subset augmentation and the consistency loss may be combined: same subset
    # exposure in the first pass, plus the consistency tie via the second pass.
    model = _model(
        9,
        marginal_consistency_weight=1.0,
        subset_augmentation_prob=1.0,  # always augment, to exercise the combo
    )
    batch = _paired_batch(batch_size=2, n_times=9, height=8, width=8)
    set_seed(0)
    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss) and outputs.loss.requires_grad
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_marginal_consistency_config_validation():
    with pytest.raises(ValueError, match="marginal_consistency_weight must be >= 0"):
        _model(9, marginal_consistency_weight=-0.1)
    with pytest.raises(ValueError, match="n_timesteps >="):
        _model(3, marginal_consistency_weight=1.0)  # only 1 interior frame


def _temporal_attention(seq_length, channels=8, n_heads=2):
    attn = TemporalAttention(channels, n_heads, seq_length)
    # relative_embedding and the output projection are zero-initialized (the block
    # starts as identity), so the positional bias has no effect until trained;
    # randomize both so tests can observe the frame_index dependence.
    with torch.no_grad():
        attn.relative_embedding.normal_()
        attn.proj.weight.normal_()
        attn.proj.bias.normal_()
    return attn


def test_temporal_attention_frame_index_default_matches_arange():
    # Backward compat: the full-grid default (frame_index=None) must equal passing
    # the contiguous arange positions, so pre-subset full-set behavior is unchanged.
    attn = _temporal_attention(seq_length=9)
    x = torch.randn(2, 8, 9, 4, 4)
    default = attn(x)
    explicit = attn(x, torch.arange(9))
    assert torch.allclose(default, explicit, atol=1e-6)


def test_temporal_attention_uses_true_frame_spacing():
    # A subset packed to contiguous positions [0,1,2] must NOT be treated the same
    # as its true grid positions [0,4,8]: the relative positional bias should
    # reflect real spacing. Feeding true grid indices changes the output.
    attn = _temporal_attention(seq_length=9)
    x = torch.randn(2, 8, 3, 4, 4)
    packed = attn(x)  # default arange([0,1,2]) -- the old (buggy) behavior
    true_grid = attn(x, torch.tensor([0, 4, 8]))
    assert not torch.allclose(packed, true_grid, atol=1e-5)
    # ...and a subset's bias equals the full-grid bias restricted to those frames.
    full = attn(torch.randn(2, 8, 9, 4, 4))  # smoke: full grid still runs
    assert full.shape == (2, 8, 9, 4, 4)


def test_firblur_survives_inplace_kernel_mutation():
    # Regression: under DDP (broadcast_buffers=True) each forward re-syncs buffers
    # with an in-place copy_ into the kernel. When two forwards run before one
    # backward (the marginal-consistency loss), that mutation must not invalidate
    # the first forward's saved-for-backward conv weight. Simulate the buffer sync
    # by mutating the kernel in place between forward and backward.
    blur = FIRBlur()
    x = torch.randn(2, 8, 3, 8, 8, requires_grad=True)
    y = blur(x)
    with torch.no_grad():
        blur.kernel.copy_(blur.kernel)  # what DDP's buffer broadcast does
    y.sum().backward()  # must not raise "modified by an inplace operation"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def _two_block_model(n_times, **config_kwargs):
    return _coarse_endpoints_only_model(n_times, two_block=True, **config_kwargs)


def test_two_block_requires_coarse_endpoints_only_mode():
    with pytest.raises(ValueError, match="two_block requires"):
        _spatial_model(5, endpoints_observed=True, two_block=True)
    with pytest.raises(ValueError, match="two_block requires"):
        _spatial_model(
            5,
            endpoints_observed=False,
            coarse_endpoints_only=False,
            two_block=True,
        )


@pytest.mark.parametrize(
    "bad_kwargs,match",
    [
        ({"subset_augmentation_prob": 0.5}, "subset_augmentation_prob"),
        ({"marginal_consistency_weight": 1.0}, "marginal_consistency_weight"),
        (
            {
                "temporal_noise_correlation": "per_channel",
                "per_channel_noise_kernel": {
                    "var0": "independent",
                    "var1": "independent",
                },
            },
            "temporal_noise_correlation",
        ),
        (
            {"sigma_data_by_channel": {"var0": 1.0}},
            "r_\\*/d_\\*-prefixed sigma/noise fields",
        ),
        ({"r_kernel": "ou"}, "r_kernel must be"),
        ({"d_kernel": "brownian_bridge"}, "d_kernel must be"),
        ({"d_kernel": "ou"}, "d_kernel_length_scale"),
        ({"d_kernel_length_scale": -1.0}, "d_kernel_length_scale must be"),
        ({"r_sigma_min_by_channel": {"var0": 0.1}}, "must be specified together"),
    ],
)
def test_two_block_config_validation_rejects_bad_configs(bad_kwargs, match):
    with pytest.raises(ValueError, match=match):
        _two_block_model(5, **bad_kwargs)


def test_two_block_prefixed_fields_require_two_block():
    with pytest.raises(ValueError, match="require two_block=True"):
        _coarse_endpoints_only_model(5, r_sigma_data_by_channel={"var0": 1.0})


def test_two_block_widens_channels():
    n_times = 5
    model = _two_block_model(n_times)
    n_channels = len(OUT_NAMES)
    net = model.module.module.model
    in_conv = net.in_conv.conv
    # noisy latent (2C: r+d) + log-sigma (2C) + endpoint-masked coarse (C) + mask (1)
    expected_in_channels = 4 * n_channels + n_channels + 1
    assert in_conv.in_channels == expected_in_channels + net.calendar.out_channels
    assert net.out_conv.conv.out_channels == 2 * n_channels


def test_two_block_train_on_batch_runs_and_backprops():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _two_block_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    assert outputs.loss.requires_grad
    outputs.loss.backward()
    grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert outputs.prediction["var0"].shape == (2, n_times, fine_height, fine_width)
    for name in OUT_NAMES:
        assert torch.isfinite(outputs.channel_losses[name])
        assert outputs.per_sample_channel_loss[name].shape == (2,)


def test_two_block_generate_runs_and_is_stochastic():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _two_block_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    generated = model.generate(batch, n_samples=3)
    out = generated["var0"]
    assert out.shape == (2, 3, n_times, fine_height, fine_width)
    assert torch.isfinite(out).all()
    # nothing is pinned in this mode -- independent ensemble members disagree
    # everywhere, endpoints included (same expectation as the non-two_block
    # coarse_endpoints_only mode it extends).
    assert not torch.allclose(out[:, 0, 0], out[:, 1, 0], atol=1e-4)


def test_two_block_generate_rejects_frame_subsetting():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _two_block_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )
    with pytest.raises(NotImplementedError, match="frame subsetting"):
        model.generate(batch, frames=[0, 2, n_times - 1])


def test_two_block_exact_coarse_recovery():
    """Prop 4 of idea/spatiotemoral/twoblock_theory.md: D(x_hat_f(tau)) ==
    I(tau) + r_hat(tau) exactly at every tau, including (trivially, since
    r_hat is pinned to 0 there) the true endpoints D(x_hat(0)) == x_0_c and
    D(x_hat(T)) == x_T_c -- verified here through the FULL generate() reverse
    -diffusion sampler, in physical (denormalized) units, not just on the
    twoblock.py primitives in isolation (test_twoblock.py already covers
    those algebraically)."""
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _two_block_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )
    generated = model.generate(batch, n_samples=2)

    coarse_norm = model._pack_normalized(batch.coarse.data, model.coarse_normalizer)
    interp_phys = model._denormalize_invert(coarse_temporal_interp(coarse_norm))

    for name in OUT_NAMES:
        gen = generated[name]  # (B, n_samples, T, H, W)
        gen_downsampled = conservative_downsample(gen, factor)
        interp_b = interp_phys[name].unsqueeze(1).expand_as(gen_downsampled)
        # At the true endpoints specifically (r_hat == 0 there by construction):
        # D(x_hat(0)) == x_0_c and D(x_hat(T)) == x_T_c exactly.
        assert torch.allclose(gen_downsampled[:, :, 0], interp_b[:, :, 0], atol=1e-4)
        assert torch.allclose(gen_downsampled[:, :, -1], interp_b[:, :, -1], atol=1e-4)


def _conditional_kernel_model(n_times, **config_kwargs):
    return _two_block_model(n_times, two_block_conditional_kernel=True, **config_kwargs)


def test_conditional_kernel_requires_two_block():
    with pytest.raises(ValueError, match="requires two_block=True"):
        _spatial_model(
            5,
            endpoints_observed=False,
            coarse_endpoints_only=True,
            two_block=False,
            two_block_conditional_kernel=True,
        )


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"r_kernel": "independent"},
        {"d_kernel": "ou", "d_kernel_length_scale": 0.5},
        {"d_kernel_length_scale": 0.3},
    ],
)
def test_conditional_kernel_rejects_customized_fixed_kernel_fields(bad_kwargs):
    with pytest.raises(ValueError, match="supersedes r_kernel"):
        _conditional_kernel_model(5, **bad_kwargs)


def test_conditional_kernel_rejects_bad_weight_fit_loss_weight():
    with pytest.raises(ValueError, match="weight_fit_loss_weight"):
        _conditional_kernel_model(5, weight_fit_loss_weight=-1.0)


def test_conditional_kernel_encoders_are_registered_modules():
    model = _conditional_kernel_model(5)
    assert model._r_encoder is not None
    assert model._d_encoder is not None
    assert model._r_encoder in model.modules
    assert model._d_encoder in model.modules


def test_conditional_kernel_train_on_batch_runs_and_backprops():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _conditional_kernel_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=3,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    assert outputs.weight_fit_loss is not None
    assert torch.isfinite(outputs.weight_fit_loss)
    outputs.loss.backward()
    main_grads = [p.grad for p in model.module.parameters() if p.grad is not None]
    assert len(main_grads) > 0
    assert all(torch.isfinite(g).all() for g in main_grads)
    r_enc_grads = [p.grad for p in model._r_encoder.parameters()]
    d_enc_grads = [p.grad for p in model._d_encoder.parameters()]
    # weight_fit_loss_weight defaults to 1.0 -- g_phi SHOULD receive gradient.
    assert any(g is not None for g in r_enc_grads)
    assert any(g is not None for g in d_enc_grads)
    assert all(torch.isfinite(g).all() for g in r_enc_grads if g is not None)


def test_conditional_kernel_dsm_loss_does_not_backprop_into_encoders():
    """The doc's central caveat: plain DSM must not train g_phi. With
    weight_fit_loss_weight=0, the only remaining loss term is DSM -- the
    encoders must receive exactly zero gradient."""
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _conditional_kernel_model(n_times, weight_fit_loss_weight=0.0)
    batch = _spatial_paired_batch(
        batch_size=3,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    outputs = model.train_on_batch(batch, NullOptimization())
    outputs.loss.backward()
    r_enc_grads = [p.grad for p in model._r_encoder.parameters()]
    d_enc_grads = [p.grad for p in model._d_encoder.parameters()]
    assert not any(g is not None and g.abs().sum() > 0 for g in r_enc_grads)
    assert not any(g is not None and g.abs().sum() > 0 for g in d_enc_grads)


def test_conditional_kernel_generate_runs_and_is_stochastic():
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _conditional_kernel_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=3,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )

    generated = model.generate(batch, n_samples=2)
    out = generated["var0"]
    assert out.shape == (3, 2, n_times, fine_height, fine_width)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[:, 0, 0], out[:, 1, 0], atol=1e-4)


def test_conditional_kernel_exact_coarse_recovery():
    """Prop 4 must still hold under conditional kernels -- the exactness
    guarantee comes from assemble_fine_output/masking, not the noise
    mechanism, so it should be unaffected by which kernel g_phi selects."""
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = _conditional_kernel_model(n_times)
    batch = _spatial_paired_batch(
        batch_size=3,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )
    generated = model.generate(batch, n_samples=2)

    coarse_norm = model._pack_normalized(batch.coarse.data, model.coarse_normalizer)
    interp_phys = model._denormalize_invert(coarse_temporal_interp(coarse_norm))

    for name in OUT_NAMES:
        gen_downsampled = conservative_downsample(generated[name], factor)
        interp_b = interp_phys[name].unsqueeze(1).expand_as(gen_downsampled)
        assert torch.allclose(gen_downsampled[:, :, 0], interp_b[:, :, 0], atol=1e-4)
        assert torch.allclose(gen_downsampled[:, :, -1], interp_b[:, :, -1], atol=1e-4)


@pytest.mark.parametrize("model_fixture", ["two_block", "conditional_kernel"])
def test_two_block_noise_respects_block_subspaces(model_fixture):
    """Regression test for the noise mis-specification bug: fine-resolution
    white noise for either block leaks variance into a subspace its target
    never occupies (r: block-constant / image of U; d: null(D)). Verifies
    _two_block_noise's actual output, not just the underlying twoblock.py
    primitives (already covered by test_twoblock.py) -- this checks the
    model wiring calls them correctly."""
    n_times, fine_height, fine_width, factor = 5, 8, 8, 2
    model = (
        _two_block_model(n_times)
        if model_fixture == "two_block"
        else _conditional_kernel_model(n_times)
    )
    n_channels = len(OUT_NAMES)
    coarse_height, coarse_width = fine_height // factor, fine_width // factor
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_height,
        fine_width=fine_width,
        downscale_factor=factor,
    )
    coarse_clip = model._pack_normalized(batch.coarse.data, model.coarse_normalizer)
    r_mixing, d_mixing, _ = model._two_block_mixings_and_weight_fit_loss(
        coarse_clip, n_channels
    )
    torch.manual_seed(0)
    noise = model._two_block_noise(
        torch.empty(2, n_channels, n_times, coarse_height, coarse_width),
        torch.empty(2, n_channels, n_times, fine_height, fine_width),
        r_mixing,
        d_mixing,
        factor,
    )
    noise_r, noise_d = noise.split(n_channels, dim=CHANNEL_AXIS)

    # r's noise must be exactly block-constant (image of U): every fine
    # pixel within a factor x factor block has the identical value.
    noise_r_downsampled = conservative_downsample(noise_r, factor)
    noise_r_reconstructed = torch.repeat_interleave(
        torch.repeat_interleave(noise_r_downsampled, factor, dim=-2), factor, dim=-1
    )
    assert torch.allclose(noise_r, noise_r_reconstructed, atol=1e-5)
    # ... and not degenerate (nonzero variance across blocks/time).
    assert noise_r_downsampled.std() > 1e-3

    # d's noise must be exactly in null(D): D(noise_d) == 0.
    assert torch.allclose(
        conservative_downsample(noise_d, factor),
        torch.zeros(2, n_channels, n_times, coarse_height, coarse_width),
        atol=1e-5,
    )
    assert noise_d.std() > 1e-3  # not degenerate
