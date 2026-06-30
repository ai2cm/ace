import datetime

import cftime
import pytest
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import NullOptimization
from fme.core.rand import set_seed
from fme.downscaling.data.datasets import (
    PairedVideoBatchData,
    VideoBatchData,
    VideoBatchItem,
)
from fme.downscaling.data.time_encoding import compute_calendar_features
from fme.downscaling.noise import LogNormalNoiseDistribution, LogUniformNoiseDistribution
from fme.downscaling.video_models import VideoDiffusionModelConfig

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


def _model(n_times, temporal_noise_correlation="independent"):
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
    )
    return config.build()


def test_train_on_batch_runs_and_backprops():
    n_times, height, width = 5, 8, 8
    model = _model(n_times)
    batch = _paired_batch(batch_size=2, n_times=n_times, height=height, width=width)

    outputs = model.train_on_batch(batch, NullOptimization())
    assert torch.isfinite(outputs.loss)
    assert outputs.loss.requires_grad
    # gradients flow into the network
    outputs.loss.backward()
    grads = [
        p.grad for p in model.module.parameters() if p.grad is not None
    ]
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
        bare.sigma_data.flatten(), torch.tensor([0.2, 1.0], device=bare.sigma_data.device)
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
