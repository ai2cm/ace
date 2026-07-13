import pytest
import torch

from fme.downscaling.noise import (
    LogNormalNoiseDistribution,
    LogUniformNoiseDistribution,
    condition_with_noise_for_training,
)


@pytest.mark.parametrize(
    "noise_distribution",
    [
        LogNormalNoiseDistribution(p_mean=0.0, p_std=1.0),
        LogUniformNoiseDistribution(p_min=0.01, p_max=100),
    ],
)
def test_noise_distribution(noise_distribution):
    batch_size = 10
    noise = noise_distribution.sample(batch_size=batch_size, device="cpu")
    assert noise.shape == (batch_size, 1, 1, 1)
    assert noise.dtype == torch.float32


def test_condition_with_noise_scalar_exponent():
    batch_size, channels = 4, 3
    targets_norm = torch.zeros(batch_size, channels, 2, 2)
    result = condition_with_noise_for_training(
        targets_norm,
        LogNormalNoiseDistribution(p_mean=0.0, p_std=1.0),
        sigma_data=0.5,
        loss_weight_exponent=1.0,
    )
    assert result.sigma.shape == (batch_size, 1, 1, 1)
    assert result.weight.shape == (batch_size, 1, 1, 1)


def test_condition_with_noise_per_channel_exponent():
    batch_size, channels = 4, 3
    targets_norm = torch.zeros(batch_size, channels, 2, 2)
    sigma_data = 0.5
    exponents = torch.tensor([1.0, 0.5, 0.75]).reshape(1, channels, 1, 1)
    result = condition_with_noise_for_training(
        targets_norm,
        LogNormalNoiseDistribution(p_mean=0.0, p_std=1.0),
        sigma_data=sigma_data,
        loss_weight_exponent=exponents,
    )
    assert result.weight.shape == (batch_size, channels, 1, 1)
    base = (result.sigma**2 + sigma_data**2) / (result.sigma * sigma_data) ** 2
    for i, exponent in enumerate([1.0, 0.5, 0.75]):
        expected = base[:, 0, :, :] ** exponent
        assert torch.allclose(result.weight[:, i, :, :], expected)
