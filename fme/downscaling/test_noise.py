import pytest
from fme.downscaling.noise import LogNormalNoiseDistribution, LogUniformNoiseDistribution


@pytest.mark.parametrize(
    "noise_distribution", 
    [
        LogNormalNoiseDistribution(p_mean=0.0, p_std=1.0),
        LogUniformNoiseDistribution(p_min=0.01, p_max=100)
    ]
)
def test_noise_distribution(noise_distribution):
    batch_size = 10
    assert noise_distribution.sample(batch_size=batch_size, device="cpu").shape == (batch_size, 1, 1, 1)
