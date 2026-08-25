import pytest
import torch

from fme.downscaling.noise import (
    LogNormalNoiseDistribution,
    LogUniformNoiseDistribution,
    brownian_bridge_mixing_matrix,
    uniform_frame_times,
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


def _bridge_corr(tau):
    """Reference unit-diagonal Brownian-bridge correlation on interior times."""
    s, t = tau.reshape(-1, 1), tau.reshape(1, -1)
    bb = torch.minimum(s, t) - s * t
    std = bb.diagonal().sqrt()
    return (bb / torch.outer(std, std)).to(torch.float32)


@pytest.mark.parametrize("n_timesteps", [3, 5, 9])
def test_brownian_bridge_mixing_matrix(n_timesteps):
    tau = uniform_frame_times(n_timesteps)
    mixing = brownian_bridge_mixing_matrix(tau)
    assert mixing.shape == (n_timesteps, n_timesteps)
    assert mixing.dtype == torch.float32

    cov = mixing @ mixing.T  # covariance of M @ Z for white Z

    # Endpoints carry no noise.
    assert torch.allclose(cov[0], torch.zeros(n_timesteps))
    assert torch.allclose(cov[-1], torch.zeros(n_timesteps))
    assert torch.allclose(cov[:, 0], torch.zeros(n_timesteps))
    assert torch.allclose(cov[:, -1], torch.zeros(n_timesteps))

    # Interior block matches the unit-diagonal (correlation) Brownian bridge.
    expected_corr = _bridge_corr(tau[1:-1])
    interior = cov[1:-1, 1:-1]
    assert torch.allclose(interior.diagonal(), torch.ones(n_timesteps - 2), atol=1e-5)
    assert torch.allclose(interior, expected_corr, atol=1e-5)


def test_brownian_bridge_subset_matches_full_grid_marginal():
    """A subset's mixing matrix reproduces the full grid's correlations exactly:
    generating a subset of frames draws from the full-window bridge's marginal."""
    full_tau = uniform_frame_times(9)
    full_cov = brownian_bridge_mixing_matrix(full_tau)
    full_cov = full_cov @ full_cov.T

    # keep endpoints (0, 8) and a non-uniform interior subset (3, 6)
    idx = torch.tensor([0, 3, 6, 8])
    subset = brownian_bridge_mixing_matrix(full_tau[idx])
    subset_cov = subset @ subset.T

    assert torch.allclose(subset_cov, full_cov[idx][:, idx], atol=1e-5)


def test_brownian_bridge_mixing_matrix_requires_interior():
    with pytest.raises(ValueError, match="at least 3 frames"):
        brownian_bridge_mixing_matrix(uniform_frame_times(2))
