import abc
import dataclasses

import torch

from fme.core.rand import log_normal_sample, log_uniform_sample, randn_like


@dataclasses.dataclass
class ConditionedTarget:
    """
    A class to hold the conditioned targets and the loss weighting.

    Attributes:
        latents: The normalized targets with noise added.
        sigma: The noise level.
        weight: The loss weighting.
    """

    latents: torch.Tensor
    sigma: torch.Tensor
    weight: torch.Tensor


class NoiseDistribution(abc.ABC):
    @abc.abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pass


@dataclasses.dataclass
class LogNormalNoiseDistribution(NoiseDistribution):
    p_mean: float
    p_std: float

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return log_normal_sample(
            p_mean=self.p_mean,
            p_std=self.p_std,
            shape=(batch_size, 1, 1, 1),
            dtype=torch.float32,
        ).to(device)


@dataclasses.dataclass
class LogUniformNoiseDistribution(NoiseDistribution):
    p_min: float
    p_max: float

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return log_uniform_sample(
            p_min=self.p_min,
            p_max=self.p_max,
            shape=(batch_size, 1, 1, 1),
            dtype=torch.float32,
        ).to(device)


def brownian_bridge_mixing_matrix(n_timesteps: int) -> torch.Tensor:
    """Temporal mixing matrix for endpoint-pinned, time-correlated noise.

    Returns a ``(T, T)`` matrix ``M`` such that, given white noise
    ``Z ~ N(0, I)`` of shape ``(..., T, ...)`` mixed along the time axis as
    ``E = M @ Z``, the noise ``E`` has a Brownian-bridge *correlation* structure
    across the interior frames and is exactly zero on the two endpoint frames.

    The Brownian-bridge covariance on normalized times ``tau_k = k / (T - 1)``
    (with bridge length 1) is ``k_BB(s, t) = min(s, t) - s * t``. It vanishes at
    both endpoints, matching the endpoint-pinned residual the video model
    diffuses. We normalize it to a *correlation* matrix (unit diagonal on the
    interior) so that, when scaled by the EDM noise level ``sigma``, each frame's
    marginal noise std stays ``sigma`` -- only the cross-time correlation is
    introduced. ``M`` is the Cholesky factor of that correlation matrix placed
    into the interior block; the endpoint rows/cols are all zero.

    Args:
        n_timesteps: Number of frames ``T`` (>= 3: two endpoints + interior).

    Returns:
        A ``(T, T)`` float32 tensor on the CPU.
    """
    if n_timesteps < 3:
        raise ValueError(
            "Brownian-bridge noise needs at least 3 frames (2 endpoints + 1 "
            f"interior), got n_timesteps={n_timesteps}."
        )
    n_interior = n_timesteps - 2
    tau = torch.arange(1, n_timesteps - 1, dtype=torch.float64) / (n_timesteps - 1)
    s = tau.reshape(-1, 1)
    t = tau.reshape(1, -1)
    cov = torch.minimum(s, t) - s * t  # interior Brownian-bridge covariance
    std = cov.diagonal().sqrt()
    corr = cov / torch.outer(std, std)  # normalize to unit diagonal
    chol = torch.linalg.cholesky(corr)  # lower-triangular, corr == chol @ chol.T
    mixing = torch.zeros(n_timesteps, n_timesteps, dtype=torch.float64)
    mixing[1 : 1 + n_interior, 1 : 1 + n_interior] = chol
    return mixing.to(torch.float32)


def condition_with_noise_for_training(
    targets_norm: torch.Tensor,
    noise_distribution: NoiseDistribution,
    sigma_data: float,
    loss_weight_exponent: float = 1.0,
) -> ConditionedTarget:
    """
    Condition the targets with noise for training.

    Args:
        targets_norm: The normalized targets.
        noise_distribution: The noise distribution to use for conditioning.
        sigma_data: The standard deviation of the data,
            used to determine loss weighting.
        loss_weight_exponent: Exponent applied to the base EDM loss weight
            ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2``. The default
            of 1.0 gives the standard EDM weighting (~1/sigma^2 for small
            sigma). Use 0.5 for ~1/sigma weighting (square root of EDM weight).

    Returns:
        The conditioned targets and the loss weighting.
    """
    sigma = noise_distribution.sample(targets_norm.shape[0], targets_norm.device)
    weight = (
        (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    ) ** loss_weight_exponent
    noise = randn_like(targets_norm) * sigma
    latents = targets_norm + noise
    return ConditionedTarget(latents=latents, sigma=sigma, weight=weight)
