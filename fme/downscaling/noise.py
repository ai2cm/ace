import abc
import dataclasses

import numpy as np
import torch

from fme.core.rand import randn, randn_like


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


@dataclasses.dataclass
class NoiseDistribution(abc.ABC):
    clamp_min: float = dataclasses.field(default=0.0, kw_only=True)
    clamp_max: float = dataclasses.field(default=float("inf"), kw_only=True)

    @abc.abstractmethod
    def _sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pass

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.clamp(
            self._sample(batch_size, device), self.clamp_min, self.clamp_max
        )


@dataclasses.dataclass
class LogNormalNoiseDistribution(NoiseDistribution):
    p_mean: float
    p_std: float

    def _sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        rnd = randn([batch_size, 1, 1, 1], device=device)
        # This is taken from EDM's original implementation in EDMLoss:
        # https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L72-L80  # noqa: E501
        return (rnd * self.p_std + self.p_mean).exp()


@dataclasses.dataclass
class LogUniformNoiseDistribution(NoiseDistribution):
    p_min: float
    p_max: float

    def _sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        sigma = np.exp(
            np.random.uniform(np.log(self.p_min), np.log(self.p_max), batch_size)
        )
        return torch.tensor(sigma, device=device).reshape(batch_size, 1, 1, 1)


@dataclasses.dataclass
class SkewedLogNormalNoiseDistribution(NoiseDistribution):
    loc: float = -1.2  # Location (xi) in log-space
    scale: float = 1.5  # Scale (omega) in log-space
    alpha: float = 3.0  # Skewness; > 0 favors higher noise (better tails)
    p_min: float = 0.01  # Recommended floor for 512x512
    p_max: float = 80.0  # Typical EDM max

    def _sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # 1. Sample two standard normals for the transformation method
        z0 = np.random.randn(batch_size)
        z1 = np.random.randn(batch_size)

        # 2. Skew-Normal transformation in log-space
        delta = self.alpha / np.sqrt(1 + self.alpha**2)
        # y follows SkewNormal(loc, scale, alpha)
        y = self.loc + self.scale * (delta * np.abs(z0) + np.sqrt(1 - delta**2) * z1)

        # 3. Convert to sigma and enforce bounds
        sigma = np.exp(y)
        sigma = np.clip(sigma, self.p_min, self.p_max)

        return torch.tensor(sigma, device=device, dtype=torch.float32).reshape(
            batch_size, 1, 1, 1
        )


def condition_with_noise_for_training(
    targets_norm: torch.Tensor,
    noise_distribution: NoiseDistribution,
    sigma_data: float,
) -> ConditionedTarget:
    """
    Condition the targets with noise for training.

    Args:
        targets_norm: The normalized targets.
        noise_distribution: The noise distribution to use for conditioning.
        sigma_data: The standard deviation of the data,
            used to determine loss weighting.

    Returns:
        The conditioned targets and the loss weighting.
    """
    sigma = noise_distribution.sample(targets_norm.shape[0], targets_norm.device)
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    noise = randn_like(targets_norm) * sigma
    latents = targets_norm + noise
    return ConditionedTarget(latents=latents, sigma=sigma, weight=weight)
