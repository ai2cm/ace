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


class NoiseDistribution(abc.ABC):
    @abc.abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pass


@dataclasses.dataclass
class LogNormalNoiseDistribution(NoiseDistribution):
    p_mean: float
    p_std: float

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        rnd = randn([batch_size, 1, 1, 1], device=device)
        # This is taken from EDM's original implementation in EDMLoss:
        # https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L72-L80  # noqa: E501
        return (rnd * self.p_std + self.p_mean).exp()


@dataclasses.dataclass
class LogUniformNoiseDistribution(NoiseDistribution):
    p_min: float
    p_max: float

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        sigma = np.exp(
            np.random.uniform(np.log(self.p_min), np.log(self.p_max), batch_size)
        )
        return torch.tensor(sigma, device=device).reshape(batch_size, 1, 1, 1)


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
