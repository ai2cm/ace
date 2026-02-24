import dataclasses

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


def condition_with_noise_for_training(
    targets_norm: torch.Tensor,
    p_std: float,
    p_mean: float,
    sigma_data: float,
) -> ConditionedTarget:
    """
    Condition the targets with noise for training.

    Args:
        targets_norm: The normalized targets.
        p_std: The standard deviation of the noise distribution used during training.
        p_mean: The mean of the noise distribution used during training.
        sigma_data: The standard deviation of the data,
            used to determine loss weighting.

    Returns:
        The conditioned targets and the loss weighting.
    """
    rnd_normal = randn([targets_norm.shape[0], 1, 1, 1], device=targets_norm.device)
    # This is taken from EDM's original implementation in EDMLoss:
    # https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L72-L80  # noqa: E501
    sigma = (rnd_normal * p_std + p_mean).exp()
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    noise = randn_like(targets_norm) * sigma
    latents = targets_norm + noise
    return ConditionedTarget(latents=latents, sigma=sigma, weight=weight)
