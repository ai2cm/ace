import dataclasses
from typing import Literal

import torch


@dataclasses.dataclass
class UniformMaskingConfig:
    """
    Masks a uniformly sampled count of masked variables per sample.

    Parameters:
        max_masked_vars: Maximum number of masked variables. "max" resolves to
            the total number of input channels. The number of masked variables
            is sampled uniformly from ``[0, max_masked_vars]`` per sample, so a
            sample may have no masked variables.
    """

    kind: Literal["uniform"] = "uniform"
    max_masked_vars: int | str = "max"

    def __post_init__(self):
        if self.max_masked_vars != "max" and (
            not isinstance(self.max_masked_vars, int) or self.max_masked_vars < 0
        ):
            raise ValueError(
                "max_masked_vars must be a non-negative int or 'max', got "
                f"{self.max_masked_vars!r}"
            )

    def sample_mask(
        self,
        n_channels: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[batch_size, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.
        The number of masked variables per sample is drawn uniformly from
        ``[0, max_masked_vars]``.
        """
        return _sample_uniform(self, n_channels, batch_size, device)


@dataclasses.dataclass
class PerVariableMaskingConfig:
    """
    Masks each variable independently with a fixed Bernoulli rate.

    Parameters:
        rate: Probability that any single variable is masked for a given sample.
    """

    kind: Literal["per_variable"] = "per_variable"
    rate: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {self.rate}")

    def sample_mask(
        self,
        n_channels: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[batch_size, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.
        """
        return _sample_per_variable(self, n_channels, batch_size, device)


VariableMaskingConfig = UniformMaskingConfig | PerVariableMaskingConfig


def _sample_uniform(
    config: UniformMaskingConfig,
    n_channels: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    max_n = (
        n_channels if config.max_masked_vars == "max" else int(config.max_masked_vars)
    )
    max_n = min(max_n, n_channels)
    # Draw a random count of masked vars per sample in [0, max_n].
    n_masks = torch.randint(0, max_n + 1, (batch_size,), device=device)
    noise = torch.rand(batch_size, n_channels, device=device)
    # rank[i, j]: ordinal rank of channel j in sample i.
    rank = noise.argsort(dim=1).argsort(dim=1)
    return rank >= n_masks.unsqueeze(1)


def _sample_per_variable(
    config: PerVariableMaskingConfig,
    n_channels: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    keep_prob = 1.0 - config.rate
    return torch.bernoulli(
        torch.full((batch_size, n_channels), keep_prob, device=device)
    ).bool()
