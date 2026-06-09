import dataclasses

import torch


@dataclasses.dataclass
class UniformMaskingConfig:
    """
    Masks a uniformly sampled count of channels per sample.

    Parameters:
        min_vars: Minimum number of channels to mask. "min" resolves to 0.
        max_vars: Maximum number of channels to mask. "max" resolves to the
            total number of input channels.
    """

    min_vars: int | str = "min"
    max_vars: int | str = "max"

    def __post_init__(self):
        if self.min_vars != "min" and (
            not isinstance(self.min_vars, int) or self.min_vars < 0
        ):
            raise ValueError(
                f"min_vars must be a non-negative int or 'min', got {self.min_vars!r}"
            )
        if self.max_vars != "max" and (
            not isinstance(self.max_vars, int) or self.max_vars < 0
        ):
            raise ValueError(
                f"max_vars must be a non-negative int or 'max', got {self.max_vars!r}"
            )
        if (
            isinstance(self.min_vars, int)
            and isinstance(self.max_vars, int)
            and self.min_vars > self.max_vars
        ):
            raise ValueError(
                f"min_vars ({self.min_vars}) must be <= max_vars ({self.max_vars})"
            )


@dataclasses.dataclass
class PerVariableMaskingConfig:
    """
    Masks each channel independently with a fixed Bernoulli rate.

    Parameters:
        rate: Probability that any single channel is masked for a given sample.
    """

    rate: float

    def __post_init__(self):
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {self.rate}")


@dataclasses.dataclass
class VariableMaskingConfig:
    """
    Training-time input channel dropout config. Exactly one of ``uniform`` or
    ``per_variable`` must be set.

    Parameters:
        uniform: Uniform count-based masking config.
        per_variable: Independent per-channel Bernoulli masking config.
    """

    uniform: UniformMaskingConfig | None = None
    per_variable: PerVariableMaskingConfig | None = None

    def __post_init__(self):
        if (self.uniform is None) == (self.per_variable is None):
            raise ValueError(
                "Exactly one of 'uniform' or 'per_variable' must be provided in "
                "VariableMaskingConfig."
            )

    def sample_mask(
        self,
        n_channels: int,
        batch_size: int,
        device: torch.device,
        n_ensemble: int = 1,
    ) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[batch_size, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.

        When ``n_ensemble > 1``, masks are sampled for ``batch_size //
        n_ensemble`` base samples and then repeated so that every ensemble
        member belonging to the same base sample receives the same mask.
        """
        base_batch_size = _get_base_batch_size(batch_size, n_ensemble)
        if self.uniform is not None:
            mask = _sample_uniform(self.uniform, n_channels, base_batch_size, device)
        else:
            assert self.per_variable is not None
            mask = _sample_per_variable(
                self.per_variable, n_channels, base_batch_size, device
            )
        return _repeat_ensemble_mask(mask, n_ensemble)


def _get_base_batch_size(batch_size: int, n_ensemble: int) -> int:
    if batch_size % n_ensemble != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by n_ensemble ({n_ensemble})"
        )
    return batch_size // n_ensemble


def _repeat_ensemble_mask(mask: torch.Tensor, n_ensemble: int) -> torch.Tensor:
    """Repeat each row of mask n_ensemble times (interleaved)."""
    return torch.repeat_interleave(mask, n_ensemble, dim=0)


def _sample_uniform(
    config: UniformMaskingConfig,
    n_channels: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    min_n = 0 if config.min_vars == "min" else int(config.min_vars)
    max_n = n_channels if config.max_vars == "max" else int(config.max_vars)
    max_n = min(max_n, n_channels)
    # For each sample, draw a random count then assign random ranks to channels.
    # Channels whose rank < n_masks[i] are masked (False); the rest are kept (True).
    n_masks = torch.randint(min_n, max_n + 1, (batch_size,), device=device)
    noise = torch.rand(batch_size, n_channels, device=device)
    # rank[i, j] = ordinal rank of channel j within sample i (0 = first masked)
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
