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
    ) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[batch_size, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.
        """
        if self.uniform is not None:
            return _sample_uniform(self.uniform, n_channels, batch_size, device)
        assert self.per_variable is not None
        return _sample_per_variable(self.per_variable, n_channels, batch_size, device)


def _sample_uniform(
    config: UniformMaskingConfig,
    n_channels: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    min_n = 0 if config.min_vars == "min" else int(config.min_vars)
    max_n = n_channels if config.max_vars == "max" else int(config.max_vars)
    max_n = min(max_n, n_channels)
    mask = torch.ones(batch_size, n_channels, dtype=torch.bool, device=device)
    for i in range(batch_size):
        n_mask = int(torch.randint(min_n, max_n + 1, ()).item())
        if n_mask > 0:
            indices = torch.randperm(n_channels, device=device)[:n_mask]
            mask[i, indices] = False
    return mask


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
