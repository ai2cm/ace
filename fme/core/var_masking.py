import dataclasses
from typing import Literal

import torch

CO2_NAME = "global_mean_co2"


@dataclasses.dataclass
class UniformMaskingConfig:
    """
    Masks a uniformly sampled count of channels per sample.

    Parameters:
        min_vars: Minimum number of channels to mask. Defaults to 1.
        max_vars: Maximum number of channels to mask. "max" resolves to the
            total number of input channels.
        co2_rate: Optional independent masking rate for the ``global_mean_co2``
            channel. When set, that channel is masked at this rate regardless of
            the uniform base mask. When ``None``, CO2 is masked together with the
            other channels (the base sampler still covers the CO2 slot, a minor
            count quirk for the uniform sampler). Requires ``global_mean_co2`` to
            be an input channel.
    """

    kind: Literal["uniform"] = "uniform"
    min_vars: int = 1
    max_vars: int | str = "max"
    co2_rate: float | None = None

    def __post_init__(self):
        if not isinstance(self.min_vars, int) or self.min_vars < 0:
            raise ValueError(
                f"min_vars must be a non-negative int, got {self.min_vars!r}"
            )
        if self.max_vars != "max" and (
            not isinstance(self.max_vars, int) or self.max_vars < 0
        ):
            raise ValueError(
                f"max_vars must be a non-negative int or 'max', got {self.max_vars!r}"
            )
        if isinstance(self.max_vars, int) and self.min_vars > self.max_vars:
            raise ValueError(
                f"min_vars ({self.min_vars}) must be <= max_vars ({self.max_vars})"
            )
        _validate_co2_rate(self.co2_rate)

    def sample_mask(
        self,
        n_channels: int,
        batch_size: int,
        device: torch.device,
        n_ensemble: int = 1,
        channel_names: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[batch_size, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.

        When ``n_ensemble > 1``, masks are sampled for ``batch_size //
        n_ensemble`` base samples and then repeated so that every ensemble
        member belonging to the same base sample receives the same mask.

        When ``co2_rate`` is set, ``channel_names`` must be provided to locate
        the CO2 channel; its column is overwritten with an independent sample.
        """
        base_batch_size = _get_base_batch_size(batch_size, n_ensemble)
        mask = _sample_uniform(self, n_channels, base_batch_size, device)
        mask = _apply_co2_override(mask, self.co2_rate, channel_names, device)
        return _repeat_ensemble_mask(mask, n_ensemble)

    def validate_names(self, channel_names: list[str]) -> None:
        _validate_co2_in_names(self.co2_rate, channel_names)


@dataclasses.dataclass
class PerVariableMaskingConfig:
    """
    Masks each channel independently with a fixed Bernoulli rate.

    Parameters:
        rate: Probability that any single channel is masked for a given sample.
        co2_rate: Optional independent masking rate for the ``global_mean_co2``
            channel. When set, that channel is masked at this rate regardless of
            ``rate``. When ``None``, CO2 is masked at ``rate`` like every other
            channel. Requires ``global_mean_co2`` to be an input channel.
    """

    kind: Literal["per_variable"] = "per_variable"
    rate: float = 0.0
    co2_rate: float | None = None

    def __post_init__(self):
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {self.rate}")
        _validate_co2_rate(self.co2_rate)

    def sample_mask(
        self,
        n_channels: int,
        batch_size: int,
        device: torch.device,
        n_ensemble: int = 1,
        channel_names: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[batch_size, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.

        When ``n_ensemble > 1``, masks are sampled for ``batch_size //
        n_ensemble`` base samples and then repeated so that every ensemble
        member belonging to the same base sample receives the same mask.

        When ``co2_rate`` is set, ``channel_names`` must be provided to locate
        the CO2 channel; its column is overwritten with an independent sample.
        """
        base_batch_size = _get_base_batch_size(batch_size, n_ensemble)
        mask = _sample_per_variable(self, n_channels, base_batch_size, device)
        mask = _apply_co2_override(mask, self.co2_rate, channel_names, device)
        return _repeat_ensemble_mask(mask, n_ensemble)

    def validate_names(self, channel_names: list[str]) -> None:
        _validate_co2_in_names(self.co2_rate, channel_names)


VariableMaskingConfig = UniformMaskingConfig | PerVariableMaskingConfig


def _validate_co2_rate(co2_rate: float | None) -> None:
    if co2_rate is not None and not 0.0 <= co2_rate <= 1.0:
        raise ValueError(f"co2_rate must be in [0, 1], got {co2_rate}")


def _validate_co2_in_names(co2_rate: float | None, channel_names: list[str]) -> None:
    if co2_rate is not None and CO2_NAME not in channel_names:
        raise ValueError(
            f"co2_rate is set but {CO2_NAME!r} is not an input channel: "
            f"{channel_names}"
        )


def _apply_co2_override(
    mask: torch.Tensor,
    co2_rate: float | None,
    channel_names: list[str] | None,
    device: torch.device,
) -> torch.Tensor:
    """Overwrite the CO2 column of ``mask`` with an independent keep sample.

    No-op when ``co2_rate`` is None. Raises if ``co2_rate`` is set but the CO2
    channel cannot be located in ``channel_names``.
    """
    if co2_rate is None:
        return mask
    if channel_names is None or CO2_NAME not in channel_names:
        raise ValueError(
            f"co2_rate is set but {CO2_NAME!r} is not an input channel: "
            f"{channel_names}"
        )
    idx = channel_names.index(CO2_NAME)
    keep_prob = 1.0 - co2_rate
    mask[:, idx] = torch.bernoulli(
        torch.full((mask.shape[0],), keep_prob, device=device)
    ).bool()
    return mask


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
    min_n = config.min_vars
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
