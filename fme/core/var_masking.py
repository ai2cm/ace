import dataclasses
from typing import Literal

import torch

CO2_NAME = "global_mean_co2"


@dataclasses.dataclass
class UniformMaskingConfig:
    """
    Masks a uniformly sampled count of masked variables per sample.

    Parameters:
        max_masked_vars: Maximum number of masked variables. "max" resolves to
            the total number of input channels. The number of masked variables
            is sampled uniformly from ``[0, max_masked_vars]`` per sample, so a
            sample may have no masked variables.
        co2_rate: Optional independent masking rate for the ``global_mean_co2``
            channel. When set, that channel is masked at this rate regardless of
            the uniform base mask. When ``None``, CO2 is masked together with the
            other channels. Requires ``global_mean_co2`` to be an input channel.
    """

    kind: Literal["uniform"] = "uniform"
    max_masked_vars: int | str = "max"
    co2_rate: float | None = None

    def __post_init__(self):
        if self.max_masked_vars != "max" and (
            not isinstance(self.max_masked_vars, int) or self.max_masked_vars < 0
        ):
            raise ValueError(
                "max_masked_vars must be a non-negative int or 'max', got "
                f"{self.max_masked_vars!r}"
            )
        _validate_co2_rate(self.co2_rate)

    def validate_names(self, channel_names: list[str]) -> None:
        _validate_co2_in_names(self.co2_rate, channel_names)

    def sample_mask(
        self,
        n_channels: int,
        batch_size: int,
        device: torch.device,
        channel_names: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[batch_size, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.
        The number of masked variables per sample is drawn uniformly from
        ``[0, max_masked_vars]``.

        When ``co2_rate`` is set, ``channel_names`` must be provided to locate
        the CO2 channel; its column is overwritten with an independent sample.
        """
        mask = _sample_uniform(self, n_channels, batch_size, device)
        return _apply_co2_override(mask, self.co2_rate, channel_names, device)


@dataclasses.dataclass
class PerVariableMaskingConfig:
    """
    Masks each variable independently with a fixed Bernoulli rate.

    Parameters:
        rate: Probability that any single variable is masked for a given sample.
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
        channel_names: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[batch_size, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.

        When ``co2_rate`` is set, ``channel_names`` must be provided to locate
        the CO2 channel; its column is overwritten with an independent sample.
        """
        mask = _sample_per_variable(self, n_channels, batch_size, device)
        return _apply_co2_override(mask, self.co2_rate, channel_names, device)

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
