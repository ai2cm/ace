from collections.abc import Iterable

from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig


def trivial_normalization(
    names: Iterable[str], mean: float = 0.0, std: float = 1.0
) -> NormalizationConfig:
    """
    Create a NormalizationConfig with the same mean and std for all names.
    """
    return NormalizationConfig(
        means={name: mean for name in names},
        stds={name: std for name in names},
    )


def trivial_network_and_loss_normalization(
    names: Iterable[str], mean: float = 0.0, std: float = 1.0
) -> NetworkAndLossNormalizationConfig:
    """
    Create a NetworkAndLossNormalizationConfig with the same mean and std for
    all names, using the network normalization for the loss.
    """
    return NetworkAndLossNormalizationConfig(
        network=trivial_normalization(names, mean=mean, std=std),
    )
