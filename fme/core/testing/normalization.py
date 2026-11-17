import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import xarray as xr

from fme.core.normalizer import (
    GroupedNormalizationConfig,
    NetworkAndLossNormalizationConfig,
    NormalizationConfig,
)


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


def uniform_grouped_normalization(
    names: Iterable[str],
    groups: Mapping[str, tuple[float, float]],
    default_label: str,
    pinned_variables: Sequence[str] = (),
) -> GroupedNormalizationConfig:
    """
    Create a GroupedNormalizationConfig in which each group uses the same mean
    and std for all names.

    Args:
        names: Names each group provides constants for.
        groups: Mapping from dataset label to ``(mean, std)``.
        default_label: Label whose group is used for unlabeled batches.
        pinned_variables: Names which always use the pooled constants.
    """
    names = list(names)
    return GroupedNormalizationConfig(
        groups={
            label: trivial_normalization(names, mean=mean, std=std)
            for label, (mean, std) in groups.items()
        },
        default_label=default_label,
        pinned_variables=list(pinned_variables),
    )


def write_normalization_stats(
    path: pathlib.Path,
    constants: Mapping[str, float | np.ndarray | torch.Tensor],
    coords: Mapping[str, np.ndarray] | None = None,
    dims: Sequence[str] | None = None,
    encoding: Mapping[str, Mapping[str, Any]] | None = None,
) -> pathlib.Path:
    """
    Write scalar or per-grid-cell normalization constants to a netCDF file
    and return its path. Per-grid-cell constants are written on ``dims``,
    which default to the dimensions of ``coords``.
    """
    coords = dict(coords or {})
    grid_dims = tuple(coords) if dims is None else tuple(dims)
    data_vars = {
        name: (grid_dims if np.ndim(value) else (), np.asarray(value))
        for name, value in constants.items()
    }
    xr.Dataset(data_vars, coords=coords).to_netcdf(path, encoding=encoding)
    return path
