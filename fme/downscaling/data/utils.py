from collections.abc import Sequence

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.properties import DatasetProperties


def scale_slice(slice_: slice, scale: int) -> slice:
    if slice_ == slice(None):
        return slice_
    start = slice_.start * scale if slice_.start is not None else None
    stop = slice_.stop * scale if slice_.stop is not None else None
    return slice(start, stop)


def expand_and_fold_tensor(
    tensor: torch.Tensor, num_samples: int, sample_dim: int
) -> torch.Tensor:
    static_shape = tensor.shape[sample_dim:]
    expanded_shape = [-1 for _ in tensor.shape]
    expanded_shape.insert(sample_dim, num_samples)
    expanded = tensor.unsqueeze(sample_dim).expand(*expanded_shape)
    return expanded.reshape(-1, *static_shape)


def check_leading_dim(
    name: str, current_leading: Sequence[int], expected_leading: Sequence[int]
):
    if current_leading != expected_leading:
        raise ValueError(
            f"Expected leading dimension of {name} shape {expected_leading}, got "
            f"{current_leading}"
        )


def get_latlon_coords_from_properties(
    properties: DatasetProperties,
) -> LatLonCoordinates:
    if not isinstance(properties.horizontal_coordinates, LatLonCoordinates):
        raise NotImplementedError(
            "Horizontal coordinates must be of type LatLonCoordinates"
        )
    return properties.horizontal_coordinates
