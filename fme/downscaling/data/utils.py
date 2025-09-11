import dataclasses
import random
from collections.abc import Sequence

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.properties import DatasetProperties


@dataclasses.dataclass
class ClosedInterval:
    start: float
    stop: float

    def __post_init__(self):
        assert self.start < self.stop  # Do not allow empty, start = stop

    def __contains__(self, value: float):
        return self.start <= value <= self.stop


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


def adjust_fine_coord_range(
    coord_range: ClosedInterval,
    full_coarse_coord: torch.Tensor,
    full_fine_coord: torch.Tensor,
) -> ClosedInterval:
    """
    Arbitrary min/max bounds in the lat_range and lon_range config args are
    not guaranteed to subselect the fine data such that it exactly matches the
    edges of the subselected coarse data. This function adjusts the coordinate
    range for fine subselection to ensure this in the subselected dataset.
    """
    if full_fine_coord.shape[0] % full_coarse_coord.shape[0] != 0:
        raise ValueError(
            "Full fine lat size must be evenly divisible by coarse lat size."
        )
    downscale_factor = full_fine_coord.shape[0] // full_coarse_coord.shape[0]

    if downscale_factor == 1:
        return coord_range

    # The fine grid that exactly covers the coarse grid should have downscale_factor//2
    # fine points on either side of the min/max coarse coord gridpoints.
    n_half_fine = downscale_factor // 2
    coarse_min = full_coarse_coord[full_coarse_coord >= coord_range.start][0]
    coarse_max = full_coarse_coord[full_coarse_coord <= coord_range.stop][-1]
    fine_min = full_fine_coord[full_fine_coord < coarse_min][-n_half_fine]
    fine_max = full_fine_coord[full_fine_coord > coarse_max][n_half_fine - 1]

    return ClosedInterval(start=fine_min, stop=fine_max)


def paired_shuffle(a: list, b: list) -> tuple[list, list]:
    if len(a) != len(b):
        raise ValueError("Lists in paired shuffle must have the same length.")
    indices = list(range(len(a)))
    random.shuffle(indices)
    return [a[i] for i in indices], [b[i] for i in indices]


def get_offset(random_offset: bool, full_size: int, patch_size: int) -> int:
    if random_offset:
        max_offset = min(patch_size - 1, full_size - patch_size)
        return random.randint(0, max_offset)
    return 0


def scale_tuple(extent: tuple[int, int], scale_factor: int) -> tuple[int, int]:
    return (extent[0] * scale_factor, extent[1] * scale_factor)
