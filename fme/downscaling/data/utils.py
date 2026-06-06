import dataclasses
import random
from collections.abc import Sequence

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.properties import DatasetProperties
from fme.core.device import get_device
from fme.core.metrics import spherical_area_weights


def null_generator(num: int):
    # Used to fill in null topography field when patch generator is used.
    for _ in range(num):
        yield None


@dataclasses.dataclass
class ClosedInterval:
    """
    Defines a closed interval [start, stop] and provides utility methods for working
    with coordinate tensors. The interval includes both the start and stop values
    and stop must be greater than start.

    Parameters:
        start: The minimum value of the interval (inclusive).
        stop: The maximum value of the interval (inclusive).
    """

    start: float
    stop: float

    def __post_init__(self):
        assert self.start < self.stop  # Do not allow empty, start = stop

    def __contains__(self, value: float):
        return self.start <= value <= self.stop

    def slice_from(self, coords: torch.Tensor) -> slice:
        """
        Return a slice that selects all elements of `coords` within this
        specified interval. This assumes `coords` is monotonically increasing.

        Args:
            coords: A 1-D tensor of coordinate values. Must be monotonically
                increasing. Values must be in the same units as `self.start`
                and `self.stop`.

        Returns:
            A `slice` object suitable for indexing `coords` or any tensor whose
            corresponding dimension aligns with `coords`.

        Raises:
            ValueError: If no element of `coords` falls within this interval.
        """
        mask = (coords >= self.start) & (coords <= self.stop)
        if not mask.any():
            raise ValueError(
                f"Requested interval range {self} does not overlap with coordinate"
                f" range [{coords.min().item()}, {coords.max().item()}]"
            )
        indices = mask.nonzero(as_tuple=True)[0]
        return slice(indices[0].item(), indices[-1].item() + 1)

    def subset_of(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Return a subset of `coords` that falls within this specified interval.
        This assumes `coords` is monotonically increasing.
        """
        return coords[self.slice_from(coords)]

    @property
    def finite_values(self) -> tuple[float, float]:
        """
        Return longitude constrained values for this interval,
        handling infinite endpoints as 0 and 360, respectively.
        """
        start = self.start if self.start != -float("inf") else 0.0
        stop = self.stop if self.stop != float("inf") else 360.0
        return start, stop


def _validate_rollable_lon(lon_coords: torch.Tensor) -> None:
    """Raise if lon_coords cannot be rolled across the prime meridian.

    Rolling adds 360 to the coordinates that wrap around the end of the array,
    which yields a monotonic, contiguous result only if the grid extends the whole
    globe: it must be uniformly spaced and the gap from the last point back to
    the first must equal that spacing (i.e. the grid reaches the 360° wrap
    point). This is the precondition the wrap arithmetic assumes.

    The wrap-gap test ``lon[0] + 360 - lon[-1] == spacing`` is convention
    independent: for any global grid ``lon[-1] - lon[0] == 360 - spacing``,
    whether expressed as [0, 360), [-180, 180), or any offset.
    """
    if lon_coords.numel() < 2:
        raise ValueError(
            "Cannot roll a longitude grid with fewer than 2 points across the "
            "prime meridian."
        )

    lon = lon_coords.detach().double()
    diffs = lon[1:] - lon[:-1]
    spacing = float(diffs.mean())
    tol = abs(spacing) * 1e-3
    if float((diffs - spacing).abs().max()) > tol:
        raise ValueError(
            "Longitude coordinates are not uniformly spaced; cannot roll across "
            "the prime meridian."
        )
    wrap_gap = float(lon[0] + 360.0 - lon[-1])
    if abs(wrap_gap - spacing) > tol:
        raise ValueError(
            "Longitude coordinates do not span the full globe; cannot roll across "
            f"the prime meridian (wrap gap {wrap_gap:.4f}°, grid spacing "
            f"{spacing:.4f}°)."
        )


def coords_require_lon_roll(coarse_lon: torch.Tensor) -> bool:
    """
    Return True if coarse_lon spans the prime meridian and needs rolling.

    Triggers when any longitude is negative (domain shifted west of 0°) or
    exceeds 360° (domain shifted east of 360°).

    Args:
        coarse_lon: 1-D tensor of longitudes (e.g. 0–360°).
    """
    return bool(coarse_lon.min() < 0.0 or coarse_lon.max() > 360.0)


def find_roll_anchor(lon_coords: torch.Tensor, anchor: float) -> int:
    """
    Leftward roll that places the first coord >= (anchor mod 360) at index 0.

    The roll amount is the number of coordinates strictly below the anchor
    longitude, which is taken mod 360 so that any caller convention (negative
    or > 360) is accepted. Used directly when the target coordinates are known
    such as the start of a longitude interval.

    The result is reduced modulo the grid length: rolling is cyclic, so when
    every coordinate falls below the anchor the roll is a full rotation, which
    is a no-op and canonicalizes to 0.

    Assumes lon_coords are monotonically increasing and cyclic.

    Args:
        lon_coords: 1-D tensor of monotonically increasing longitudes (e.g. 0–360°).
        anchor: The longitude to anchor at index 0 after rolling.
    """
    n = len(lon_coords)
    below = int((lon_coords < anchor % 360.0).sum().item())
    return below % n


def find_roll_anchor_from_interval(
    lon_coords: torch.Tensor, lon_interval: ClosedInterval
) -> int:
    """
    Find the roll anchor index for a longitude interval. See
    :func:`find_roll_anchor` for details.

    Returns 0 unless the interval crosses the prime meridian (its effective
    start is below 0° or its effective stop is above 360°). An in-range interval
    needs no roll, so this mirrors the :func:`coords_require_lon_roll` gate used
    on the model side and avoids rolling a non-global grid, which cannot be
    rolled across the seam.

    Args:
        lon_coords: 1-D tensor of monotonically increasing longitudes (e.g. 0–360°).
        lon_interval: The desired longitude interval.
    """
    lon_start, lon_stop = lon_interval.finite_values
    if lon_start >= 0.0 and lon_stop <= 360.0:
        return 0
    return find_roll_anchor(lon_coords, lon_start)


def roll_lon_coords(
    lon_coords: torch.Tensor, roll_amount: int, lon_start: float
) -> torch.Tensor:
    """
    Re-express a global longitude grid as a monotonically increasing sequence
    that starts near ``lon_start``, matching the same leftward roll applied to
    the data by :func:`roll_lon_data` (so coords and data stay aligned).

    Used so that we can use the user-supplied interval for slicing of the
    rolled coordinates, which is more intuitive if the rolled coordinates are in
    the same convention (e.g. -90 to 270 for an interval starting at -90)
    rather than the original (e.g. 0 to 360).

    Worked example -- 1 degree grid ``[0.5, 1.5, ..., 359.5]`` with
    ``roll_amount=270`` and ``lon_start=-90``:

    1. roll left by 270 -> ``[270.5, ..., 359.5, 0.5, ..., 269.5]`` (drops at seam)
    2. +360 on wrapped tail -> ``[270.5, ..., 359.5, 360.5, ..., 629.5]`` (monotonic)
    3. shift by -360 period -> ``[-89.5, ..., -0.5, 0.5, ..., 269.5]`` (-90 convention)

    Args:
        lon_coords: 1-D tensor of monotonically increasing longitudes.
        roll_amount: Leftward roll from :func:`find_roll_anchor` -- the number
            of coordinates below the anchor, which wrap around to the tail.
        lon_start: Target first longitude; selects which 360 degree window the
            result lands in (e.g. negative for a domain expressed west of 0).

    Returns:
        A new tensor of the same shape, monotonically increasing, with
        ``result[0] ≈ lon_start``.
    """
    if roll_amount == 0:
        return lon_coords
    _validate_rollable_lon(lon_coords)
    n = len(lon_coords)
    rolled = torch.roll(lon_coords, -roll_amount).clone()
    rolled[n - roll_amount :] += 360.0
    period_offset = lon_start - (lon_start % 360.0)
    return rolled + period_offset


def roll_data_lon_dim(
    tensor: torch.Tensor, roll_amount: int, lon_dim: int = -1
) -> torch.Tensor:
    """Roll a data tensor along its longitude dimension by roll_amount positions."""
    if roll_amount == 0:
        return tensor
    return torch.roll(tensor, -roll_amount, dims=lon_dim)


def roll_latlon_coords(
    coords: LatLonCoordinates, roll_amount: int, lon_start: float
) -> LatLonCoordinates:
    """
    Return a new LatLonCoordinates with lon rolled by roll_amount.

    Currently expressed as a function and not a method until roll is
    used outside of downscaling.
    """
    return LatLonCoordinates(
        lat=coords.lat,
        lon=roll_lon_coords(coords.lon, roll_amount, lon_start),
    )


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
    downscale_factor: int | None = None,
) -> ClosedInterval:
    """
    Arbitrary min/max bounds in the lat_range and lon_range config args are
    not guaranteed to subselect the fine data such that it exactly matches the
    edges of the subselected coarse data. This function adjusts the coordinate
    range for fine subselection to ensure this in the subselected dataset.

    If downscale factor is not provided, it is assumed that the coarse and fine
    coordinate tensors correspond to the same region bounds.

    Raises:
        ValueError: If coord_range is too close to the boundary of full_fine_coord
            such that fewer than downscale_factor // 2 fine points exist beyond the
            outermost selected coarse point on either side. For global latitude grids,
            this is avoided by restricting coord_range to within ±88° (i.e. away from
            the poles).
    """
    if downscale_factor is None:
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

    n_fine_below = int((full_fine_coord < coarse_min).sum())
    n_fine_above = int((full_fine_coord > coarse_max).sum())
    if n_fine_below < n_half_fine or n_fine_above < n_half_fine:
        raise ValueError(
            f"coord_range {coord_range} is too close to the boundary of "
            f"full_fine_coord [{full_fine_coord.min():.2f}, "
            f"{full_fine_coord.max():.2f}]. Need at least {n_half_fine} fine "
            f"point(s) beyond each coarse boundary; got {n_fine_below} below "
            f"and {n_fine_above} above. Restrict the coordinate range away from "
            f"the domain edges."
        )

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


@dataclasses.dataclass
class BatchedLatLonCoordinates:
    """
    Container for batched latitude and longitude coordinates.
    Expects leading batch dimensions (that are the same) for
    lat and lon coordinates.
    """

    lat: torch.Tensor
    lon: torch.Tensor
    dims: list[str] = dataclasses.field(default_factory=lambda: ["batch", "lat", "lon"])

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.lat.dim() != 2 or self.lon.dim() != 2:
            raise ValueError(
                f"Expected 2D lat and lon coordinates, got shapes {self.lat.shape} "
                f"and {self.lon.shape}."
            )

        if self.lat.shape[0] != self.lon.shape[0]:
            raise ValueError(
                f"Latitude batch dimension {self.lat.shape[0]} does not match "
                f"longitude batch dimension {self.lon.shape[0]}"
            )

    @classmethod
    def from_sequence(
        cls,
        items: Sequence[LatLonCoordinates],
    ) -> "BatchedLatLonCoordinates":
        lats = torch.utils.data.default_collate([i.lat for i in items])
        lons = torch.utils.data.default_collate([i.lon for i in items])
        return BatchedLatLonCoordinates(lats, lons)

    @property
    def area_weights(self) -> torch.Tensor:
        return spherical_area_weights(self.lat, self.lon.shape[-1])

    def to_device(self) -> "BatchedLatLonCoordinates":
        device = get_device()
        return BatchedLatLonCoordinates(self.lat.to(device), self.lon.to(device))

    def __getitem__(self, k):
        lats = self.lat[k]
        lons = self.lon[k]

        return LatLonCoordinates(lat=lats, lon=lons)

    def __eq__(self, other):
        return torch.equal(self.lat, other.lat) and torch.equal(self.lon, other.lon)

    def __len__(self):
        return self.lat.shape[0]
