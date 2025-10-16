import dataclasses
from collections.abc import Generator

import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.downscaling.data.patching import Patch
from fme.downscaling.data.utils import ClosedInterval


def _range_to_slice(coords: torch.Tensor, range: ClosedInterval) -> slice:
    mask = (coords >= range.start) & (coords <= range.stop)
    indices = mask.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return slice(0, 0)
    return slice(indices[0].item(), indices[-1].item() + 1)


@dataclasses.dataclass
class Topography:
    data: torch.Tensor
    coords: LatLonCoordinates

    def __post_init__(self):
        if len(self.data.shape) != 2:
            raise ValueError(f"Topography data must be 2D. Got shape {self.data.shape}")
        if self.data.shape[0] != len(self.coords.lat) or self.data.shape[1] != len(
            self.coords.lon
        ):
            raise ValueError(
                f"Topography data shape {self.data.shape} does not match "
                f"coordinates shape {(len(self.coords.lat), len(self.coords.lon))}"
            )

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def subset_latlon(
        self,
        lat_interval: ClosedInterval,
        lon_interval: ClosedInterval,
    ) -> "Topography":
        lat_slice = _range_to_slice(self.coords.lat, lat_interval)
        lon_slice = _range_to_slice(self.coords.lon, lon_interval)
        return self._latlon_index_slice(lat_slice=lat_slice, lon_slice=lon_slice)

    def to_device(self) -> "Topography":
        device = get_device()
        return Topography(
            data=self.data.to(device),
            coords=LatLonCoordinates(
                lat=self.coords.lat.to(device),
                lon=self.coords.lon.to(device),
            ),
        )

    def _apply_patch(self, patch: Patch):
        return self._latlon_index_slice(
            lat_slice=patch.input_slice.y, lon_slice=patch.input_slice.x
        )

    def _latlon_index_slice(
        self,
        lat_slice: slice,
        lon_slice: slice,
    ) -> "Topography":
        sliced_data = self.data[lat_slice, lon_slice]
        sliced_latlon = LatLonCoordinates(
            lat=self.coords.lat[lat_slice],
            lon=self.coords.lon[lon_slice],
        )
        return Topography(
            data=sliced_data,
            coords=sliced_latlon,
        )

    def generate_from_patches(
        self,
        patches: list[Patch],
    ) -> Generator["Topography", None, None]:
        for patch in patches:
            yield self._apply_patch(patch)


def get_normalized_topography(path: str, topography_name: str = "HGTsfc"):
    if path.endswith(".zarr"):
        topography = xr.open_zarr(path, mask_and_scale=False)[topography_name]
    else:
        topography = xr.open_dataset(path, mask_and_scale=False)[topography_name]
    if "time" in topography.dims:
        topography = topography.isel(time=0).squeeze()
    if len(topography.shape) != 2:
        raise ValueError(
            f"unexpected shape {topography.shape} for topography."
            "Currently, only lat/lon topography is supported."
        )
    lat_name, lon_name = topography.dims[-2:]
    coords = LatLonCoordinates(
        lon=torch.tensor(topography[lon_name].values),
        lat=torch.tensor(topography[lat_name].values),
    )

    topography_normalized = (topography - topography.mean()) / topography.std()

    return Topography(data=torch.tensor(topography_normalized.values), coords=coords)


def get_topography_downscale_factor(
    topography_shape: tuple[int, ...], data_coords_shape: tuple[int, ...]
):
    if len(topography_shape) != 2 or len(data_coords_shape) != 2:
        raise ValueError(
            f"Expected 2D shapes for topography {topography_shape} and "
            f"data coordinates {data_coords_shape}, got {len(topography_shape)}D "
            f"and {len(data_coords_shape)}D."
        )
    if (
        topography_shape[0] % data_coords_shape[0] != 0
        or topography_shape[1] % data_coords_shape[1] != 0
    ):
        raise ValueError(
            f"Topography shape {topography_shape} must be evenly "
            f"divisible by horizontal shape {data_coords_shape}"
        )
    topography_downscale_factor = topography_shape[0] // data_coords_shape[0]
    if topography_downscale_factor != topography_shape[1] // data_coords_shape[1]:
        raise ValueError(
            f"Topography shape {topography_shape} must have the same scale factor "
            "between lat and lon dimensions as data coordinates "
            f"shape {data_coords_shape}"
        )
    return topography_downscale_factor
