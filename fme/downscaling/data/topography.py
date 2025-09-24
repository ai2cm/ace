import dataclasses
from collections.abc import Sequence

import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.downscaling.data.utils import BatchedLatLonCoordinates, ClosedInterval


@dataclasses.dataclass
class Topography:
    data: torch.Tensor
    coords: LatLonCoordinates

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        if self.data.shape[0] != len(self.coords.lat) or self.data.shape[1] != len(
            self.coords.lon
        ):
            raise ValueError(
                f"Topography data shape {self.data.shape} does not match "
                f"coordinates shape {(len(self.coords.lat), len(self.coords.lon))}"
            )
        return self.data.shape

    def subset_latlon(
        self,
        lat_interval: ClosedInterval,
        lon_interval: ClosedInterval,
    ) -> "Topography":
        subset_lats = torch.tensor(
            [
                i
                for i in range(len(self.coords.lat))
                if float(self.coords.lat[i]) in lat_interval
            ]
        )
        subset_lons = torch.tensor(
            [
                i
                for i in range(len(self.coords.lon))
                if float(self.coords.lon[i]) in lon_interval
            ]
        )
        subset_topography = torch.index_select(self.data, -2, subset_lats)
        subset_topography = torch.index_select(subset_topography, -1, subset_lons)
        return Topography(
            data=subset_topography,
            coords=LatLonCoordinates(
                lat=self.coords.lat[subset_lats], lon=self.coords.lon[subset_lons]
            ),
        )

    def to(self, device: torch.device) -> "Topography":
        return Topography(
            data=self.data.to(device),
            coords=LatLonCoordinates(
                lat=self.coords.lat.to(device),
                lon=self.coords.lon.to(device),
            ),
        )


@dataclasses.dataclass
class BatchedTopography:
    data: torch.Tensor
    coords: BatchedLatLonCoordinates

    @classmethod
    def from_sequence(
        cls,
        items: Sequence["Topography"],
    ) -> "BatchedTopography":
        topo = torch.utils.data.default_collate([i.data for i in items])
        coords = BatchedLatLonCoordinates.from_sequence([i.coords for i in items])
        return cls(topo, coords)

    def __getitem__(self, k):
        return Topography(self.data[k], self.coords[k])

    def to(self, device: torch.device) -> "BatchedTopography":
        return BatchedTopography(
            data=self.data.to(device),
            coords=BatchedLatLonCoordinates(
                lat=self.coords.lat.to(device),
                lon=self.coords.lon.to(device),
            ),
        )


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
