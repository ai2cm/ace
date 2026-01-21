import dataclasses
from collections.abc import Generator, Iterator

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


# TODO: rename to StaticInput, make _apply_patch public
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

    def to_state(self) -> dict:
        return {
            "data": self.data.cpu(),
            "coords": self.coords.to_state(),
        }


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

    return Topography(
        data=torch.tensor(topography_normalized.values, dtype=torch.float32),
        coords=coords,
    )


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


@dataclasses.dataclass
class StaticInputs:
    fields: list[Topography]

    def __post_init__(self):
        for i, field in enumerate(self.fields[1:]):
            if field.coords != self.fields[0].coords:
                raise ValueError(
                    f"All StaticInput fields must have the same coordinates. "
                    f"Fields {i} and 0 do not match coordinates."
                )

    def __getitem__(self, index: int):
        return self.fields[index]

    @property
    def input_tensors(self) -> list[torch.Tensor]:
        if len(self.fields) > 0:
            return [field.data for field in self.fields]
        else:
            return torch.tensor([])

    @property
    def coords(self) -> LatLonCoordinates:
        if len(self.fields) == 0:
            raise ValueError("No fields in StaticInputs to get coordinates from.")
        return self.fields[0].coords

    @property
    def shape(self) -> tuple[int, int]:
        if len(self.fields) == 0:
            raise ValueError("No fields in StaticInputs to get shape from.")
        return self.fields[0].shape

    def subset_latlon(
        self,
        lat_interval: ClosedInterval,
        lon_interval: ClosedInterval,
    ) -> "StaticInputs":
        return StaticInputs(
            fields=[
                field.subset_latlon(lat_interval, lon_interval) for field in self.fields
            ]
        )

    def to_device(self) -> "StaticInputs":
        return StaticInputs(fields=[field.to_device() for field in self.fields])

    def generate_from_patches(
        self,
        patches: list[Patch],
    ) -> Iterator["StaticInputs"]:
        for patch in patches:
            yield StaticInputs(
                fields=[field._apply_patch(patch) for field in self.fields]
            )

    def to_state(self) -> dict:
        return {
            "fields": [field.to_state() for field in self.fields],
        }

    @classmethod
    def from_state(cls, state: dict) -> "StaticInputs":
        return cls(
            fields=[
                Topography(
                    data=field_state["data"],
                    coords=LatLonCoordinates(
                        lat=field_state["coords"]["lat"],
                        lon=field_state["coords"]["lon"],
                    ),
                )
                for field_state in state["fields"]
            ]
        )
