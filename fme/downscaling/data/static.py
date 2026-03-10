import dataclasses
from collections.abc import Generator, Iterator

import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.downscaling.data.patching import Patch
from fme.downscaling.data.utils import ClosedInterval


@dataclasses.dataclass
class StaticInput:
    data: torch.Tensor
    coords: LatLonCoordinates

    def __post_init__(self):
        if len(self.data.shape) != 2:
            raise ValueError(f"Topography data must be 2D. Got shape {self.data.shape}")
        if self.data.shape[0] != len(self.coords.lat) or self.data.shape[1] != len(
            self.coords.lon
        ):
            raise ValueError(
                f"Static inputs data shape {self.data.shape} does not match lat/lon "
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
    ) -> "StaticInput":
        lat_slice = lat_interval.slice_of(self.coords.lat)
        lon_slice = lon_interval.slice_of(self.coords.lon)
        return self._latlon_index_slice(lat_slice=lat_slice, lon_slice=lon_slice)

    def to_device(self) -> "StaticInput":
        device = get_device()
        return StaticInput(
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
    ) -> "StaticInput":
        sliced_data = self.data[lat_slice, lon_slice]
        sliced_latlon = LatLonCoordinates(
            lat=self.coords.lat[lat_slice],
            lon=self.coords.lon[lon_slice],
        )
        return StaticInput(
            data=sliced_data,
            coords=sliced_latlon,
        )

    def generate_from_patches(
        self,
        patches: list[Patch],
    ) -> Generator["StaticInput", None, None]:
        for patch in patches:
            yield self._apply_patch(patch)

    def get_state(self) -> dict:
        return {
            "data": self.data.cpu(),
            "coords": self.coords.get_state(),
        }


def get_normalized_static_input(path: str, field_name: str):
    if path.endswith(".zarr"):
        static_input = xr.open_zarr(path, mask_and_scale=False)[field_name]
    else:
        static_input = xr.open_dataset(path, mask_and_scale=False)[field_name]
    if "time" in static_input.dims:
        static_input = static_input.isel(time=0).squeeze()
    if len(static_input.shape) != 2:
        raise ValueError(
            f"unexpected shape {static_input.shape} for static input."
            "Currently, only lat/lon static input is supported."
        )
    lat_name, lon_name = static_input.dims[-2:]
    coords = LatLonCoordinates(
        lon=torch.tensor(static_input[lon_name].values),
        lat=torch.tensor(static_input[lat_name].values),
    )

    static_input_normalized = (static_input - static_input.mean()) / static_input.std()

    return StaticInput(
        data=torch.tensor(static_input_normalized.values, dtype=torch.float32),
        coords=coords,
    )


@dataclasses.dataclass
class StaticInputs:
    fields: list[StaticInput]

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

    def get_state(self) -> dict:
        return {
            "fields": [field.get_state() for field in self.fields],
        }

    @classmethod
    def from_state(cls, state: dict) -> "StaticInputs":
        return cls(
            fields=[
                StaticInput(
                    data=field_state["data"],
                    coords=LatLonCoordinates(
                        lat=field_state["coords"]["lat"],
                        lon=field_state["coords"]["lon"],
                    ),
                )
                for field_state in state["fields"]
            ]
        )
