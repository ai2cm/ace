import dataclasses

import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.downscaling.data.utils import ClosedInterval


@dataclasses.dataclass
class StaticInput:
    data: torch.Tensor

    def __post_init__(self):
        if len(self.data.shape) != 2:
            raise ValueError(
                f"StaticInput data must be 2D. Got shape {self.data.shape}"
            )
        self._shape = (self.data.shape[0], self.data.shape[1])

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def subset(
        self,
        lat_slice: slice,
        lon_slice: slice,
    ) -> "StaticInput":
        return StaticInput(data=self.data[lat_slice, lon_slice])

    def to_device(self) -> "StaticInput":
        device = get_device()
        return StaticInput(data=self.data.to(device))

    def get_state(self) -> dict:
        return {
            "data": self.data.cpu(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "StaticInput":
        return cls(data=state["data"])


def _get_normalized_static_input(path: str, field_name: str):
    """
    Load a static input field from a given file path and field name and
    normalize it.

    Only supports 2D lat/lon static inputs. If the input has a time dimension, it is
    squeezed by taking the first time step. The lat/lon coordinates are
    assumed to be the last two dimensions of the loaded dataset dimensions.
    """
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

    static_input_normalized = (static_input - static_input.mean()) / static_input.std()

    return StaticInput(
        data=torch.tensor(static_input_normalized.values, dtype=torch.float32),
    )


@dataclasses.dataclass
class StaticInputs:
    fields: list[StaticInput]
    coords: LatLonCoordinates

    def __post_init__(self):
        for i, field in enumerate(self.fields[1:]):
            if field.shape != self.fields[0].shape:
                raise ValueError(
                    f"All StaticInput fields must have the same shape. "
                    f"Fields {i + 1} and 0 do not match shapes."
                )
        if self.fields and self.coords.shape != self.fields[0].shape:
            raise ValueError(
                f"Coordinates shape {self.coords.shape} does not match fields shape "
                f"{self.fields[0].shape} for StaticInputs."
            )

    def __getitem__(self, index: int):
        return self.fields[index]

    @property
    def shape(self) -> tuple[int, int]:
        if len(self.fields) == 0:
            raise ValueError("No fields in StaticInputs to get shape from.")
        return self.fields[0].shape

    def subset(
        self,
        lat_interval: ClosedInterval,
        lon_interval: ClosedInterval,
    ) -> "StaticInputs":
        lat_slice = lat_interval.slice_from(self.coords.lat)
        lon_slice = lon_interval.slice_from(self.coords.lon)
        return StaticInputs(
            fields=[field.subset(lat_slice, lon_slice) for field in self.fields],
            coords=LatLonCoordinates(
                lat=lat_interval.subset_of(self.coords.lat),
                lon=lon_interval.subset_of(self.coords.lon),
            ),
        )

    def to_device(self) -> "StaticInputs":
        return StaticInputs(
            fields=[field.to_device() for field in self.fields],
            coords=self.coords.to(get_device()),
        )

    def get_state(self) -> dict:
        return {
            "fields": [field.get_state() for field in self.fields],
            "coords": self.coords.get_state(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "StaticInputs":
        """Reconstruct StaticInputs from a state dict.

        Args:
            state: State dict from get_state().
            coords: Override coordinates. If None, reads coords from the state dict.
                Pass explicitly when loading old-format checkpoints that stored coords
                outside of the StaticInputs state.
        """
        return cls(
            fields=[
                StaticInput.from_state(field_state) for field_state in state["fields"]
            ],
            coords=LatLonCoordinates(
                lat=state["coords"]["lat"],
                lon=state["coords"]["lon"],
            ),
        )


def load_static_inputs(
    static_inputs_config: dict[str, str], coords: LatLonCoordinates
) -> StaticInputs:
    """
    Load normalized static inputs from a mapping of field names to file paths.
    Returns an empty StaticInputs (no fields) if the config is None or empty.
    """
    fields = [
        _get_normalized_static_input(path, field_name)
        for field_name, path in static_inputs_config.items()
    ]
    return StaticInputs(fields=fields, coords=coords)
