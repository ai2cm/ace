import dataclasses

import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.downscaling.data.utils import ClosedInterval, roll_lon_coords, roll_lon_data


@dataclasses.dataclass
class StaticInput:
    data: torch.Tensor

    def __post_init__(self):
        if len(self.data.shape) != 2:
            raise ValueError(
                f"StaticInput data must be 2D. Got shape {self.data.shape}"
            )

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def subset(
        self,
        lat_slice: slice,
        lon_slice: slice,
    ) -> "StaticInput":
        return StaticInput(data=self.data[lat_slice, lon_slice])

    def to_device(self) -> "StaticInput":
        return StaticInput(data=self.data.to(get_device()))

    def get_state(self) -> dict:
        return {"data": self.data.cpu()}

    @classmethod
    def from_state(cls, state: dict) -> "StaticInput":
        return cls(data=state["data"].to(get_device(), copy=True))


_LAT_NAMES = ("lat", "latitude", "grid_yt")
_LON_NAMES = ("lon", "longitude", "grid_xt")


def _load_coords_from_ds(ds: xr.Dataset) -> LatLonCoordinates:
    lat_name = next((n for n in _LAT_NAMES if n in ds.coords), None)
    lon_name = next((n for n in _LON_NAMES if n in ds.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(
            "Could not find lat/lon coordinates in dataset. "
            f"Expected one of {_LAT_NAMES} for lat and {_LON_NAMES} for lon."
        )
    return LatLonCoordinates(
        lat=torch.tensor(ds[lat_name].values, dtype=torch.float32),
        lon=torch.tensor(ds[lon_name].values, dtype=torch.float32),
    )


def _open_ds_from_path(path: str) -> xr.Dataset:
    if path.endswith(".zarr"):
        ds = xr.open_zarr(path, mask_and_scale=False)
    else:
        ds = xr.open_dataset(path, mask_and_scale=False)
    return ds


def load_coords_from_path(path: str) -> LatLonCoordinates:
    ds = _open_ds_from_path(path)
    return _load_coords_from_ds(ds)


def _get_normalized_static_input(
    path: str, field_name: str
) -> tuple[StaticInput, LatLonCoordinates]:
    """
    Load a static input field from a given file path and field name and
    normalize it.

    Only supports 2D lat/lon static inputs. If the input has a time dimension, it is
    squeezed by taking the first time step.

    Raises ValueError if lat/lon coordinates are not found in the dataset.
    """
    ds = _open_ds_from_path(path)
    coords = _load_coords_from_ds(ds)
    da = ds[field_name]

    if "time" in da.dims:
        da = da.isel(time=0).squeeze()
    if len(da.shape) != 2:
        raise ValueError(
            f"unexpected shape {da.shape} for static input. "
            "Currently, only lat/lon static input is supported."
        )

    static_input_normalized = (da - da.mean()) / da.std()
    return StaticInput(
        data=torch.tensor(static_input_normalized.values, dtype=torch.float32),
    ), coords


def _has_legacy_coords_in_state(state: dict) -> bool:
    return bool(state.get("fields")) and "coords" in state["fields"][0]


def _has_coords_in_state(state: dict) -> bool:
    return "coords" in state or _has_legacy_coords_in_state(state)


def _sync_state_coordinates(state: dict) -> dict:
    """Migrate old per-field coord format to top-level coords format."""
    if _has_legacy_coords_in_state(state):
        state = dict(state)
        state["coords"] = state["fields"][0]["coords"]
    return state


def _validate_coords(
    case: str, coord1: LatLonCoordinates, coord2: LatLonCoordinates
) -> None:
    if coord1 != coord2:
        raise ValueError(f"Coordinates do not match between static inputs: {case}")


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
                f"Coordinates shape {self.coords.shape} does not match "
                f"fields shape {self.fields[0].shape}."
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
            fields=[
                StaticInput(data=field.data[lat_slice, lon_slice])
                for field in self.fields
            ],
            coords=LatLonCoordinates(
                lat=lat_interval.subset_of(self.coords.lat),
                lon=lon_interval.subset_of(self.coords.lon),
            ),
        )

    def roll(self, roll_amount: int, lon_start: float) -> "StaticInputs":
        """
        Roll the data and lon coordinates of the StaticInputs by the specified amount.
        """
        if roll_amount == 0:
            return self
        rolled_lon = roll_lon_coords(self.coords.lon, roll_amount, lon_start)
        return StaticInputs(
            fields=[
                StaticInput(data=roll_lon_data(f.data, roll_amount, lon_dim=-1))
                for f in self.fields
            ],
            coords=LatLonCoordinates(lat=self.coords.lat, lon=rolled_lon),
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
        if not _has_coords_in_state(state):
            raise ValueError(
                "No coordinates found in state for StaticInputs. Load with "
                "from_state_backwards_compatible if loading from a checkpoint "
                "saved prior to current coordinate serialization format."
            )
        state = _sync_state_coordinates(state)
        return cls(
            fields=[
                StaticInput.from_state(field_state) for field_state in state["fields"]
            ],
            coords=LatLonCoordinates(
                lat=state["coords"]["lat"].to(get_device(), copy=True),
                lon=state["coords"]["lon"].to(get_device(), copy=True),
            ),
        )

    @classmethod
    def from_state_backwards_compatible(
        cls,
        state: dict,
        static_inputs_config: dict[str, str],
    ) -> "StaticInputs | None":
        if state and static_inputs_config:
            raise ValueError(
                "Checkpoint contains static inputs but static_inputs_config is "
                "also provided. Backwards compatibility loading only supports "
                "a single source of StaticInputs info."
            )
        if _has_coords_in_state(state):
            return cls.from_state(state)
        elif static_inputs_config:
            return load_static_inputs(static_inputs_config)
        else:
            return None


def load_static_inputs(
    static_inputs_config: dict[str, str],
) -> StaticInputs:
    """
    Load normalized static inputs from a mapping of field names to file paths.

    Coordinates are read from each field's source dataset and validated to be
    consistent between fields. Raises ValueError if any field's dataset lacks
    lat/lon coordinates or if coordinates differ between fields.
    """
    coords_to_use = None
    fields = []
    for field_name, path in static_inputs_config.items():
        si, coords = _get_normalized_static_input(path, field_name)
        fields.append(si)
        if coords_to_use is None:
            coords_to_use = coords
        else:
            _validate_coords(field_name, coords, coords_to_use)

    if coords_to_use is None:
        raise ValueError("load_static_inputs requires at least one field.")

    return StaticInputs(fields=fields, coords=coords_to_use)
