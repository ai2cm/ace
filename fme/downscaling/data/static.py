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


def _load_coords_from_ds(ds: xr.Dataset) -> LatLonCoordinates:
    lat_name = next((n for n in ["lat", "latitude", "grid_yt"] if n in ds.coords), None)
    lon_name = next(
        (n for n in ["lon", "longitude", "grid_xt"] if n in ds.coords), None
    )
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not find lat/lon coordinates in dataset. "
            "Expected 'lat'/'latitude'/'grid_yt' and 'lon'/'longitude'/'grid_xt'."
        )
    return LatLonCoordinates(
        lat=torch.tensor(ds[lat_name].values, dtype=torch.float32),
        lon=torch.tensor(ds[lon_name].values, dtype=torch.float32),
    )


def load_fine_coords_from_path(path: str) -> LatLonCoordinates:
    if path.endswith(".zarr"):
        ds = xr.open_zarr(path)
    else:
        ds = xr.open_dataset(path)

    return _load_coords_from_ds(ds)


def _get_normalized_static_input(
    path: str, field_name: str
) -> tuple[StaticInput, LatLonCoordinates | None]:
    """
    Load a static input field from a given file path and field name and
    normalize it.

    Only supports 2D lat/lon static inputs. If the input has a time dimension, it is
    squeezed by taking the first time step. The lat/lon coordinates are
    assumed to be the last two dimensions of the loaded dataset dimensions.
    """
    if path.endswith(".zarr"):
        ds = xr.open_zarr(path, mask_and_scale=False)
    else:
        ds = xr.open_dataset(path, mask_and_scale=False)

    da = ds[field_name]
    try:
        coords = _load_coords_from_ds(ds)
    except ValueError:
        # no coords available
        coords = None

    if "time" in da.dims:
        da = da.isel(time=0).squeeze()
    if len(da.shape) != 2:
        raise ValueError(
            f"unexpected shape {da.shape} for static input."
            "Currently, only lat/lon static input is supported."
        )

    static_input_normalized = (da - da.mean()) / da.std()

    return StaticInput(
        data=torch.tensor(static_input_normalized.values, dtype=torch.float32),
    ), coords


def _has_legacy_coords_in_state(state: dict) -> bool:
    return "fields" in state and state["fields"] and "coords" in state["fields"][0]


def _sync_state_coordinates(state: dict) -> dict:
    # if necessary adjusts legacy coordinate to expected
    # format for state loading
    state = state.copy()
    if _has_legacy_coords_in_state(state):
        state["coords"] = state["fields"][0]["coords"]
    return state


def _has_coords_in_state(state: dict) -> bool:
    if "coords" in state or _has_legacy_coords_in_state(state):
        return True
    else:
        return False


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
                lat=state["coords"]["lat"],
                lon=state["coords"]["lon"],
            ),
        )

    @classmethod
    def from_state_backwards_compatible(
        cls,
        state: dict,
        static_inputs_config: dict[str, str],
        fine_coordinates_path: str | None,
    ) -> "StaticInputs":
        if state and static_inputs_config:
            raise ValueError(
                "Checkpoint contains static inputs but static_inputs_config is "
                "also provided. Backwards compatibility loading only supports "
                "a single source of StaticInputs info."
            )

        if fine_coordinates_path and _has_coords_in_state(state):
            raise ValueError(
                "State contains coordinates but fine_coordinates_path is also provided."
                " Only one source of coordinate info can be used for backwards "
                "compatibility loading of StaticInputs."
            )
        elif not _has_coords_in_state(state) and not fine_coordinates_path:
            raise ValueError(
                "No coordinates found in state and no fine_coordinates_path provided. "
                "Cannot load StaticInputs without coordinates."
            )

        # All compatibility cases:
        # Serialized StaticInputs exist, which always had coordinates stored
        # No serialized static inputs or specified inputs, load coordinates
        # Specified static input fields and specified coordinates

        if _has_coords_in_state(state):
            return cls.from_state(state)
        else:
            assert fine_coordinates_path is not None  # for type checker
            coords = load_fine_coords_from_path(fine_coordinates_path)

        if static_inputs_config:
            return load_static_inputs(static_inputs_config, coords)
        else:
            return cls(fields=[], coords=coords)


def _validate_coords(
    case: str, coord1: LatLonCoordinates, coord2: LatLonCoordinates
) -> None:
    if not coord1 == coord2:
        raise ValueError(f"Coordinates do not match between static inputs: {case}")


def load_static_inputs(
    static_inputs_config: dict[str, str],
    fallback_coords: LatLonCoordinates,
    validate_coords: bool = True,
) -> StaticInputs:
    """
    Load normalized static inputs from a mapping of field names to file paths.
    Returns an empty StaticInputs (no fields) if the config is empty.

    Coordinates are inferred from the static input field datasets and verified
    to match between each field. If no static inputs are provided
    coordinates are used from fallback_coords.
    """
    coords_to_use = None
    fields = []
    for field_name, path in static_inputs_config.items():
        si, coords = _get_normalized_static_input(path, field_name)
        fields.append(si)

        if coords is not None and coords_to_use is None:
            coords_to_use = coords
        elif coords is not None and validate_coords:
            assert coords_to_use is not None  # for type checker
            _validate_coords(field_name, coords, coords_to_use)

    if coords_to_use is None:
        # no coords found with static inputs, use provided fallback
        coords_to_use = fallback_coords
    elif validate_coords:
        _validate_coords("fallback", coords_to_use, fallback_coords)

    return StaticInputs(fields=fields, coords=coords_to_use)
