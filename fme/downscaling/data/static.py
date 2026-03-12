import dataclasses

import torch
import xarray as xr

from fme.core.device import get_device
from fme.downscaling.data.patching import Patch


@dataclasses.dataclass
class StaticInput:
    data: torch.Tensor

    def __post_init__(self):
        if len(self.data.shape) != 2:
            raise ValueError(f"Topography data must be 2D. Got shape {self.data.shape}")

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
        device = get_device()
        return StaticInput(data=self.data.to(device))

    def _apply_patch(self, patch: Patch):
        return self.subset(lat_slice=patch.input_slice.y, lon_slice=patch.input_slice.x)

    def get_state(self) -> dict:
        return {
            "data": self.data.cpu(),
        }


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

    def __post_init__(self):
        for i, field in enumerate(self.fields[1:]):
            if field.shape != self.fields[0].shape:
                raise ValueError(
                    f"All StaticInput fields must have the same shape. "
                    f"Fields {i + 1} and 0 do not match shapes."
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
        lat_slice: slice,
        lon_slice: slice,
    ) -> "StaticInputs":
        return StaticInputs(
            fields=[field.subset(lat_slice, lon_slice) for field in self.fields]
        )

    def to_device(self) -> "StaticInputs":
        return StaticInputs(fields=[field.to_device() for field in self.fields])

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
                )
                for field_state in state["fields"]
            ]
        )


def load_static_inputs(
    static_inputs_config: dict[str, str] | None,
) -> "StaticInputs | None":
    """
    Load normalized static inputs from a mapping of field names to file paths.
    Returns None if the input config is empty.
    """
    # TODO: consolidate/simplify empty StaticInputs vs. None handling in
    #       downscaling code
    if not static_inputs_config:
        return None
    return StaticInputs(
        fields=[
            _get_normalized_static_input(path, field_name)
            for field_name, path in static_inputs_config.items()
        ]
    )
