import dataclasses
import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import (
    HorizontalCoordinates,
    OptionalDepthCoordinate,
    OptionalHybridSigmaPressureCoordinate,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.xarray import DatasetProperties
from fme.core.typing_ import TensorDict


@dataclasses.dataclass
class CoupledCoords:
    """
    Convenience wrapper for the coords in dict format.
    """

    ocean_vertical: Dict[str, np.ndarray]
    atmosphere_vertical: Dict[str, np.ndarray]
    horizontal: Dict[str, np.ndarray]

    @property
    def ocean(self) -> Dict[str, np.ndarray]:
        return {**self.ocean_vertical, **self.horizontal}

    @property
    def atmosphere(self) -> Dict[str, np.ndarray]:
        return {**self.atmosphere_vertical, **self.horizontal}


class CoupledVerticalCoordinate:
    def __init__(
        self,
        ocean: OptionalDepthCoordinate,
        atmosphere: OptionalHybridSigmaPressureCoordinate,
    ):
        self.ocean = ocean
        self.atmosphere = atmosphere

    def __eq__(self, other):
        if not isinstance(other, CoupledVerticalCoordinate):
            return False
        return self.ocean == other.ocean and self.atmosphere == other.atmosphere

    def to(self, device: torch.device) -> "CoupledVerticalCoordinate":
        return CoupledVerticalCoordinate(
            ocean=self.ocean.to(device), atmosphere=self.atmosphere.to(device)
        )


class CoupledDatasetProperties:
    def __init__(
        self,
        all_ic_times: xr.CFTimeIndex,
        ocean: DatasetProperties,
        atmosphere: DatasetProperties,
    ):
        self.all_ic_times = all_ic_times
        self.ocean = ocean
        self.atmosphere = atmosphere
        ocean_coord = ocean.vertical_coordinate
        atmos_coord = atmosphere.vertical_coordinate
        assert isinstance(ocean_coord, OptionalDepthCoordinate)
        assert isinstance(atmos_coord, OptionalHybridSigmaPressureCoordinate)
        self._vertical_coordinate = CoupledVerticalCoordinate(ocean_coord, atmos_coord)

    @property
    def vertical_coordinate(self) -> CoupledVerticalCoordinate:
        return self._vertical_coordinate

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self.atmosphere.horizontal_coordinates

    @property
    def variable_metadata(self) -> Dict[str, VariableMetadata]:
        metadata: Dict[str, VariableMetadata] = {}
        metadata.update(self.ocean.variable_metadata)
        metadata.update(self.atmosphere.variable_metadata)
        return metadata

    @property
    def timestep(self) -> datetime.timedelta:
        return self.ocean.timestep

    @property
    def is_remote(self) -> bool:
        return self.ocean.is_remote or self.atmosphere.is_remote

    @property
    def n_inner_steps(self) -> int:
        return self.ocean.timestep // self.atmosphere.timestep

    @property
    def coords(self) -> CoupledCoords:
        return CoupledCoords(
            ocean_vertical=self.vertical_coordinate.ocean.coords,
            atmosphere_vertical=self.vertical_coordinate.atmosphere.coords,
            horizontal=dict(self.horizontal_coordinates.coords),
        )

    def to_device(self) -> "CoupledDatasetProperties":
        return CoupledDatasetProperties(
            all_ic_times=self.all_ic_times,
            ocean=self.ocean.to_device(),
            atmosphere=self.atmosphere.to_device(),
        )

    def update(self, other: "CoupledDatasetProperties"):
        if (self.all_ic_times.values != other.all_ic_times.values).any():
            raise ValueError("All times must be the same for both datasets.")
        if self.vertical_coordinate != other.vertical_coordinate:
            raise ValueError("Vertical coordinates must be the same for both datasets.")
        if self.horizontal_coordinates != other.horizontal_coordinates:
            raise ValueError(
                "Horizontal coordinates must be the same for both datasets."
            )
        self.atmosphere.update(other.atmosphere)
        self.ocean.update(other.ocean)


@dataclasses.dataclass
class CoupledDatasetItem:
    ocean: Tuple[TensorDict, xr.DataArray]
    atmosphere: Tuple[TensorDict, xr.DataArray]


class CoupledDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ocean: torch.utils.data.Dataset,
        atmosphere: torch.utils.data.Dataset,
        properties: CoupledDatasetProperties,
        n_steps_fast: int,
    ):
        """
        Args:
            ocean: ocean dataset.
            atmosphere: atmosphere dataset.
            properties: the coupled dataset properties.
            n_steps_fast: number of atmosphere timesteps per ocean timestep.
        """
        self._ocean = ocean
        if properties.ocean.timestep != properties.atmosphere.timestep * n_steps_fast:
            raise ValueError(
                "Ocean and atmosphere timesteps must be consistent with "
                f"n_steps_fast, got ocean timestep {properties.ocean.timestep} "
                f"and atmosphere timestep {properties.atmosphere.timestep} "
                f"with n_steps_fast={n_steps_fast}."
            )
        self._atmosphere = atmosphere
        ocean_time = ocean[0][1].isel(time=0).item()
        atmos_time = atmosphere[0][1].isel(time=0).item()
        if ocean_time != atmos_time:
            raise ValueError(
                f"First time of ocean dataset is {ocean_time} "
                f"but the atmosphere's first time is {atmos_time}. "
                "Maybe align the datasets using a subset?"
            )

        self._properties = properties
        self._n_steps_fast = n_steps_fast

    @property
    def variable_metadata(self) -> Dict[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def vertical_coordinate(self) -> CoupledVerticalCoordinate:
        return self._properties.vertical_coordinate

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._properties.horizontal_coordinates

    @property
    def is_remote(self) -> bool:
        return self._properties.is_remote

    @property
    def properties(self) -> CoupledDatasetProperties:
        return self._properties

    @property
    def n_steps_fast(self) -> int:
        return self._n_steps_fast

    @property
    def all_ic_times(self) -> xr.CFTimeIndex:
        return self.properties.all_ic_times

    def __len__(self):
        return min([len(self._ocean), len(self._atmosphere)])

    def __getitem__(self, idx: int) -> CoupledDatasetItem:
        fast_idx = idx * self._n_steps_fast
        ocean = self._ocean[idx]
        atmosphere = self._atmosphere[fast_idx]
        return CoupledDatasetItem(ocean=ocean, atmosphere=atmosphere)
