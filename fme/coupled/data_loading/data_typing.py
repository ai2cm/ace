import dataclasses
import datetime

import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import (
    HorizontalCoordinates,
    OptionalDepthCoordinate,
    OptionalHybridSigmaPressureCoordinate,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.dataset import DatasetABC, DatasetItem
from fme.core.dataset.properties import DatasetProperties


@dataclasses.dataclass
class CoupledCoords:
    """
    Convenience wrapper for the coords in dict format.
    """

    ocean_vertical: dict[str, np.ndarray] | None = None
    ice_vertical: dict[str, np.ndarray]| None = None
    atmosphere_vertical: dict[str, np.ndarray] | None = None
    ocean_horizontal: dict[str, np.ndarray] | None = None
    ice_horizontal: dict[str, np.ndarray] | None = None
    atmosphere_horizontal: dict[str, np.ndarray] | None = None

    @property
    def ocean(self) -> dict[str, np.ndarray]:
        if self.ocean_horizontal is not None:
            return {**self.ocean_vertical, **self.ocean_horizontal}
        else:
            raise AttributeError("Ocean component is None")

    @property
    def ice(self) -> dict[str, np.ndarray]:
        if self.ice_horizontal is not None:
            if self.ice_vertical is not None:
                return {**self.ice_vertical, **self.ice_horizontal}
            return {**self.ice_horizontal}
        else:
            raise AttributeError("Ice component is None")

    @property
    def atmosphere(self) -> dict[str, np.ndarray]:
        if self.atmosphere_horizontal is not None:
            return {**self.atmosphere_vertical, **self.atmosphere_horizontal}
        else:
            raise AttributeError("Atmosphere component is None")


class CoupledVerticalCoordinate:
    def __init__(
        self,
        ocean: OptionalDepthCoordinate | None = None,
        ice: OptionalDepthCoordinate | None = None,
        atmosphere: OptionalHybridSigmaPressureCoordinate | None = None,
    ):
        self.ocean = ocean
        self.ice = ice
        self.atmosphere = atmosphere

    def __eq__(self, other):
        if not isinstance(other, CoupledVerticalCoordinate):
            return False
        if self.atmosphere is None:
            ocean_check = self.ocean == other.ocean
            ice_check = self.ice == other.ice
            return ocean_check and ice_check
        elif self.ice is None:
            ocean_check = self.ocean == other.ocean
            atmos_check = self.atmosphere == other.atmosphere
            return ocean_check and atmos_check
        elif self.ocean is None:
            ice_check = self.ice == other.ice
            atmos_check = self.atmosphere == other.atmosphere
            return ice_check and atmos_check
        else:
            ocean_check = self.ocean == other.ocean
            ice_check = self.ice == other.ice
            atmos_check = self.atmosphere == other.atmosphere
            return ocean_check and ice_check and atmos_check

    def to(self, device: torch.device) -> "CoupledVerticalCoordinate":
        ocean_device = None
        ice_device = None
        atmos_device = None
        if self.ocean is not None:
            ocean_device = self.ocean.to(device)
        if self.ice is not None:
            ice_device = self.ice.to(device)
        if self.atmosphere is not None:
            atmos_device = self.atmosphere.to(device)
        return CoupledVerticalCoordinate(
            ocean=ocean_device,
            ice=ice_device,
            atmosphere=atmos_device
        )


class CoupledHorizontalCoordinates:
    def __init__(
        self,
        ocean: HorizontalCoordinates | None = None,
        ice: HorizontalCoordinates | None = None,
        atmosphere: HorizontalCoordinates | None = None,
    ):
        self.ocean = ocean
        self.ice = ice
        self.atmosphere = atmosphere

    def __eq__(self, other):
        if not isinstance(other, CoupledHorizontalCoordinates):
            return False
        if self.atmosphere is None:
            ocean_check = self.ocean == other.ocean
            ice_check = self.ice == other.ice
            return ocean_check and ice_check
        elif self.ice is None:
            ocean_check = self.ocean == other.ocean
            atmos_check = self.atmosphere == other.atmosphere
            return ocean_check and atmos_check
        elif self.ocean is None:
            ice_check = self.ice == other.ice
            atmos_check = self.atmosphere == other.atmosphere
            return ice_check and atmos_check
        else:
            ocean_check = self.ocean == other.ocean
            ice_check = self.ice == other.ice
            atmos_check = self.atmosphere == other.atmosphere
            return ocean_check and ice_check and atmos_check

    def to(self, device: torch.device) -> "CoupledHorizontalCoordinates":
        ocean_device = None
        ice_device = None
        atmos_device = None
        if self.ocean is not None:
            ocean_device = self.ocean.to(device)
        if self.ice is not None:
            ice_device = self.ice.to(device)
        if self.atmosphere is not None:
            atmos_device = self.atmosphere.to(device)
        return CoupledHorizontalCoordinates(
            ocean=ocean_device,
            ice=ice_device,
            atmosphere=atmos_device
        )


class CoupledDatasetProperties:
    def __init__(
        self,
        ocean: DatasetProperties | None = None,
        ice: DatasetProperties | None = None,
        atmosphere: DatasetProperties | None = None,
    ):
        self.ocean = ocean
        self.ice = ice
        self.atmosphere = atmosphere
        ocean_vcoord = None
        ice_vcoord = None
        atmos_vcoord = None
        ocean_hcoord = None
        ice_hcoord = None
        atmos_hcoord = None
        if self.ocean is not None:
            ocean_vcoord = ocean.vertical_coordinate
            ocean_hcoord = ocean.horizontal_coordinates
            assert isinstance(ocean_vcoord, OptionalDepthCoordinate)
        if self.ice is not None:
            ice_vcoord = ice.vertical_coordinate
            ice_hcoord = ice.horizontal_coordinates
            assert isinstance(ice_vcoord, OptionalDepthCoordinate)
        if self.atmosphere is not None:
            atmos_vcoord = atmosphere.vertical_coordinate
            atmos_hcoord = atmosphere.horizontal_coordinates
            assert isinstance(atmos_vcoord, OptionalHybridSigmaPressureCoordinate)

        self._vertical_coordinate = CoupledVerticalCoordinate(
            ocean_vcoord,
            ice_vcoord,
            atmos_vcoord
        )
        self._horizontal_coordinates = CoupledHorizontalCoordinates(
            ocean_hcoord,
            ice_hcoord,
            atmos_hcoord
        )

    @property
    def atmosphere_timestep(self) -> datetime.timedelta:
        assert self.atmosphere.timestep is not None
        return self.atmosphere.timestep

    @property
    def ocean_timestep(self) -> datetime.timedelta:
        assert self.ocean.timestep is not None
        return self.ocean.timestep

    @property
    def ice_timestep(self) -> datetime.timedelta:
        assert self.ice.timestep is not None
        return self.ice.timestep

    @property
    def vertical_coordinate(self) -> CoupledVerticalCoordinate:
        return self._vertical_coordinate

    @property
    def horizontal_coordinates(self) -> CoupledHorizontalCoordinates:
        return self._horizontal_coordinates

    @property
    def variable_metadata(self) -> dict[str, VariableMetadata]:
        metadata: dict[str, VariableMetadata] = {}
        if self.ocean is not None:
            metadata.update(self.ocean.variable_metadata)
        if self.ice is not None:
            metadata.update(self.ice.variable_metadata)
        if self.atmosphere is not None:
            metadata.update(self.atmosphere.variable_metadata)
        return metadata

    @property
    def timestep(self) -> datetime.timedelta:
        if self.ocean is not None:
            return self.ocean_timestep
        else:
            return self.ice_timestep

    @property
    def is_remote(self) -> bool:
        if self.atmosphere is None:
            remote_ocean = self.ocean.is_remote
            remote_ice = self.ice.is_remote
            remote = remote_ocean or remote_ice
        elif self.ice is None:
            remote_atmos = self.atmosphere.is_remote
            remote_ocean = self.ocean.is_remote
            remote = remote_atmos or remote_ocean
        elif self.ocean is None:
            remote_ice = self.ice.is_remote
            remote_atmos = self.atmosphere.is_remote
            remote = remote_ice or remote_atmos
        else:
            remote_ice = self.ice.is_remote
            remote_atmos = self.atmosphere.is_remote
            remote_ocean = self.ocean.is_remote
            remote = remote_ice or remote_atmos or remote_ocean
        return remote

    @property
    def n_inner_steps(self) -> int:
        if self.atmosphere is None:
            return self.ocean_timestep // self.ice_timestep
        elif self.ice is None:
            return self.ocean_timestep // self.atmosphere_timestep
        elif self.ocean is None:
            return self.ice_timestep // self.atmosphere_timestep
        else:
            return self.ocean_timestep // self.atmosphere_timestep

    @property
    def coords(self) -> CoupledCoords:
        ocean_vcoord = None
        ocean_hcoord = None
        atmos_vcoord = None
        atmos_hcoord = None
        ice_hcoord = None
        ice_vcoord = None
        if self.ocean is not None:
            ocean_vcoord = self.vertical_coordinate.ocean.coords
            ocean_hcoord = dict(self.horizontal_coordinates.ocean.coords)
        if self.ice is not None:
            ice_vcoord = self.vertical_coordinate.ice.coords
            ice_hcoord = dict(self.horizontal_coordinates.ice.coords)
        if self.atmosphere is not None:
            atmos_vcoord = self.vertical_coordinate.atmosphere.coords
            atmos_hcoord = dict(self.horizontal_coordinates.atmosphere.coords)
        return CoupledCoords(
            ocean_vertical=ocean_vcoord,
            atmosphere_vertical=atmos_vcoord,
            ocean_horizontal=ocean_hcoord,
            ice_horizontal=ice_hcoord,
            atmosphere_horizontal=atmos_hcoord,
        )

    def to_device(self) -> "CoupledDatasetProperties":
        ocean_device = None
        ice_device = None
        atmos_device = None
        if self.ocean is not None:
            ocean_device = self.ocean.to_device()
        if self.ice is not None:
            ice_device = self.ice.to_device()
        if self.atmosphere is not None:
            atmos_device = self.atmosphere.to_device()
        return CoupledDatasetProperties(
            ocean=ocean_device,
            ice=ice_device,
            atmosphere=atmos_device
        )

    def update(self, other: "CoupledDatasetProperties"):
        if self.vertical_coordinate != other.vertical_coordinate:
            raise ValueError("Vertical coordinates must be the same for both datasets.")
        if self.horizontal_coordinates != other.horizontal_coordinates:
            raise ValueError(
                "Horizontal coordinates must be the same for both datasets."
            )
        if self.atmosphere is not None:
            self.atmosphere.update(other.atmosphere)
        if self.ocean is not None:
            self.ocean.update(other.ocean)
        if self.ice is not None:
            self.ice.update(other.ice)


@dataclasses.dataclass
class CoupledDatasetItem:
    ocean: DatasetItem | None = None
    ice: DatasetItem | None = None
    atmosphere: DatasetItem | None = None


class CoupledDataset:
    def __init__(
        self,
        properties: CoupledDatasetProperties,
        n_steps_fast: int,
        ocean: DatasetABC | None = None,
        ice: DatasetABC | None = None,
        atmosphere: DatasetABC | None = None,
    ):
        """
        Args:
            ocean: ocean dataset.
            ice: ice dataset.
            atmosphere: atmosphere dataset.
            properties: the coupled dataset properties.
            n_steps_fast: number of atmosphere timesteps per ocean timestep.
        """
        self._ocean = ocean
        self._ice = ice
        self._atmosphere = atmosphere
        if self._atmosphere is None:
            if properties.ocean_timestep != properties.ice_timestep * n_steps_fast:
                raise ValueError(
                    "Ocean and Ice timesteps must be consistent with "
                    f"n_steps_fast, got ocean timestep {properties.ocean.timestep} "
                    f"and ice timestep {properties.ice.timestep} "
                    f"with n_steps_fast={n_steps_fast}."
                )
            ocean_time = self._ocean[0][1].isel(time=0).item()
            ice_time = self._ice[0][1].isel(time=0).item()
            if ocean_time != ice_time:
                raise ValueError(
                    f"First time of ocean dataset is {ocean_time} "
                    f"but the ice's first time is {ice_time}. "
                    "Maybe align the datasets using a subset?"
                )
        elif self._ice is None:
            if properties.ocean_timestep != properties.atmosphere_timestep * n_steps_fast:
                raise ValueError(
                    "Ocean and Atmosphere timesteps must be consistent with "
                    f"n_steps_fast, got ocean timestep {properties.ocean.timestep} "
                    f"and atmosphere timestep {properties.atmosphere.timestep} "
                    f"with n_steps_fast={n_steps_fast}."
                )
            ocean_time = self._ocean[0][1].isel(time=0).item()
            atmos_time = self._atmosphere[0][1].isel(time=0).item()
            if ocean_time != atmos_time:
                raise ValueError(
                    f"First time of ocean dataset is {ocean_time} "
                    f"but the atmosphere's first time is {atmos_time}. "
                    "Maybe align the datasets using a subset?"
                )
        elif self._ocean is None:
            if properties.ice_timestep != properties.atmosphere_timestep:
                raise ValueError(
                    "Ice and Atmosphere timesteps must be consistent, "
                    f"got ice timestep {properties.ice.timestep} "
                    f"and atmosphere timestep {properties.atmosphere.timestep}."
                )
            ice_time = self._ice[0][1].isel(time=0).item()
            atmos_time = self._atmosphere[0][1].isel(time=0).item()
            if ice_time != atmos_time:
                raise ValueError(
                    f"First time of ice dataset is {ice_time} "
                    f"but the atmosphere's first time is {atmos_time}. "
                    "Maybe align the datasets using a subset?"
                )
        else:
            if properties.ice_timestep != properties.atmosphere_timestep:
                raise ValueError(
                    "Ice and Atmosphere timesteps must be consistent, "
                    f"got ice timestep {properties.ice.timestep} "
                    f"and atmosphere timestep {properties.atmosphere.timestep}."
                )
            if properties.ocean_timestep != properties.atmosphere_timestep * n_steps_fast:
                raise ValueError(
                    "Ocean and Atmosphere timesteps must be consistent with "
                    f"n_steps_fast, got ocean timestep {properties.ocean.timestep} "
                    f"and atmosphere timestep {properties.atmosphere.timestep} "
                    f"with n_steps_fast={n_steps_fast}."
                )
            ice_time = self._ice[0][1].isel(time=0).item()
            ocean_time = self._ocean[0][1].isel(time=0).item()
            atmos_time = self._atmosphere[0][1].isel(time=0).item()
            if ice_time != atmos_time:
                raise ValueError(
                    f"First time of ice dataset is {ice_time} "
                    f"but the atmosphere's first time is {atmos_time}. "
                    "Maybe align the datasets using a subset?"
                )
            if ocean_time != atmos_time:
                raise ValueError(
                    f"First time of ocean dataset is {ocean_time} "
                    f"but the atmosphere's first time is {atmos_time}. "
                    "Maybe align the datasets using a subset?"
                )

        self._properties = properties
        self._n_steps_fast = n_steps_fast

    @property
    def properties(self) -> CoupledDatasetProperties:
        return self._properties

    @property
    def n_steps_fast(self) -> int:
        return self._n_steps_fast

    @property
    def all_ic_times(self) -> xr.CFTimeIndex:
        if self._atmosphere is None:
            return self._ocean.sample_start_times
        elif self._ice is None:
            return self._ocean.sample_start_times
        elif self._ocean is None:
            return self._ice.sample_start_times
        else:
            return self._ocean.sample_start_times

    def __len__(self):
        if self._atmosphere is None:
            return min([len(self._ocean), len(self._ice)])
        elif self._ice is None:
            return min([len(self._ocean), len(self._atmosphere)])
        elif self._ocean is None:
            return min([len(self._ice), len(self._atmosphere)])
        else:
            return min([len(self._ocean), len(self._ice), len(self._atmosphere)])

    def __getitem__(self, idx: int) -> CoupledDatasetItem:
        fast_idx = idx * self._n_steps_fast
        ocean = None
        ice = None
        atmosphere = None
        if self._ocean is not None:
            ocean = self._ocean[idx]
        if self._ice is not None:
            ice = self._ice[fast_idx]
        if self._atmosphere is not None:
            atmosphere = self._atmosphere[fast_idx]
        return CoupledDatasetItem(ocean=ocean, ice=ice, atmosphere=atmosphere)

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        if self._atmosphere is None:
            try:
                self._ocean.validate_inference_length(max_start_index, max_window_len)
            except ValueError as e:
                raise ValueError(
                    "The ocean dataset has an insufficient number of timepoints."
                ) from e
            ice_max_start_index = max_start_index * self.n_steps_fast
            ice_max_window_len = (max_window_len - 1) * self.n_steps_fast + 1
            try:
                self._ice.validate_inference_length(
                    ice_max_start_index, ice_max_window_len
                )
            except ValueError as e:
                raise ValueError(
                    "The ice dataset has an insufficient number of timepoints."
                ) from e
        elif self._ice is None:
            try:
                self._ocean.validate_inference_length(max_start_index, max_window_len)
            except ValueError as e:
                raise ValueError(
                    "The ocean dataset has an insufficient number of timepoints."
                ) from e
            atmos_max_start_index = max_start_index * self.n_steps_fast
            atmos_max_window_len = (max_window_len - 1) * self.n_steps_fast + 1
            try:
                self._atmosphere.validate_inference_length(
                    atmos_max_start_index, atmos_max_window_len
                )
            except ValueError as e:
                raise ValueError(
                    "The atmosphere dataset has an insufficient number of timepoints."
                ) from e
        elif self._ocean is None:
            try:
                self._ice.validate_inference_length(max_start_index, max_window_len)
            except ValueError as e:
                raise ValueError(
                    "The ice dataset has an insufficient number of timepoints."
                ) from e
            atmos_max_start_index = max_start_index * self.n_steps_fast
            atmos_max_window_len = (max_window_len - 1) * self.n_steps_fast + 1
            try:
                self._atmosphere.validate_inference_length(
                    atmos_max_start_index, atmos_max_window_len
                )
            except ValueError as e:
                raise ValueError(
                    "The atmosphere dataset has an insufficient number of timepoints."
                ) from e
        else:
            try:
                self._ocean.validate_inference_length(max_start_index, max_window_len)
            except ValueError as e:
                raise ValueError(
                    "The ocean dataset has an insufficient number of timepoints."
                ) from e
            ice_max_start_index = max_start_index * self.n_steps_fast
            ice_max_window_len = (max_window_len - 1) * self.n_steps_fast + 1
            try:
                self._ice.validate_inference_length(
                    ice_max_start_index, ice_max_window_len
                )
            except ValueError as e:
                raise ValueError(
                    "The ice dataset has an insufficient number of timepoints."
                ) from e
            atmos_max_start_index = max_start_index * self.n_steps_fast
            atmos_max_window_len = (max_window_len - 1) * self.n_steps_fast + 1
            try:
                self._atmosphere.validate_inference_length(
                    atmos_max_start_index, atmos_max_window_len
                )
            except ValueError as e:
                raise ValueError(
                    "The atmosphere dataset has an insufficient number of timepoints."
                ) from e

    def set_epoch(self, epoch: int):
        if self._ocean is not None:
            self._ocean.set_epoch(epoch)
        if self._ice is not None:
            self._ice.set_epoch(epoch)
        if self._atmosphere is not None:
            self._atmosphere.set_epoch(epoch)
