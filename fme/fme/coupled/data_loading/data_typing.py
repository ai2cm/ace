import dataclasses
from typing import Dict, Mapping, Tuple

import torch
import xarray as xr

from fme.core.data_loading._xarray import DatasetProperties
from fme.core.data_loading.data_typing import (
    Dataset,
    HorizontalCoordinates,
    SigmaCoordinates,
    VariableMetadata,
)
from fme.core.typing_ import TensorDict


class CoupledProperties:
    def __init__(self, ocean: DatasetProperties, atmosphere: DatasetProperties):
        self.ocean = ocean
        self.atmosphere = atmosphere

    @property
    def sigma_coordinates(self) -> SigmaCoordinates:
        return self.atmosphere.sigma_coordinates

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
    def is_remote(self) -> bool:
        return self.ocean.is_remote or self.atmosphere.is_remote

    def update(self, other: "CoupledProperties"):
        if self.sigma_coordinates != other.sigma_coordinates:
            raise ValueError("Sigma coordinates must be the same for both datasets.")
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


class CoupledDataset(Dataset):
    def __init__(
        self,
        ocean: torch.utils.data.Dataset,
        atmosphere: torch.utils.data.Dataset,
        properties: CoupledProperties,
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
        self._properties = properties
        self._n_steps_fast = n_steps_fast

    @property
    def variable_metadata(self) -> Mapping[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def sigma_coordinates(self) -> SigmaCoordinates:
        return self._properties.sigma_coordinates

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._properties.horizontal_coordinates

    @property
    def is_remote(self) -> bool:
        return self._properties.is_remote

    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> Tuple[TensorDict, xr.DataArray]:
        raise NotImplementedError(
            "CoupledDataset has no get_sample_by_time_slice method"
        )

    def __len__(self):
        return min([len(self._ocean), len(self._atmosphere)])

    def __getitem__(self, idx: int) -> CoupledDatasetItem:
        fast_idx = idx * self._n_steps_fast
        ocean = self._ocean[idx]
        atmosphere = self._atmosphere[fast_idx]
        assert atmosphere[1].isel(time=0) == ocean[1].isel(time=0)
        return CoupledDatasetItem(ocean=ocean, atmosphere=atmosphere)
