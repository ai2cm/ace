import dataclasses
import datetime
from typing import Dict, Mapping, Tuple

import xarray as xr

from fme.core.data_loading._xarray import XarrayDataset
from fme.core.data_loading.data_typing import (
    Dataset,
    HorizontalCoordinates,
    SigmaCoordinates,
    VariableMetadata,
)
from fme.core.typing_ import TensorDict


@dataclasses.dataclass
class CoupledDatasetItem:
    ocean: Tuple[TensorDict, xr.DataArray]
    atmosphere: Tuple[TensorDict, xr.DataArray]


class CoupledDataset(Dataset):
    def __init__(
        self,
        ocean: XarrayDataset,
        atmosphere: XarrayDataset,
        ocean_timestep: datetime.timedelta,
        n_steps_fast: int,
    ):
        """
        Args:
            ocean: ocean dataset
            atmosphere: atmosphere dataset
            ocean_timestep: ocean timestep
            n_steps_fast: number of atmosphere timesteps per ocean timestep
        """
        self.ocean = ocean
        self.atmosphere = atmosphere
        atmosphere_timestep = datetime.timedelta(
            seconds=ocean_timestep.total_seconds() / n_steps_fast
        )
        for ts, ds in [(ocean_timestep, ocean), (atmosphere_timestep, atmosphere)]:
            try:
                timestep = ds.timestep
            except ValueError as e:
                raise ValueError(
                    f"{str(e)} Timesteps must be inferred for CoupledDataset."
                )
            if timestep != ts:
                raise ValueError(
                    "Loaded dataset had an unexpected timestep. Expected "
                    f"{ts} but got {timestep}."
                )

        self._n_steps_fast = n_steps_fast

        metadata: Dict[str, VariableMetadata] = {}
        for ds in [ocean, atmosphere]:
            metadata.update(ds.variable_metadata)
        self._is_remote = any(ds.is_remote for ds in [ocean, atmosphere])
        self._variable_metadata = metadata
        self._sigma_coordinates = atmosphere.sigma_coordinates
        self._horizontal_coordinates = atmosphere.horizontal_coordinates

    @property
    def variable_metadata(self) -> Mapping[str, VariableMetadata]:
        return self._variable_metadata

    @property
    def sigma_coordinates(self) -> SigmaCoordinates:
        return self._sigma_coordinates

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._horizontal_coordinates

    @property
    def is_remote(self) -> bool:
        return self._is_remote

    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> Tuple[TensorDict, xr.DataArray]:
        raise NotImplementedError(
            "CoupledDataset has no get_sample_by_time_slice method"
        )

    def __len__(self):
        return min([len(self.ocean), len(self.atmosphere)])

    def __getitem__(self, idx: int) -> CoupledDatasetItem:
        fast_idx = idx * self._n_steps_fast
        ocean = self.ocean[idx]
        atmosphere = self.atmosphere[fast_idx]
        assert atmosphere[1].isel(time=0) == ocean[1].isel(time=0)
        return CoupledDatasetItem(ocean=ocean, atmosphere=atmosphere)
