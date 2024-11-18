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
        atmosphere_timestep: datetime.timedelta,
    ):
        self.ocean = ocean
        self.atmosphere = atmosphere
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

        if atmosphere_timestep > ocean_timestep:
            raise ValueError(f"Atmosphere timestep must be no larger than the ocean's.")
        self.timestep = ocean_timestep

        n_steps_fast = ocean_timestep / atmosphere_timestep
        if n_steps_fast != int(n_steps_fast):
            raise ValueError(
                f"Expected atmosphere timestep {atmosphere_timestep} to be a multiple "
                f"of ocean timestep {ocean_timestep}."
            )
        self._n_steps_fast = int(n_steps_fast)

        # check for misconfigured DataRequirements n_timesteps in the atmosphere
        slow_n_steps = self.ocean.n_steps
        fast_n_steps = (slow_n_steps - 1) * self._n_steps_fast + 1
        if self.atmosphere.n_steps != fast_n_steps:
            raise ValueError(
                f"Atmosphere dataset timestep is {atmosphere_timestep} and "
                f"ocean dataset timestep is {ocean_timestep}, "
                f"so we need {self._n_steps_fast} atmosphere steps for each of the "
                f"{slow_n_steps - 1} ocean steps, giving {fast_n_steps} total "
                "timepoints (including IC) per sample, but atmosphere dataset "
                f"was configured to return {self.atmosphere.n_steps} steps."
            )

        metadata: Dict[str, VariableMetadata] = {}
        for ds in [ocean, atmosphere]:
            metadata.update(ds.variable_metadata)
        self._is_remote = any(ds.is_remote for ds in [ocean, atmosphere])
        self._variable_metadata = metadata
        self._sigma_coordinates = atmosphere.sigma_coordinates
        self._horizontal_coordinates = atmosphere.horizontal_coordinates

    @property
    def n_forward_steps(self) -> int:
        return self.ocean.n_steps - 1

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
