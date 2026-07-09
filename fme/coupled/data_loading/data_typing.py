import dataclasses
import datetime
from typing import cast

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

_VertCoordType = OptionalDepthCoordinate | OptionalHybridSigmaPressureCoordinate


@dataclasses.dataclass
class CoupledCoords:
    """
    Convenience wrapper for the coords in dict format.
    """

    ocean_vertical: dict[str, np.ndarray] | None = None
    ice_vertical: dict[str, np.ndarray] | None = None
    atmosphere_vertical: dict[str, np.ndarray] | None = None
    ocean_horizontal: dict[str, np.ndarray] | None = None
    ice_horizontal: dict[str, np.ndarray] | None = None
    atmosphere_horizontal: dict[str, np.ndarray] | None = None

    @property
    def ocean(self) -> dict[str, np.ndarray]:
        if self.ocean_horizontal is not None:
            return {**(self.ocean_vertical or {}), **self.ocean_horizontal}
        raise AttributeError("Ocean component is None")

    @property
    def ice(self) -> dict[str, np.ndarray]:
        if self.ice_horizontal is not None:
            return {**(self.ice_vertical or {}), **self.ice_horizontal}
        raise AttributeError("Ice component is None")

    @property
    def atmosphere(self) -> dict[str, np.ndarray]:
        if self.atmosphere_horizontal is not None:
            return {**(self.atmosphere_vertical or {}), **self.atmosphere_horizontal}
        raise AttributeError("Atmosphere component is None")


class CoupledVerticalCoordinate:
    def __init__(
        self,
        ocean: OptionalDepthCoordinate | None = None,
        ice: OptionalDepthCoordinate | None = None,
        atmosphere: OptionalHybridSigmaPressureCoordinate | None = None,
    ):
        self._components: dict[str, _VertCoordType] = {
            name: val
            for name, val in [
                ("ocean", ocean),
                ("ice", ice),
                ("atmosphere", atmosphere),
            ]
            if val is not None
        }

    @classmethod
    def from_components(
        cls,
        components: dict[str, _VertCoordType],
    ) -> "CoupledVerticalCoordinate":
        obj = cls()
        obj._components = components
        return obj

    @property
    def ocean(self) -> OptionalDepthCoordinate | None:
        return cast(OptionalDepthCoordinate | None, self._components.get("ocean"))

    @property
    def ice(self) -> OptionalDepthCoordinate | None:
        return cast(OptionalDepthCoordinate | None, self._components.get("ice"))

    @property
    def atmosphere(self) -> OptionalHybridSigmaPressureCoordinate | None:
        return cast(
            OptionalHybridSigmaPressureCoordinate | None,
            self._components.get("atmosphere"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoupledVerticalCoordinate):
            return False
        return self._components.keys() == other._components.keys() and all(
            self._components[k] == other._components[k] for k in self._components
        )

    def to(self, device: torch.device) -> "CoupledVerticalCoordinate":
        return CoupledVerticalCoordinate.from_components(
            {k: v.to(device) for k, v in self._components.items()}
        )


class CoupledHorizontalCoordinates:
    def __init__(
        self,
        ocean: HorizontalCoordinates | None = None,
        ice: HorizontalCoordinates | None = None,
        atmosphere: HorizontalCoordinates | None = None,
    ):
        self._components: dict[str, HorizontalCoordinates] = {
            name: val
            for name, val in [
                ("ocean", ocean),
                ("ice", ice),
                ("atmosphere", atmosphere),
            ]
            if val is not None
        }

    @property
    def ocean(self) -> HorizontalCoordinates | None:
        return self._components.get("ocean")

    @property
    def ice(self) -> HorizontalCoordinates | None:
        return self._components.get("ice")

    @property
    def atmosphere(self) -> HorizontalCoordinates | None:
        return self._components.get("atmosphere")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoupledHorizontalCoordinates):
            return False
        return self._components.keys() == other._components.keys() and all(
            self._components[k] == other._components[k] for k in self._components
        )

    def to(self, device: torch.device) -> "CoupledHorizontalCoordinates":
        return CoupledHorizontalCoordinates(
            **{k: v.to(device) for k, v in self._components.items()}
        )


class CoupledDatasetProperties:
    def __init__(
        self,
        ocean: DatasetProperties | None = None,
        ice: DatasetProperties | None = None,
        atmosphere: DatasetProperties | None = None,
    ):
        self._components: dict[str, DatasetProperties] = {
            name: val
            for name, val in [
                ("ocean", ocean),
                ("ice", ice),
                ("atmosphere", atmosphere),
            ]
            if val is not None
        }
        vcoords: dict[str, _VertCoordType] = {}
        for name, props in self._components.items():
            if name == "atmosphere":
                vcoords[name] = cast(
                    OptionalHybridSigmaPressureCoordinate, props.vertical_coordinate
                )
            else:
                vcoords[name] = cast(OptionalDepthCoordinate, props.vertical_coordinate)
        self._vertical_coordinate = CoupledVerticalCoordinate.from_components(vcoords)
        self._horizontal_coordinates = CoupledHorizontalCoordinates(
            **{
                name: props.horizontal_coordinates
                for name, props in self._components.items()
            }
        )

    @property
    def ocean(self) -> DatasetProperties | None:
        return self._components.get("ocean")

    @property
    def ice(self) -> DatasetProperties | None:
        return self._components.get("ice")

    @property
    def atmosphere(self) -> DatasetProperties | None:
        return self._components.get("atmosphere")

    @property
    def atmosphere_timestep(self) -> datetime.timedelta:
        props = self._components["atmosphere"]
        assert props.timestep is not None
        return props.timestep

    @property
    def ocean_timestep(self) -> datetime.timedelta:
        props = self._components["ocean"]
        assert props.timestep is not None
        return props.timestep

    @property
    def ice_timestep(self) -> datetime.timedelta:
        props = self._components["ice"]
        assert props.timestep is not None
        return props.timestep

    @property
    def vertical_coordinate(self) -> CoupledVerticalCoordinate:
        return self._vertical_coordinate

    @property
    def horizontal_coordinates(self) -> CoupledHorizontalCoordinates:
        return self._horizontal_coordinates

    @property
    def variable_metadata(self) -> dict[str, VariableMetadata]:
        metadata: dict[str, VariableMetadata] = {}
        for props in self._components.values():
            metadata.update(props.variable_metadata)
        return metadata

    @property
    def timestep(self) -> datetime.timedelta:
        for name in ("ocean", "ice"):
            if name in self._components:
                props = self._components[name]
                assert props.timestep is not None
                return props.timestep
        return self.atmosphere_timestep

    @property
    def is_remote(self) -> bool:
        return any(props.is_remote for props in self._components.values())

    @property
    def n_inner_steps(self) -> int:
        if self.ocean is not None:
            if self.atmosphere is not None:
                return self.ocean_timestep // self.atmosphere_timestep
            else:
                assert self.ice is not None
                return self.ocean_timestep // self.ice_timestep
        else:
            assert self.ice is not None and self.atmosphere is not None
            return self.ice_timestep // self.atmosphere_timestep

    @property
    def coords(self) -> CoupledCoords:
        kwargs: dict[str, dict[str, np.ndarray]] = {}
        for name in self._components:
            vcoord = self._vertical_coordinate._components.get(name)
            hcoord = self._horizontal_coordinates._components.get(name)
            assert vcoord is not None and hcoord is not None
            kwargs[f"{name}_vertical"] = vcoord.coords
            kwargs[f"{name}_horizontal"] = dict(hcoord.coords)
        return CoupledCoords(**kwargs)

    def to_device(self) -> "CoupledDatasetProperties":
        return CoupledDatasetProperties(
            **{name: props.to_device() for name, props in self._components.items()}
        )

    def update(self, other: "CoupledDatasetProperties"):
        if self._vertical_coordinate != other._vertical_coordinate:
            raise ValueError("Vertical coordinates must be the same for both datasets.")
        if self._horizontal_coordinates != other._horizontal_coordinates:
            raise ValueError(
                "Horizontal coordinates must be the same for both datasets."
            )
        for name, props in self._components.items():
            assert name in other._components
            props.update(other._components[name])


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
            ocean: ocean dataset (slow component, idx at the coarse timestep).
            ice: ice dataset (fast component, idx at the fine timestep).
            atmosphere: atmosphere dataset (fast component, idx at the fine timestep).
            properties: the coupled dataset properties.
            n_steps_fast: number of fast timesteps per slow (ocean) timestep.
        """
        self._slow_datasets: dict[str, DatasetABC] = {
            name: ds for name, ds in [("ocean", ocean)] if ds is not None
        }
        self._fast_datasets: dict[str, DatasetABC] = {
            name: ds
            for name, ds in [("ice", ice), ("atmosphere", atmosphere)]
            if ds is not None
        }
        self._validate_timesteps(properties, n_steps_fast)
        self._validate_time_alignment()
        self._properties = properties
        self._n_steps_fast = n_steps_fast

    def _validate_timesteps(
        self, properties: CoupledDatasetProperties, n_steps_fast: int
    ) -> None:
        if not self._slow_datasets:
            # No slow component: all fast components must share the same timestep.
            fast_names = list(self._fast_datasets)
            if len(fast_names) > 1:
                fast_ts_0 = properties._components[fast_names[0]].timestep
                for fast_name in fast_names[1:]:
                    other_ts = properties._components[fast_name].timestep
                    if other_ts != fast_ts_0:
                        raise ValueError(
                            f"{fast_names[0].capitalize()} and {fast_name} timesteps "
                            f"must be consistent, got {fast_ts_0} and {other_ts}."
                        )
            return
        if not self._fast_datasets:
            return
        slow_name = next(iter(self._slow_datasets))
        slow_ts = properties._components[slow_name].timestep
        fast_names = list(self._fast_datasets)
        fast_ts = properties._components[fast_names[0]].timestep
        for fast_name in fast_names[1:]:
            other_ts = properties._components[fast_name].timestep
            if other_ts != fast_ts:
                raise ValueError(
                    f"Fast components {fast_names[0]} and {fast_name} must have "
                    f"the same timestep, got {fast_ts} and {other_ts}."
                )
        assert slow_ts is not None and fast_ts is not None
        if slow_ts != fast_ts * n_steps_fast:
            fast_names_str = " and ".join(fast_names)
            raise ValueError(
                f"{slow_name.capitalize()} and {fast_names_str} timesteps must be "
                f"consistent with n_steps_fast, got {slow_name} timestep {slow_ts} "
                f"and fast timestep {fast_ts} with n_steps_fast={n_steps_fast}."
            )

    def _validate_time_alignment(self) -> None:
        all_datasets = {**self._slow_datasets, **self._fast_datasets}
        if len(all_datasets) < 2:
            return
        times = {
            name: ds[0][1].isel(time=0).item() for name, ds in all_datasets.items()
        }
        reference_name, reference_time = next(iter(times.items()))
        for name, t in times.items():
            if name != reference_name and t != reference_time:
                raise ValueError(
                    f"First time of {reference_name} dataset is {reference_time} "
                    f"but the {name}'s first time is {t}. "
                    "Maybe align the datasets using a subset?"
                )

    @property
    def properties(self) -> CoupledDatasetProperties:
        return self._properties

    @property
    def n_steps_fast(self) -> int:
        return self._n_steps_fast

    @property
    def all_ic_times(self) -> xr.CFTimeIndex:
        for ds in {**self._slow_datasets, **self._fast_datasets}.values():
            return ds.sample_start_times
        raise ValueError("No datasets available")

    def __len__(self) -> int:
        return min(
            len(ds) for ds in {**self._slow_datasets, **self._fast_datasets}.values()
        )

    def __getitem__(self, idx: int) -> CoupledDatasetItem:
        fast_idx = idx * self._n_steps_fast
        components: dict[str, DatasetItem] = {}
        for name, ds in self._slow_datasets.items():
            components[name] = ds[idx]
        for name, ds in self._fast_datasets.items():
            components[name] = ds[fast_idx]
        return CoupledDatasetItem(**components)

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        for name, ds in self._slow_datasets.items():
            try:
                ds.validate_inference_length(max_start_index, max_window_len)
            except ValueError as e:
                raise ValueError(
                    f"The {name} dataset has an insufficient number of timepoints."
                ) from e
        fast_max_start = max_start_index * self._n_steps_fast
        fast_max_window = (max_window_len - 1) * self._n_steps_fast + 1
        for name, ds in self._fast_datasets.items():
            try:
                ds.validate_inference_length(fast_max_start, fast_max_window)
            except ValueError as e:
                raise ValueError(
                    f"The {name} dataset has an insufficient number of timepoints."
                ) from e

    def set_epoch(self, epoch: int):
        for ds in {**self._slow_datasets, **self._fast_datasets}.values():
            ds.set_epoch(epoch)
