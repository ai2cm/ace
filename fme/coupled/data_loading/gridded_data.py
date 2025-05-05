import datetime
import logging
from typing import Literal

import torch

from fme.ace.data_loading.gridded_data import SizedMap
from fme.core.coordinates import HorizontalCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.properties import DatasetProperties
from fme.core.generics.data import DataLoader, GriddedDataABC, InferenceDataABC
from fme.core.gridded_ops import GriddedOperations
from fme.coupled.data_loading.batch_data import CoupledBatchData, CoupledPrognosticState
from fme.coupled.data_loading.data_typing import (
    CoupledCoords,
    CoupledDatasetProperties,
    CoupledVerticalCoordinate,
)
from fme.coupled.requirements import CoupledPrognosticStateDataRequirements


class GriddedData(GriddedDataABC[CoupledBatchData]):
    def __init__(
        self,
        loader: DataLoader[CoupledBatchData],
        properties: CoupledDatasetProperties,
        sampler: torch.utils.data.Sampler | None = None,
    ):
        """
        Args:
            loader: torch DataLoader, which returns batches of type
                TensorMapping where keys indicate variable name.
                Each tensor has shape
                [batch_size, face, time_window_size, n_channels, n_x_coord, n_y_coord].
            properties: Properties of the dataset, including variable metadata and
                coordinates.
            sampler: Optional sampler for the data loader. Provided to allow support for
                distributed training.
        """
        self._loader = loader
        self._properties = properties.to_device()
        self._sampler = sampler
        self._batch_size: int | None = None

    @property
    def loader(self) -> DataLoader[CoupledBatchData]:
        def to_device(x: CoupledBatchData) -> CoupledBatchData:
            return x.to_device()

        return SizedMap(to_device, self._loader)

    @property
    def variable_metadata(self) -> dict[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def vertical_coordinate(self) -> CoupledVerticalCoordinate:
        return self._properties.vertical_coordinate

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._properties.horizontal_coordinates

    @property
    def timestep(self) -> datetime.timedelta:
        return self._properties.timestep

    @property
    def coords(self) -> CoupledCoords:
        return CoupledCoords(
            ocean_vertical=self.vertical_coordinate.ocean.coords,
            atmosphere_vertical=self.vertical_coordinate.atmosphere.coords,
            horizontal=dict(self.horizontal_coordinates.coords),
        )

    @property
    def grid(self) -> Literal["equiangular", "legendre-gauss", "healpix"]:
        return self.horizontal_coordinates.grid

    @property
    def gridded_operations(self) -> GriddedOperations:
        return self.horizontal_coordinates.gridded_operations

    @property
    def n_samples(self) -> int:
        return len(self._loader.dataset)  # type: ignore

    @property
    def n_batches(self) -> int:
        return len(self._loader)  # type: ignore

    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            example_data = next(iter(self.loader)).ocean_data.data
            example_tensor = next(iter(example_data.values()))
            self._batch_size = example_tensor.shape[0]
        return self._batch_size

    def set_epoch(self, epoch: int):
        """
        Set the epoch for the data loader sampler, if it is a distributed sampler.
        """
        if self._sampler is not None and isinstance(
            self._sampler, torch.utils.data.DistributedSampler
        ):
            self._sampler.set_epoch(epoch)

    def log_info(self, name: str):
        logging.info(
            f"Dataset {name} has {self.n_samples} samples, {self.n_batches} batches, "
            f"batch size {self.batch_size}, timestep {self.timestep}."
        )


def get_initial_condition(
    loader: DataLoader[CoupledBatchData],
    requirements: CoupledPrognosticStateDataRequirements,
) -> CoupledPrognosticState:
    for batch in loader:
        return batch.get_start(requirements).to_device()
    raise ValueError("No initial condition found in loader")


class InferenceGriddedData(InferenceDataABC[CoupledPrognosticState, CoupledBatchData]):
    def __init__(
        self,
        loader: DataLoader[CoupledBatchData],
        initial_condition: CoupledPrognosticState
        | CoupledPrognosticStateDataRequirements,
        properties: CoupledDatasetProperties,
    ):
        self._loader = loader
        self._properties = properties.to_device()
        self._n_initial_conditions: int | None = None
        if isinstance(initial_condition, CoupledPrognosticStateDataRequirements):
            self._initial_condition: CoupledPrognosticState = get_initial_condition(
                loader, initial_condition
            )
        else:
            self._initial_condition = initial_condition.to_device()

    @property
    def atmosphere_properties(self) -> DatasetProperties:
        return self._properties.atmosphere

    @property
    def ocean_properties(self) -> DatasetProperties:
        return self._properties.ocean

    @property
    def initial_condition(self) -> CoupledPrognosticState:
        return self._initial_condition

    @property
    def loader(self) -> DataLoader[CoupledBatchData]:
        def on_device(x: CoupledBatchData) -> CoupledBatchData:
            return x.to_device()

        return SizedMap(on_device, self._loader)

    @property
    def variable_metadata(self) -> dict[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def ocean_timestep(self) -> datetime.timedelta:
        return self._properties.ocean.timestep

    @property
    def atmosphere_timestep(self) -> datetime.timedelta:
        return self._properties.atmosphere.timestep

    @property
    def n_inner_steps(self) -> int:
        return self._properties.n_inner_steps

    @property
    def vertical_coordinate(self) -> CoupledVerticalCoordinate:
        return self._properties.vertical_coordinate

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._properties.horizontal_coordinates

    @property
    def coords(self) -> CoupledCoords:
        return self._properties.coords
