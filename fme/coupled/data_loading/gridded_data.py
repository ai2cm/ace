import datetime
import logging
from typing import Literal, Mapping, Optional, Union

import numpy as np
import torch

from fme.ace.data_loading.gridded_data import SizedMap
from fme.core.coordinates import HorizontalCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.generics.data import DataLoader, GriddedDataABC, InferenceDataABC
from fme.core.gridded_ops import GriddedOperations
from fme.coupled.data_loading.batch_data import CoupledBatchData, CoupledPrognosticState
from fme.coupled.data_loading.data_typing import (
    CoupledDatasetProperties,
    CoupledVerticalCoordinate,
)
from fme.coupled.requirements import CoupledPrognosticStateDataRequirements


class GriddedData(GriddedDataABC[CoupledBatchData]):
    def __init__(
        self,
        loader: DataLoader[CoupledBatchData],
        variable_metadata: Mapping[str, VariableMetadata],
        vertical_coordinate: CoupledVerticalCoordinate,
        horizontal_coordinates: HorizontalCoordinates,
        timestep: datetime.timedelta,
        sampler: Optional[torch.utils.data.Sampler] = None,
    ):
        """
        Args:
            loader: torch DataLoader, which returns batches of type
                TensorMapping where keys indicate variable name.
                Each tensor has shape
                [batch_size, face, time_window_size, n_channels, n_x_coord, n_y_coord].
            variable_metadata: Metadata for each variable.
            area_weights: Weights for each grid cell, used for computing area-weighted
                averages. Has shape [n_x_coord, n_y_coord].
            vertical_coordinate: Vertical coordinate for each grid cell, used for
                computing pressure levels.
            horizontal_coordinates: horizontal coordinates for the data.
            timestep: Timestep of the model.
            sampler: Optional sampler for the data loader. Provided to allow support for
                distributed training.
        """
        self._loader = loader
        self._variable_metadata = variable_metadata
        self._vertical_coordinate = vertical_coordinate
        self._horizontal_coordinates = horizontal_coordinates
        self._timestep = timestep
        self._sampler = sampler
        self._batch_size: Optional[int] = None

    @property
    def loader(self) -> DataLoader[CoupledBatchData]:
        def to_device(x: CoupledBatchData) -> CoupledBatchData:
            return x.to_device()

        return SizedMap(to_device, self._loader)

    @property
    def variable_metadata(self) -> Mapping[str, VariableMetadata]:
        return self._variable_metadata

    @property
    def vertical_coordinate(self) -> CoupledVerticalCoordinate:
        return self._vertical_coordinate

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._horizontal_coordinates

    @property
    def timestep(self) -> datetime.timedelta:
        return self._timestep

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {
            **self.horizontal_coordinates.coords,
            **self.vertical_coordinate.coords,
        }

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
        initial_condition: Union[
            CoupledPrognosticState, CoupledPrognosticStateDataRequirements
        ],
        properties: CoupledDatasetProperties,
    ):
        self._loader = loader
        self._properties = properties.to_device()
        self._n_initial_conditions: Optional[int] = None
        if isinstance(initial_condition, CoupledPrognosticStateDataRequirements):
            self._initial_condition: CoupledPrognosticState = get_initial_condition(
                loader, initial_condition
            )
        else:
            self._initial_condition = initial_condition.to_device()

    @property
    def initial_condition(self) -> CoupledPrognosticState:
        return self._initial_condition

    @property
    def loader(self) -> DataLoader[CoupledBatchData]:
        def on_device(x: CoupledBatchData) -> CoupledBatchData:
            return x.to_device()

        return SizedMap(on_device, self._loader)

    @property
    def variable_metadata(self) -> Mapping[str, VariableMetadata]:
        return self._properties.variable_metadata
