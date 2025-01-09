import dataclasses
import datetime
import logging
from typing import List, Literal, Mapping, Optional, Sequence

import numpy as np
import torch

from fme.ace.data_loading.batch_data import (
    BatchData,
    PairedData,
    PrognosticState,
    SizedMap,
)
from fme.core.coordinates import HorizontalCoordinates, HybridSigmaPressureCoordinate
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.generics.data import DataLoader, GriddedDataABC
from fme.core.gridded_ops import GriddedOperations
from fme.coupled.data_loading.data_typing import CoupledDatasetItem


class CoupledPrognosticState:
    """
    Thin typing wrapper around CoupledBatchData to indicate that the data is
    a prognostic state, such as an initial condition or final state when
    evolving forward in time.
    """

    def __init__(self, ocean_data: PrognosticState, atmosphere_data: PrognosticState):
        self.ocean_data = ocean_data
        self.atmosphere_data = atmosphere_data

    def to_device(self) -> "CoupledPrognosticState":
        return CoupledPrognosticState(
            self.ocean_data.to_device(), self.atmosphere_data.to_device()
        )

    def as_batch_data(self) -> "CoupledBatchData":
        return CoupledBatchData(
            self.ocean_data.as_batch_data(), self.atmosphere_data.as_batch_data()
        )


@dataclasses.dataclass
class CoupledBatchData:
    ocean_data: BatchData
    atmosphere_data: BatchData

    @classmethod
    def new_on_device(
        cls,
        ocean_data: BatchData,
        atmosphere_data: BatchData,
    ) -> "CoupledBatchData":
        return CoupledBatchData(ocean_data=ocean_data, atmosphere_data=atmosphere_data)

    @classmethod
    def new_on_cpu(
        cls,
        ocean_data: BatchData,
        atmosphere_data: BatchData,
    ) -> "CoupledBatchData":
        return CoupledBatchData(ocean_data=ocean_data, atmosphere_data=atmosphere_data)

    def to_device(self) -> "CoupledBatchData":
        return CoupledBatchData.new_on_device(
            ocean_data=self.ocean_data.to_device(),
            atmosphere_data=self.atmosphere_data.to_device(),
        )

    @classmethod
    def collate_fn(
        cls,
        samples: Sequence[CoupledDatasetItem],
        horizontal_dims: List[str],
        sample_dim_name: str = "sample",
    ) -> "CoupledBatchData":
        """
        Collate function for use with PyTorch DataLoader. Separates out ocean
        and atmosphere sample tuples and constructs BatchData instances for
        each of the two components.

        """
        ocean_data = BatchData.from_sample_tuples(
            [x.ocean for x in samples], sample_dim_name=sample_dim_name
        )
        atmosphere_data = BatchData.from_sample_tuples(
            [x.atmosphere for x in samples],
            horizontal_dims=horizontal_dims,
            sample_dim_name=sample_dim_name,
        )
        return CoupledBatchData.new_on_cpu(ocean_data, atmosphere_data)


@dataclasses.dataclass
class CoupledPairedData:
    """
    A container for the data and time coordinates of a batch, with paired
    prediction and target data.
    """

    ocean_data: PairedData
    atmosphere_data: PairedData


class CoupledGriddedData(GriddedDataABC[CoupledBatchData]):
    def __init__(
        self,
        loader: DataLoader[CoupledBatchData],
        variable_metadata: Mapping[str, VariableMetadata],
        vertical_coordinate: HybridSigmaPressureCoordinate,
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
        self._vertical_coordinates = vertical_coordinate
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
    def vertical_coordinate(self) -> HybridSigmaPressureCoordinate:
        return self._vertical_coordinates

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
