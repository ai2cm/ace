import dataclasses
import datetime
import logging
from typing import Generic, List, Literal, Mapping, Optional, Sequence, TypeVar

import numpy as np
import torch

from fme.core.data_loading.batch_data import (
    CPU,
    AnyDevice,
    BatchData,
    CurrentDevice,
    GriddedDataABC,
    SizedMap,
)
from fme.core.data_loading.data_typing import (
    HorizontalCoordinates,
    SigmaCoordinates,
    VariableMetadata,
)
from fme.core.generics.data import DataLoader
from fme.core.gridded_ops import GriddedOperations
from fme.coupled.data_loading.data_typing import CoupledDatasetItem

DeviceType = TypeVar("DeviceType", bound=AnyDevice)


@dataclasses.dataclass
class CoupledBatchData(Generic[DeviceType]):
    ocean_data: BatchData[DeviceType]
    atmosphere_data: BatchData[DeviceType]

    @classmethod
    def new_on_device(
        cls,
        ocean_data: BatchData[CurrentDevice],
        atmosphere_data: BatchData[CurrentDevice],
    ) -> "CoupledBatchData[CurrentDevice]":
        return CoupledBatchData[CurrentDevice](
            ocean_data=ocean_data, atmosphere_data=atmosphere_data
        )

    @classmethod
    def new_on_cpu(
        cls,
        ocean_data: BatchData[CPU],
        atmosphere_data: BatchData[CPU],
    ) -> "CoupledBatchData[CPU]":
        return CoupledBatchData[CPU](
            ocean_data=ocean_data, atmosphere_data=atmosphere_data
        )

    def to_device(self) -> "CoupledBatchData[CurrentDevice]":
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
    ) -> "CoupledBatchData[CPU]":
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


class CoupledGriddedData(GriddedDataABC[CoupledBatchData]):
    def __init__(
        self,
        loader: DataLoader[CoupledBatchData[CPU]],
        variable_metadata: Mapping[str, VariableMetadata],
        sigma_coordinates: SigmaCoordinates,
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
            sigma_coordinates: Sigma coordinates for each grid cell, used for computing
                pressure levels.
            horizontal_coordinates: horizontal coordinates for the data.
            timestep: Timestep of the model.
            sampler: Optional sampler for the data loader. Provided to allow support for
                distributed training.
        """
        self._loader = loader
        self._variable_metadata = variable_metadata
        self._sigma_coordinates = sigma_coordinates
        self._horizontal_coordinates = horizontal_coordinates
        self._timestep = timestep
        self._sampler = sampler
        self._batch_size: Optional[int] = None

    @property
    def loader(self) -> DataLoader[CoupledBatchData[CurrentDevice]]:
        def to_device(x: CoupledBatchData[CPU]) -> CoupledBatchData[CurrentDevice]:
            return x.to_device()

        return SizedMap(to_device, self._loader)

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
    def timestep(self) -> datetime.timedelta:
        return self._timestep

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {
            **self.horizontal_coordinates.coords,
            **self.sigma_coordinates.coords,
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
