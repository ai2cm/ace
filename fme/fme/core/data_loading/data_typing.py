import abc
import dataclasses
from collections import namedtuple
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import xarray as xr

from fme.core.typing_ import TensorDict, TensorMapping

VariableMetadata = namedtuple("VariableMetadata", ["units", "long_name"])


@dataclasses.dataclass
class SigmaCoordinates:
    """
    Defines pressure at interface levels according to the following formula:
        p(k) = a(k) + b(k)*ps

    where ps is the surface pressure, a and b are the sigma coordinates.

    Attributes:
        ak: a(k) coefficients as a 1-dimensional tensor
        bk: b(k) coefficients as a 1-dimensional tensor
    """

    ak: torch.Tensor
    bk: torch.Tensor

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {"ak": self.ak.cpu().numpy(), "bk": self.bk.cpu().numpy()}

    def to(self, device: str) -> "SigmaCoordinates":
        return SigmaCoordinates(
            ak=self.ak.to(device),
            bk=self.bk.to(device),
        )

    def as_dict(self) -> TensorMapping:
        return {"ak": self.ak, "bk": self.bk}


@dataclasses.dataclass
class HorizontalCoordinates:
    """
    Defines a (latitude, longitude) grid.

    Attributes:
        lat: 1-dimensional tensor of latitudes
        lon: 1-dimensional tensor of longitudes
    """

    lat: torch.Tensor
    lon: torch.Tensor

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {"lat": self.lat.cpu().numpy(), "lon": self.lon.cpu().numpy()}


class Dataset(torch.utils.data.Dataset, abc.ABC):
    @abc.abstractproperty
    def metadata(self) -> Mapping[str, VariableMetadata]:
        ...

    @abc.abstractproperty
    def area_weights(self) -> torch.Tensor:
        ...

    @abc.abstractproperty
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        ...

    @abc.abstractproperty
    def sigma_coordinates(self) -> SigmaCoordinates:
        ...

    @abc.abstractmethod
    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> Tuple[TensorDict, xr.DataArray]:
        """
        Returns a sample of data for the given time slice.

        Args:
            time_slice: The time slice to return data for.

        Returns:
            A tuple whose first item is a mapping from variable
            name to tensor of shape [n_time, n_lat, n_lon] and
            whose second item is a time coordinate array.
        """
        ...


@dataclasses.dataclass
class GriddedData:
    """
    Data as required for pytorch training.

    The data is assumed to be gridded, and attributes are included for
    performing operations on gridded data.

    Attributes:
        loader: torch DataLoader, which returns batches of type
            TensorMapping where keys indicate variable name.
            Each tensor has shape
            [batch_size, time_window_size, n_channels, n_lat, n_lon].
        metadata: Metadata for each variable.
        area_weights: Weights for each grid cell, used for computing area-weighted
            averages. Has shape [n_lat, n_lon].
        sigma_coordinates: Sigma coordinates for each grid cell, used for computing
            pressure levels.
        horizontal_coordinates: Lat/lon coordinates for the data.
        sampler: Optional sampler for the data loader. Provided to allow support for
            distributed training.
    """

    loader: torch.utils.data.DataLoader
    metadata: Mapping[str, VariableMetadata]
    area_weights: torch.Tensor
    sigma_coordinates: SigmaCoordinates
    horizontal_coordinates: HorizontalCoordinates
    sampler: Optional[torch.utils.data.Sampler] = None

    @property
    def dataset(self) -> Dataset:
        return self.loader.dataset

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {
            **self.horizontal_coordinates.coords,
            **self.sigma_coordinates.coords,
        }
