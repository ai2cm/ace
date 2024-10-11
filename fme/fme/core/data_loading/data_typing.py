import abc
import dataclasses
import datetime
import logging
from collections import namedtuple
from typing import (
    Any,
    Generic,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

import numpy as np
import torch
import xarray as xr
from astropy_healpix import HEALPix

from fme.core import metrics
from fme.core.data_loading.utils import BatchData
from fme.core.gridded_ops import GriddedOperations, HEALPixOperations, LatLonOperations
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.winds import lon_lat_to_xyz

VariableMetadata = namedtuple("VariableMetadata", ["units", "long_name"])


@dataclasses.dataclass
class DimSize:
    name: str
    size: int


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
class HorizontalCoordinates(abc.ABC):
    """
    Parent class for horizontal coordinate system grids.
    Contains coords which must be subclassed to provide the coordinates.
    """

    @property
    @abc.abstractmethod
    def coords(self) -> Mapping[str, np.ndarray]:
        pass

    @property
    @abc.abstractmethod
    def xyz(self) -> Tuple[float, float, float]:
        pass

    @property
    @abc.abstractmethod
    def dims(self) -> List[str]:
        """names of model horizontal dimensions"""
        pass

    @property
    @abc.abstractmethod
    def loaded_dims(self) -> List[str]:
        """names of horizontal dimensions as loaded from training dataset"""
        pass

    @property
    @abc.abstractmethod
    def loaded_sizes(self) -> List[DimSize]:
        """sizes of horizontal dimensions as loaded from training dataset"""
        pass

    @property
    @abc.abstractmethod
    def loaded_default_sizes(self) -> List[DimSize]:
        """default sizes of horizontal data dimensions, used by testing code"""
        pass

    @property
    @abc.abstractmethod
    def grid(self) -> Literal["equiangular", "legendre-gauss", "healpix"]:
        pass

    # A temporary solution for training which allows us to aggregate along the
    # latitude dimension.
    # TODO: https://github.com/ai2cm/full-model/issues/1003
    @abc.abstractmethod
    def get_lat(self) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def area_weights(self) -> Optional[torch.Tensor]:
        pass

    @property
    @abc.abstractmethod
    def gridded_operations(self) -> GriddedOperations:
        pass

    @property
    @abc.abstractmethod
    def meshgrid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """meshgrids of latitudes and longitudes, respectively."""
        pass


@dataclasses.dataclass
class LatLonCoordinates(HorizontalCoordinates):
    """
    Defines a (latitude, longitude) grid.

    Attributes:
        lat: 1-dimensional tensor of latitudes
        lon: 1-dimensional tensor of longitudes
        loaded_lat_name: name of the latitude dimension
            as loaded from training dataset
        loaded_lon_name: name of the longitude dimension
            as loaded from training dataset
    """

    lon: torch.Tensor
    lat: torch.Tensor
    loaded_lat_name: str = "lat"
    loaded_lon_name: str = "lon"

    def __post_init__(self):
        self._area_weights = metrics.spherical_area_weights(self.lat, len(self.lon))

    @property
    def area_weights(self) -> torch.Tensor:
        return self._area_weights

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        # TODO: Replace with lat/lon name?
        return {
            "lat": self.lat.cpu().type(torch.float32).numpy(),
            "lon": self.lon.cpu().type(torch.float32).numpy(),
        }

    @property
    def xyz(self) -> Tuple[float, float, float]:
        lats, lons = np.broadcast_arrays(
            self.coords["lat"][:, None], self.coords["lon"][None, :]
        )
        return lon_lat_to_xyz(lons, lats)

    def get_lat(self) -> torch.Tensor:
        return self.lat

    @property
    def dims(self) -> List[str]:
        return ["lat", "lon"]

    @property
    def loaded_dims(self) -> List[str]:
        return [self.loaded_lat_name, self.loaded_lon_name]

    @property
    def loaded_sizes(self) -> List[DimSize]:
        return [
            DimSize(self.loaded_lat_name, len(self.lat)),
            DimSize(self.loaded_lon_name, len(self.lon)),
        ]

    @property
    def loaded_default_sizes(self) -> List[DimSize]:
        return [DimSize(self.loaded_lat_name, 16), DimSize(self.loaded_lon_name, 32)]

    @property
    def grid(self) -> Literal["equiangular", "legendre-gauss"]:
        if torch.allclose(
            self.lat[1:] - self.lat[:-1],
            self.lat[1] - self.lat[0],
        ):
            return "equiangular"
        else:
            return "legendre-gauss"

    @property
    def gridded_operations(self) -> LatLonOperations:
        return LatLonOperations(self.area_weights)

    @property
    def meshgrid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.meshgrid(self.lat, self.lon, indexing="ij")


@dataclasses.dataclass
class HEALPixCoordinates(HorizontalCoordinates):
    """
    Defines a HEALPix (face, height, width) grid. See https://healpix.jpl.nasa.gov/ for
    more information.

    Attributes:
        face: 1-dimensional tensor of faces
        height: 1-dimensional tensor of heights
        width: 1-dimensional tensor of widths
    """

    face: torch.Tensor
    height: torch.Tensor
    width: torch.Tensor

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {
            "face": self.face.cpu().type(torch.float32).numpy(),
            "height": self.height.cpu().type(torch.float32).numpy(),
            "width": self.width.cpu().type(torch.float32).numpy(),
        }

    @property
    def xyz(self) -> Tuple[float, float, float]:
        hp = HEALPix(nside=len(self.height), order="ring")
        return hp.healpix_to_xyz(
            [self.coords["face"], self.coords["height"], self.coords["width"]]
        )

    @property
    def dims(self) -> List[str]:
        return ["face", "height", "width"]

    @property
    def loaded_dims(self) -> List[str]:
        return self.dims

    @property
    def loaded_sizes(self) -> List[DimSize]:
        return [
            DimSize("face", len(self.face)),
            DimSize("height", len(self.width)),
            DimSize("width", len(self.height)),
        ]

    @property
    def loaded_default_sizes(cls) -> List[DimSize]:
        return [
            DimSize("face", 12),
            DimSize("height", 16),
            DimSize("width", 16),
        ]

    # TODO: https://github.com/ai2cm/full-model/issues/1003
    # This is currently the dummy solution.
    def get_lat(self) -> torch.Tensor:
        raise NotImplementedError(
            "healpix does not support get_lat. If latitude is needed \
            for some reason, you may use this class's self.xyz property to derive it."
        )

    @property
    def grid(self) -> Literal["healpix"]:
        return "healpix"

    @property
    def area_weights(self) -> Literal[None]:
        return None

    @property
    def gridded_operations(self) -> HEALPixOperations:
        return HEALPixOperations()

    @property
    def meshgrid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "meshgrid is not implemented yet for HEALPixCoordinates."
        )


class Dataset(torch.utils.data.Dataset, abc.ABC):
    # List of properties to transfer by default from
    # dataset if using torch Concat or Subset
    BASE_PROPERTIES: List[str] = [
        "metadata",
        "horizontal_coordinates",
        "sigma_coordinates",
        "is_remote",
    ]

    @property
    @abc.abstractmethod
    def metadata(self) -> Mapping[str, VariableMetadata]:
        ...

    @property
    @abc.abstractmethod
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        ...

    @property
    @abc.abstractmethod
    def sigma_coordinates(self) -> SigmaCoordinates:
        ...

    @property
    @abc.abstractmethod
    def is_remote(self) -> bool:
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


T = TypeVar("T", covariant=True)


class DataLoader(Protocol, Generic[T]):
    def __iter__(self) -> Iterator[T]:
        ...


class GriddedDataABC(abc.ABC, Generic[T]):
    @property
    @abc.abstractmethod
    def loader(self) -> DataLoader[T]:
        ...

    @property
    @abc.abstractmethod
    def sigma_coordinates(self) -> SigmaCoordinates:
        ...

    @property
    @abc.abstractmethod
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        ...

    @property
    @abc.abstractmethod
    def timestep(self) -> datetime.timedelta:
        ...

    @property
    @abc.abstractmethod
    def gridded_operations(self) -> GriddedOperations:
        ...

    @property
    @abc.abstractmethod
    def n_samples(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def n_batches(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def n_forward_steps(self) -> int:
        ...

    @abc.abstractmethod
    def set_epoch(self, epoch: int):
        ...

    @abc.abstractmethod
    def log_info(self, name: str):
        """
        Report information about the data using logging.info.
        """
        ...


class GriddedData(GriddedDataABC[BatchData]):
    """
    Data as required for pytorch training.

    The data is assumed to be gridded, and attributes are included for
    performing operations on gridded data.
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        metadata: Mapping[str, VariableMetadata],
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
            metadata: Metadata for each variable.
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
        self._metadata = metadata
        self._sigma_coordinates = sigma_coordinates
        self._horizontal_coordinates = horizontal_coordinates
        self._timestep = timestep
        self._sampler = sampler
        self._batch_size: Optional[int] = None

    @property
    def loader(self) -> DataLoader[BatchData]:
        return self._loader

    @property
    def metadata(self) -> Mapping[str, VariableMetadata]:
        return self._metadata

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
        return len(self._loader.dataset)

    @property
    def n_batches(self) -> int:
        return len(self._loader)

    @property
    def _first_time(self) -> Any:
        return self._loader.dataset[0][1].values[0]

    @property
    def _last_time(self) -> Any:
        return self._loader.dataset[-1][1].values[0]

    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            example_data = next(iter(self.loader)).data
            example_tensor = next(iter(example_data.values()))
            self._batch_size = example_tensor.shape[0]
        return self._batch_size

    @property
    def n_forward_steps(self) -> int:
        return self._loader.dataset.n_forward_steps

    def log_info(self, name: str):
        logging.info(
            f"{name} data: {self.n_samples} samples, " f"{self.n_batches} batches"
        )
        logging.info(f"{name} data: first sample's initial time: {self._first_time}")
        logging.info(f"{name} data: last sample's initial time: {self._last_time}")

    def set_epoch(self, epoch: int):
        """
        Set the epoch for the data loader sampler, if it is a distributed sampler.
        """
        if self._sampler is not None and isinstance(
            self._sampler, torch.utils.data.DistributedSampler
        ):
            self._sampler.set_epoch(epoch)
