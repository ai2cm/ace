import abc
import dataclasses
from collections import namedtuple
from typing import List, Literal, Mapping, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from astropy_healpix import HEALPix

from fme.core import metrics
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, SigmaCoordinates):
            return False
        return torch.allclose(self.ak, other.ak) and torch.allclose(self.bk, other.bk)

    def as_dict(self) -> TensorMapping:
        return {"ak": self.ak, "bk": self.bk}


class HorizontalCoordinates(abc.ABC):
    """
    Parent class for horizontal coordinate system grids.
    Contains coords which must be subclassed to provide the coordinates.
    """

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, LatLonCoordinates):
            return False
        return (
            torch.allclose(self.lat, other.lat)
            and torch.allclose(self.lon, other.lon)
            and self.loaded_lat_name == other.loaded_lat_name
            and self.loaded_lon_name == other.loaded_lon_name
        )

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, HEALPixCoordinates):
            return False
        return (
            torch.allclose(self.face, other.face)
            and torch.allclose(self.height, other.height)
            and torch.allclose(self.width, other.width)
        )

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
