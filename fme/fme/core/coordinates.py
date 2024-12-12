import abc
import dataclasses
from typing import List, Literal, Mapping, Optional, Tuple, TypeVar

import numpy as np
import torch
from astropy_healpix import HEALPix

from fme.core import metrics
from fme.core.constants import GRAVITY
from fme.core.gridded_ops import GriddedOperations, HEALPixOperations, LatLonOperations
from fme.core.typing_ import TensorMapping
from fme.core.winds import lon_lat_to_xyz

HC = TypeVar("HC", bound="HorizontalCoordinates")


@dataclasses.dataclass
class HybridSigmaPressureCoordinate:
    """
    Defines pressure at interface levels according to the following formula:
        p(k) = a(k) + b(k)*ps.

    where ps is the surface pressure, a and b are the sigma-pressure coordinates.

    Parameters:
        ak: a(k) coefficients as a 1-dimensional tensor
        bk: b(k) coefficients as a 1-dimensional tensor
    """

    ak: torch.Tensor
    bk: torch.Tensor

    def __post_init__(self):
        if len(self.ak.shape) != 1:
            raise ValueError(
                f"ak must be a 1-dimensional tensor. Got shape: {self.ak.shape}"
            )
        if len(self.bk.shape) != 1:
            raise ValueError(
                f"bk must be a 1-dimensional tensor. Got shape: {self.bk.shape}"
            )
        if len(self.ak) != len(self.bk):
            raise ValueError(
                f"ak and bk must have the same length. Got len(ak)={len(self.ak)} and "
                f"len(bk)={len(self.bk)}."
            )

    def __len__(self):
        """The number of vertical layer interfaces."""
        return len(self.ak)

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {"ak": self.ak.cpu().numpy(), "bk": self.bk.cpu().numpy()}

    def to(self, device: str) -> "HybridSigmaPressureCoordinate":
        return HybridSigmaPressureCoordinate(
            ak=self.ak.to(device),
            bk=self.bk.to(device),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, HybridSigmaPressureCoordinate):
            return False
        return torch.allclose(self.ak, other.ak) and torch.allclose(self.bk, other.bk)

    def as_dict(self) -> TensorMapping:
        return {"ak": self.ak, "bk": self.bk}

    def interface_pressure(self, surface_pressure: torch.Tensor) -> torch.Tensor:
        """
        Compute pressure at vertical layer interfaces.

        Args:
            surface_pressure: The surface pressure in units of Pa.

        Returns:
            A tensor of pressure at vertical layer interfaces. Will contain a new
            dimension at the end, representing the vertical.
        """
        return torch.stack(
            [ak + bk * surface_pressure for ak, bk in zip(self.ak, self.bk)],
            dim=-1,
        )

    def vertical_integral(
        self, integrand: torch.Tensor, surface_pressure: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the mass-weighted vertical integral of the integrand.

        (1 / g) * ∫ x dp

        where
        - g = acceleration due to gravity
        - x = integrad
        - p = pressure level

        Args:
            surface_pressure: The surface pressure in units of Pa.
            integrand: A tensor whose last dimension is the vertical.

        Returns:
            A tensor of same shape as integrand but without the last dimension.
        """
        if len(self.ak) != integrand.shape[-1] + 1:
            raise ValueError(
                "The last dimension of integrand must match the number of vertical "
                "layers in the hybrid sigma-pressure vertical coordinate."
            )
        interface_pressure = self.interface_pressure(surface_pressure)
        pressure_thickness = interface_pressure.diff(dim=-1)
        return (integrand * pressure_thickness).sum(dim=-1) / GRAVITY


@dataclasses.dataclass
class DimSize:
    name: str
    size: int


class HorizontalCoordinates(abc.ABC):
    """
    Parent class for horizontal coordinate system grids.
    Contains coords which must be subclassed to provide the coordinates.
    """

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abc.abstractmethod
    def to(self: HC, device: str) -> HC:
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
        """Names of model horizontal dimensions."""
        pass

    @property
    @abc.abstractmethod
    def loaded_dims(self) -> List[str]:
        """Names of horizontal dimensions as loaded from training dataset."""
        pass

    @property
    @abc.abstractmethod
    def loaded_sizes(self) -> List[DimSize]:
        """Sizes of horizontal dimensions as loaded from training dataset."""
        pass

    @property
    @abc.abstractmethod
    def loaded_default_sizes(self) -> List[DimSize]:
        """Default sizes of horizontal data dimensions, used by testing code."""
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
        """Meshgrids of latitudes and longitudes, respectively."""
        pass


@dataclasses.dataclass
class LatLonCoordinates(HorizontalCoordinates):
    """
    Defines a (latitude, longitude) grid.

    Parameters:
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
        self._area_weights: Optional[torch.Tensor] = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, LatLonCoordinates):
            return False
        return (
            torch.allclose(self.lat, other.lat)
            and torch.allclose(self.lon, other.lon)
            and self.loaded_lat_name == other.loaded_lat_name
            and self.loaded_lon_name == other.loaded_lon_name
        )

    def to(self, device: str) -> "LatLonCoordinates":
        return LatLonCoordinates(
            lon=self.lon.to(device),
            lat=self.lat.to(device),
            loaded_lat_name=self.loaded_lat_name,
            loaded_lon_name=self.loaded_lon_name,
        )

    @property
    def area_weights(self) -> torch.Tensor:
        if self._area_weights is None:
            self._area_weights = metrics.spherical_area_weights(self.lat, len(self.lon))
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

    Parameters:
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

    def to(self, device: str) -> "HEALPixCoordinates":
        return HEALPixCoordinates(
            face=self.face.to(device),
            height=self.height.to(device),
            width=self.width.to(device),
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
