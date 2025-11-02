import abc
import dataclasses
import math
import re
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Literal, TypeVar

import dacite
import numpy as np
import torch

try:
    from earth2grid import healpix as e2ghpx
except ImportError:
    e2ghpx = None

from fme.core import metrics
from fme.core.constants import GRAVITY
from fme.core.corrector.atmosphere import AtmosphereCorrector, AtmosphereCorrectorConfig
from fme.core.corrector.ocean import OceanCorrector, OceanCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.derived_variables import compute_derived_quantities
from fme.core.device import get_device
from fme.core.gridded_ops import GriddedOperations, HEALPixOperations, LatLonOperations
from fme.core.mask_provider import MaskProvider, MaskProviderABC, NullMaskProvider
from fme.core.masking import StaticMasking
from fme.core.ocean_derived_variables import compute_ocean_derived_quantities
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.winds import lon_lat_to_xyz

HC = TypeVar("HC", bound="HorizontalCoordinates")


class DeriveFnABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        pass


class NullDeriveFn(DeriveFnABC):
    def __call__(self, data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        return dict(data)


class AtmosphericDeriveFn(DeriveFnABC):
    def __init__(
        self,
        vertical_coordinate: "OptionalHybridSigmaPressureCoordinate",
        timestep: timedelta,
    ):
        self.vertical_coordinate = vertical_coordinate.to(
            "cpu"
        )  # must be on cpu for multiprocessing fork context
        self.timestep = timestep

    def __call__(self, data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        if isinstance(self.vertical_coordinate, NullVerticalCoordinate):
            vertical_coord: HybridSigmaPressureCoordinate | None = None
        else:
            vertical_coord = self.vertical_coordinate.to(get_device())
        return compute_derived_quantities(
            dict(data),
            vertical_coordinate=vertical_coord,
            timestep=self.timestep,
            forcing_data=dict(forcing_data),
        )


class OceanDeriveFn(DeriveFnABC):
    def __init__(
        self,
        depth_coordinate: "OptionalDepthCoordinate",
        timestep: timedelta,
    ):
        self.depth_coordinate = depth_coordinate.to(
            "cpu"
        )  # must be on cpu for multiprocessing fork context
        self.timestep = timestep

    def __call__(self, data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        if isinstance(self.depth_coordinate, NullVerticalCoordinate):
            depth_coord: DepthCoordinate | None = None
        else:
            depth_coord = self.depth_coordinate.to(get_device())
        return compute_ocean_derived_quantities(
            dict(data),
            depth_coordinate=depth_coord,
            timestep=self.timestep,
            forcing_data=dict(forcing_data),
        )


PostProcessFnType = Callable[[TensorMapping], TensorDict]


class NullPostProcessFn:
    def __call__(self, data: TensorMapping) -> TensorDict:
        return dict(data)


class VerticalCoordinate(abc.ABC):
    """
    A vertical coordinate system for use in the stepper.
    """

    SelfType = TypeVar("SelfType", bound="VerticalCoordinate")

    @abc.abstractmethod
    def to(self: SelfType, device: str) -> SelfType:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abc.abstractmethod
    def build_corrector(
        self,
        config: AtmosphereCorrectorConfig | CorrectorSelector,
        gridded_operations: GriddedOperations,
        timestep: timedelta,
    ) -> CorrectorABC:
        pass

    @abc.abstractmethod
    def build_derive_function(self, timestep: timedelta) -> DeriveFnABC:
        pass

    @property
    @abc.abstractmethod
    def coords(self) -> dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def as_dict(self) -> TensorMapping:
        pass


@dataclasses.dataclass
class HybridSigmaPressureCoordinate(VerticalCoordinate):
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
        if self.ak.dtype != self.bk.dtype:
            raise ValueError(
                f"ak and bk must have the same dtype. Got ak.dtype={self.ak.dtype} and "
                f"bk.dtype={self.bk.dtype}."
            )
        if self.ak.device != self.bk.device:
            raise ValueError(
                f"ak and bk must be on the same device. Got ak.device={self.ak.device} "
                f"and bk.device={self.bk.device}."
            )

    def __len__(self):
        """The number of vertical layer interfaces."""
        return len(self.ak)

    def build_corrector(
        self,
        config: AtmosphereCorrectorConfig | CorrectorSelector,
        gridded_operations: GriddedOperations,
        timestep: timedelta,
    ) -> AtmosphereCorrector:
        if (
            isinstance(config, CorrectorSelector)
            and config.type != "atmosphere_corrector"
        ):
            raise ValueError(
                f"Cannot build corrector for vertical coordinate {self} with "
                f"corrector selector {config}."
            )
        if isinstance(config, CorrectorSelector):
            config_instance = dacite.from_dict(
                data_class=AtmosphereCorrectorConfig,
                data=config.config,
                config=dacite.Config(strict=True),
            )
        else:
            config_instance = config
        return AtmosphereCorrector(
            config=config_instance,
            gridded_operations=gridded_operations,
            vertical_coordinate=self,
            timestep=timestep,
        )

    def build_derive_function(self, timestep: timedelta) -> DeriveFnABC:
        return AtmosphericDeriveFn(self, timestep)

    def get_ak(self) -> torch.Tensor:
        return self.ak

    def get_bk(self) -> torch.Tensor:
        return self.bk

    @property
    def coords(self) -> dict[str, np.ndarray]:
        return {"ak": self.ak.cpu().numpy(), "bk": self.bk.cpu().numpy()}

    @property
    def dtype(self) -> torch.dtype:
        return self.ak.dtype

    @property
    def device(self) -> str:
        return self.ak.device

    def to(self, device: str) -> "HybridSigmaPressureCoordinate":
        return HybridSigmaPressureCoordinate(
            ak=self.ak.to(device),
            bk=self.bk.to(device),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, HybridSigmaPressureCoordinate):
            return False
        try:
            torch.testing.assert_close(self.ak, other.ak)
            torch.testing.assert_close(self.bk, other.bk)
        except AssertionError:
            return False
        return True

    def __repr__(self) -> str:
        return f"HybridSigmaPressureCoordinate(\n    ak={self.ak},\n    bk={self.bk}\n)"

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


LEVEL_PATTERN = re.compile(r"_(\d+)$")


@dataclasses.dataclass
class DepthCoordinate(VerticalCoordinate):
    """
    Defines depth in meters at interface levels and accounts for a constant mask.

    Parameters:
        idepth: depth in meters at interface levels as a 1-dimensional tensor.
        mask: mask indicating valid vertical layers at each spatial point. Must be equal
            to 1 in valid points and 0 elsewhere. The last dimension is the vertical and
            it must be one shorter than idepth. The mask may have additional dimensions
            before the vertical, which are assumed to be broadcastable to match the
            integrand when computing integrals.
    """

    idepth: torch.Tensor
    mask: torch.Tensor
    surface_mask: torch.Tensor | None = None

    def __post_init__(self):
        if len(self.idepth.shape) != 1:
            raise ValueError(
                f"idepth must be a 1-dimensional tensor. Got shape: {self.idepth.shape}"
            )
        if len(self.idepth) < 2:
            raise ValueError(
                f"idepth must have at least two elements. Got {self.idepth}."
            )
        if self.idepth.shape[0] != self.mask.shape[-1] + 1:
            raise ValueError(
                "The last dimension of mask must be one shorter than length of idepth."
                f"Got idepth.shape: {self.idepth.shape} and mask.shape: "
                f"{self.mask.shape}."
            )

    def __len__(self):
        """The number of vertical layer interfaces."""
        return len(self.idepth)

    def get_mask(self) -> torch.Tensor:
        return self.mask

    def get_mask_level(self, level: int) -> torch.Tensor:
        return self.mask.select(dim=-1, index=level)

    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None:
        match = LEVEL_PATTERN.search(name)
        if match:
            # 3D variable
            level = int(match.group(1))
            return self.get_mask_level(level)
        else:
            # 2D variable
            if self.surface_mask is not None:
                return self.surface_mask
            return self.get_mask_level(0)

    def get_idepth(self) -> torch.Tensor:
        return self.idepth

    def build_corrector(
        self,
        config: AtmosphereCorrectorConfig | CorrectorSelector,
        gridded_operations: GriddedOperations,
        timestep: timedelta,
    ) -> OceanCorrector:
        if isinstance(config, AtmosphereCorrectorConfig):
            raise ValueError(
                "Cannot build corrector for depth coordinate with an "
                "AtmosphereCorrectorConfig."
            )
        elif config.type != "ocean_corrector":
            raise ValueError(
                f"Cannot build corrector for vertical coordinate {self} with "
                f"corrector selector {config}."
            )
        config_instance = dacite.from_dict(
            data_class=OceanCorrectorConfig,
            data=config.config,
            config=dacite.Config(strict=True),
        )
        return OceanCorrector(
            config=config_instance,
            gridded_operations=gridded_operations,
            vertical_coordinate=self,
            timestep=timestep,
        )

    def build_derive_function(self, timestep: timedelta) -> DeriveFnABC:
        return OceanDeriveFn(self, timestep)

    def build_output_masker(self) -> Callable[[TensorMapping], TensorDict]:
        """
        Returns a StaticMasking object that fills in NaNs outside of mask
        valid points, i.e. where the mask value is 0.

        """
        return StaticMasking(
            mask_value=0,
            fill_value=float("nan"),
            mask=self,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.idepth.dtype

    @property
    def device(self) -> str:
        return self.idepth.device

    @property
    def coords(self) -> dict[str, np.ndarray]:
        return {"idepth": self.idepth.cpu().numpy()}

    def to(self, device: str) -> "DepthCoordinate":
        idepth_on_device = self.idepth.to(device)
        mask_on_device = self.mask.to(device)
        if self.surface_mask is not None:
            surface_mask_on_device = self.surface_mask.to(device)
            return DepthCoordinate(
                idepth_on_device, mask_on_device, surface_mask_on_device
            )
        return DepthCoordinate(idepth_on_device, mask_on_device)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DepthCoordinate):
            return False
        try:
            torch.testing.assert_close(self.idepth, other.idepth)
            torch.testing.assert_close(self.mask, other.mask)
        except AssertionError:
            return False
        return True

    def __repr__(self) -> str:
        return f"DepthCoordinate(\n    idepth={self.idepth},\n    mask={self.mask}\n)"

    def as_dict(self) -> TensorMapping:
        return {"idepth": self.idepth, "mask": self.mask}

    def depth_integral(self, integrand: torch.Tensor) -> torch.Tensor:
        """Compute the depth integral of the integrand.

        ∫ x dz
        where
        - x = integrand
        - z = depth

        Args:
            integrand: A tensor whose last dimension is the vertical.

        Returns:
            A tensor of same shape as integrand but without the last dimension.

        Note:
            NaNs in the integrand are treated as zeros.
        """
        if len(self.idepth) != integrand.shape[-1] + 1:
            raise ValueError(
                f"The last dimension of integrand must match the number of vertical "
                "layers in the depth vertical coordinate. "
                f"Got integrand.shape: {integrand.shape} and idepth.shape: "
                f"{self.idepth.shape}."
            )
        layer_thickness = self.idepth.diff(dim=-1)
        ohc = (integrand * layer_thickness * self.mask).nansum(dim=-1)
        mask = self.get_mask_level(0).expand(ohc.shape)
        return ohc.where(mask > 0, float("nan"))


@dataclasses.dataclass
class NullVerticalCoordinate(VerticalCoordinate):
    """
    A null vertical coordinate system.
    """

    def __eq__(self, other) -> bool:
        return isinstance(other, NullVerticalCoordinate)

    def __repr__(self) -> str:
        return "NullVerticalCoordinate()"

    def __len__(self) -> int:
        return 0

    def build_corrector(
        self,
        config: AtmosphereCorrectorConfig | CorrectorSelector,
        gridded_operations: GriddedOperations,
        timestep: timedelta,
    ) -> CorrectorABC:
        if isinstance(config, AtmosphereCorrectorConfig):
            return AtmosphereCorrector(
                config=config,
                gridded_operations=gridded_operations,
                vertical_coordinate=None,
                timestep=timestep,
            )
        if config.type == "atmosphere_corrector":
            config_instance = dacite.from_dict(
                data_class=AtmosphereCorrectorConfig,
                data=config.config,
                config=dacite.Config(strict=True),
            )
            return AtmosphereCorrector(
                config=config_instance,
                gridded_operations=gridded_operations,
                vertical_coordinate=None,
                timestep=timestep,
            )
        elif config.type == "ocean_corrector":
            config_instance = dacite.from_dict(
                data_class=OceanCorrectorConfig,
                data=config.config,
                config=dacite.Config(strict=True),
            )
            return OceanCorrector(
                config=config_instance,
                gridded_operations=gridded_operations,
                vertical_coordinate=None,
                timestep=timestep,
            )
        else:
            raise ValueError(
                f"Invalid corrector type: {config.type}. "
                "Must be either 'atmosphere_corrector' or 'ocean_corrector'."
            )

    def build_derive_function(self, timestep: timedelta) -> DeriveFnABC:
        return NullDeriveFn()

    def to(self, device: str) -> "NullVerticalCoordinate":
        return self

    def as_dict(self) -> TensorMapping:
        return {}

    @property
    def coords(self) -> dict[str, np.ndarray]:
        return {}


OptionalHybridSigmaPressureCoordinate = (
    HybridSigmaPressureCoordinate | NullVerticalCoordinate
)

OptionalDepthCoordinate = DepthCoordinate | NullVerticalCoordinate


@dataclasses.dataclass
class SerializableVerticalCoordinate:
    """Only for use in serializing/deserializing coordinates with dacite."""

    vertical_coordinate: (
        HybridSigmaPressureCoordinate | DepthCoordinate | NullVerticalCoordinate
    )

    @classmethod
    def from_state(cls, state) -> VerticalCoordinate:
        return dacite.from_dict(
            data_class=cls,
            data={"vertical_coordinate": state},
            config=dacite.Config(strict=True),
        ).vertical_coordinate


@dataclasses.dataclass
class DimSize:
    name: str
    size: int


class HorizontalCoordinates(abc.ABC):
    """
    Parent class for horizontal coordinate system grids.
    Contains coords which must be subclassed to provide the coordinates.
    """

    SelfType = TypeVar("SelfType", bound="HorizontalCoordinates")

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
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
    def xyz(self) -> tuple[float, float, float]:
        pass

    @property
    @abc.abstractmethod
    def dims(self) -> list[str]:
        """Names of model horizontal dimensions."""
        pass

    @property
    @abc.abstractmethod
    def loaded_sizes(self) -> list[DimSize]:
        """Sizes of horizontal dimensions as loaded from training dataset."""
        pass

    @property
    @abc.abstractmethod
    def grid(self) -> Literal["equiangular", "legendre-gauss", "healpix"]:
        pass

    @property
    @abc.abstractmethod
    def area_weights(self) -> torch.Tensor | None:
        pass

    @abc.abstractmethod
    def get_gridded_operations(
        self, mask_provider: MaskProviderABC = NullMaskProvider
    ) -> GriddedOperations:
        pass

    @property
    @abc.abstractmethod
    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Meshgrids of latitudes and longitudes, respectively."""
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @abc.abstractmethod
    def to_state(self) -> TensorMapping:
        pass


@dataclasses.dataclass
class LatLonCoordinates(HorizontalCoordinates):
    """
    Defines a (latitude, longitude) grid.

    Parameters:
        lat: 1-dimensional tensor of latitudes
        lon: 1-dimensional tensor of longitudes
    """

    lon: torch.Tensor
    lat: torch.Tensor

    def __post_init__(self):
        self._area_weights: torch.Tensor | None = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, LatLonCoordinates):
            return False
        lat_eq = torch.allclose(self.lat, other.lat)
        lon_eq = torch.allclose(self.lon, other.lon)
        return lat_eq and lon_eq

    def __repr__(self) -> str:
        return f"LatLonCoordinates(\n    lat={self.lat},\n    lon={self.lon}\n"

    def to(self, device: str) -> "LatLonCoordinates":
        return LatLonCoordinates(
            lon=self.lon.to(device),
            lat=self.lat.to(device),
        )

    @property
    def area_weights(self) -> torch.Tensor:
        if self._area_weights is None:
            self._area_weights = metrics.spherical_area_weights(self.lat, len(self.lon))
        return self._area_weights

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {
            self.dims[0]: self.lat.cpu().type(torch.float32).numpy(),
            self.dims[1]: self.lon.cpu().type(torch.float32).numpy(),
        }

    @property
    def xyz(self) -> tuple[float, float, float]:
        lats, lons = np.broadcast_arrays(
            self.coords[self.dims[0]][:, None], self.coords[self.dims[1]][None, :]
        )
        return lon_lat_to_xyz(lons, lats)

    @property
    def dims(self) -> list[str]:
        return ["lat", "lon"]

    @property
    def loaded_sizes(self) -> list[DimSize]:
        return [
            DimSize(self.dims[0], len(self.lat)),
            DimSize(self.dims[1], len(self.lon)),
        ]

    @property
    def grid(self) -> Literal["equiangular", "legendre-gauss"]:
        if torch.allclose(
            self.lat[1:] - self.lat[:-1],
            self.lat[1] - self.lat[0],
        ):
            return "equiangular"
        else:
            return "legendre-gauss"

    def get_gridded_operations(
        self, mask_provider: MaskProviderABC = NullMaskProvider
    ) -> LatLonOperations:
        return LatLonOperations(self.area_weights, mask_provider)

    @property
    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.meshgrid(self.lat, self.lon, indexing="ij")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.lat), len(self.lon))

    def to_state(self) -> TensorMapping:
        return {"lat": self.lat, "lon": self.lon}


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

    def __post_init__(self):
        if not len(self.face) == 12:
            raise ValueError("HEALPixCoordinates must have 12 faces.")
        if not len(self.height) == len(self.width):
            raise ValueError(
                "HEALPixCoordinates must have the same number of heights and widths."
            )
        order = int(math.log2(len(self.width)))
        if 2**order != len(self.width):
            raise ValueError(
                "HEALPixCoordinates must have a width that is a power of 2."
            )
        self.nside = len(self.width)

    def __eq__(self, other) -> bool:
        if not isinstance(other, HEALPixCoordinates):
            return False
        if (
            self.face.shape != other.face.shape
            or self.height.shape != other.height.shape
            or self.width.shape != other.width.shape
        ):
            return False
        return (
            torch.allclose(self.face, other.face)
            and torch.allclose(self.height, other.height)
            and torch.allclose(self.width, other.width)
        )

    def __repr__(self) -> str:
        return (
            "HEALPixCoordinates(\n"
            f"    face={self.face},\n"
            f"    height={self.height},\n"
            f"    width={self.width}\n"
            ")"
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
    def xyz(self) -> tuple[float, float, float]:
        level = int(math.log2(len(self.width)))
        hpx = e2ghpx.Grid(level=level, pixel_order=e2ghpx.HEALPIX_PAD_XY)
        lats = hpx.lat
        lats = lats.reshape(len(self.face), len(self.width), len(self.height))
        lons = hpx.lon
        lons = lons.reshape(len(self.face), len(self.width), len(self.height))
        x, y, z = lon_lat_to_xyz(lat=lats, lon=lons)
        return x, y, z

    @property
    def dims(self) -> list[str]:
        return ["face", "height", "width"]

    @property
    def loaded_dims(self) -> list[str]:
        return self.dims

    @property
    def loaded_sizes(self) -> list[DimSize]:
        return [
            DimSize("face", len(self.face)),
            DimSize("height", len(self.width)),
            DimSize("width", len(self.height)),
        ]

    @property
    def grid(self) -> Literal["healpix"]:
        return "healpix"

    @property
    def area_weights(self) -> Literal[None]:
        return None

    def get_gridded_operations(
        self, mask_provider: MaskProviderABC = NullMaskProvider
    ) -> HEALPixOperations:
        # this code is necessary because when no masks are in a given dataset, we return
        # an empty MaskProvider instead of the NullMaskProvider.
        if mask_provider == NullMaskProvider:
            null_mask = True
        elif isinstance(mask_provider, MaskProvider):
            null_mask = len(mask_provider.masks) == 0
        else:
            raise TypeError(
                f"Don't know how to handle given mask_provider: {mask_provider}"
            )
        if not (null_mask):
            raise NotImplementedError(
                "HEALPixCoordinates does not support a mask provider. "
                "Use NullMaskProvider when getting gridded operations "
                "for HEALPixCoordinates."
            )
        return HEALPixOperations(self.nside)

    @property
    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        # We'll return a 3D (face, width, height) tensor representing the lat-lon
        # coordinates of this grid.
        level = int(math.log2(len(self.width)))
        hpx = e2ghpx.Grid(level=level, pixel_order=e2ghpx.HEALPIX_PAD_XY)
        lats = hpx.lat
        lats = lats.reshape(self.shape)
        lons = hpx.lon
        lons = lons.reshape(self.shape)
        return lats, lons

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.face), len(self.width), len(self.height))

    def to_state(self) -> TensorMapping:
        return {"face": self.face, "height": self.height, "width": self.width}


@dataclasses.dataclass
class SerializableHorizontalCoordinates:
    """Only for use in serializing/deserializing coordinates with dacite."""

    horizontal_coordinates: LatLonCoordinates | HEALPixCoordinates

    @classmethod
    def from_state(cls, state) -> HorizontalCoordinates:
        return dacite.from_dict(
            data_class=cls,
            data={"horizontal_coordinates": state},
            config=dacite.Config(strict=True),
        ).horizontal_coordinates
