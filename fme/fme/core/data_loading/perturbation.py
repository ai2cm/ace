import abc
import dataclasses
from typing import Any, Callable, Dict, Mapping, Tuple, Type

import dacite
import numpy as np
import torch


@dataclasses.dataclass
class PerturbationSelector:
    name: str
    config: Mapping[str, Any]

    def __post_init__(self):
        try:
            self.perturbation = get_from_registry(self.name).from_state(self.config)
        except KeyError:
            raise ValueError(
                f"unknown module type {self.name}, "
                f"known module types are {list(PERTURBATION_REGISTRY.keys())}"
            )


@dataclasses.dataclass
class SSTPerturbation:
    """
    Configuration for sea surface temperature perturbations
    applied to initial condition and forcing data.
    Currently, this is strictly applied to both.

    Attributes:
        sst: List of perturbation selectors for SST perturbations.
    """

    sst: list[PerturbationSelector]


@dataclasses.dataclass
class PerturbationConfig(abc.ABC):
    """
    Returns a perturbation function config class
    """

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "PerturbationConfig":
        """
        Create a PerturbationSelector from a dictionary containing all the information
        needed to build a PerturbationConfig.
        """
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @abc.abstractmethod
    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        ocean_fraction: torch.Tensor,
    ) -> None:
        ...


PERTURBATION_REGISTRY: Dict[str, Type[PerturbationConfig]] = {}


def get_from_registry(name) -> Type[PerturbationConfig]:
    return PERTURBATION_REGISTRY[name]


def register(
    name: str,
) -> Callable[[Type[PerturbationConfig]], Type[PerturbationConfig]]:
    """
    Register a new PerturbationConfig type with the PERTURBATION_REGISTRY.

    This is useful for adding new PerturbationConfig types to the registry from
    other modules.

    Args:
        name: name of the PerturbationConfig type to register

    Returns:
        a decorator which registers the decorated class with the PERTURBATION_REGISTRY
    """
    if not isinstance(name, str):
        raise TypeError(
            f"name must be a string, got {name}, "
            "make sure to use as @register('module_name')"
        )

    def decorator(cls: Type[PerturbationConfig]) -> Type[PerturbationConfig]:
        PERTURBATION_REGISTRY[name] = cls
        return cls

    return decorator


def _get_ocean_mask(ocean_fraction: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
    return ocean_fraction > cutoff


@register("constant")
@dataclasses.dataclass
class ConstantConfig(PerturbationConfig):
    """
    Configuration for a constant perturbation.
    """

    amplitude: float = 1.0

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        ocean_fraction: torch.Tensor,
    ):
        ocean_mask = _get_ocean_mask(ocean_fraction)
        data[ocean_mask] += self.amplitude  # type: ignore


@register("greens_function")
@dataclasses.dataclass
class GreensFunctionConfig(PerturbationConfig):
    """
    Configuration for a single sinusoidal patch of a Green's function perturbation.
    See equation 1 in Bloch‐Johnson, J., et al. (2024).

    Attributes:
        amplitude: The amplitude of the perturbation,
            maximum is reached at (lat_center, lon_center).
        lat_center: The latitude at the center of the patch in degrees.
        lon_center: The longitude at the center of the patch in degrees.
        lat_width: latitudinal width of the patch in degrees.
        lon_width: longitudinal width of the patch in degrees.
    """

    amplitude: float = 1.0
    lat_center: float = 0.0
    lon_center: float = 0.0
    lat_width: float = 10.0
    lon_width: float = 10.0

    def __post_init__(self):
        self._lat_center_rad = np.deg2rad(self.lat_center)
        self._lon_center_rad = np.deg2rad(self.lon_center)
        self._lat_width_rad = np.deg2rad(self.lat_width)
        self._lon_width_rad = np.deg2rad(self.lon_width)

    def _wrap_longitude_discontinuity(
        self,
        lon: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assume longitude is in the range [0, 360) degrees.
        If the patch crosses the discontinuity at 0/360 degrees,
        shift the longitude accordingly.
        """
        lon_min = self.lon_center - self.lon_width / 2.0
        lon_max = self.lon_center + self.lon_width / 2.0
        if lon_min < 0:
            lon_shifted = ((lon + 180) % 360) - 180
            lon_in_patch = (lon_shifted > lon_min) & (lon_shifted < lon_max)
        elif lon_max > 360:
            lon_in_patch = (lon > lon_min) | (lon < lon_max % 360)
            lon_shifted = ((lon + 180) % 360) + 180
        else:
            lon_in_patch = (lon > lon_min) & (lon < lon_max)
            lon_shifted = lon
        return lon_in_patch, lon_shifted

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        ocean_fraction: torch.Tensor,
    ):
        lat_in_patch = torch.abs(lat - self.lat_center) < self.lat_width / 2.0
        lon_in_patch, lon_shifted = self._wrap_longitude_discontinuity(lon)
        mask = lat_in_patch & lon_in_patch
        ocean_mask = _get_ocean_mask(ocean_fraction)
        perturbation = self.amplitude * (
            torch.cos(
                torch.pi
                / 2
                * (lat.deg2rad() - self._lat_center_rad)
                / (self._lat_width_rad / 2.0)
            )
            ** 2
            * torch.cos(
                torch.pi
                / 2
                * (lon_shifted.deg2rad() - self._lon_center_rad)
                / (self._lon_width_rad / 2.0)
            )
            ** 2
        )
        mask = mask.expand(data.shape)
        perturbation = perturbation.expand(data.shape)
        data[mask & ocean_mask] += perturbation[mask & ocean_mask]
