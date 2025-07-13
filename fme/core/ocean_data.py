from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Protocol

import torch

from fme.core.constants import DENSITY_OF_WATER_CM4, SPECIFIC_HEAT_OF_WATER_CM4
from fme.core.stacker import Stacker
from fme.core.typing_ import TensorDict, TensorMapping

OCEAN_FIELD_NAME_PREFIXES = MappingProxyType(
    {
        "sea_water_potential_temperature": ["thetao_"],
        "sea_water_salinity": ["so_"],
        "sea_water_x_velocity": ["uo_"],
        "sea_water_y_velocity": ["vo_"],
        "sea_surface_height_above_geoid": ["zos"],
        "sea_surface_temperature": ["sst"],
        "sea_ice_fraction": ["sea_ice_fraction"],
        "sea_ice_thickness": ["sea_ice_thickness"],
        "sea_ice_volume": ["sea_ice_volume"],
        "ocean_sea_ice_fraction": ["ocean_sea_ice_fraction"],
        "land_fraction": ["land_fraction"],
        "net_downward_surface_heat_flux": ["hfds"],
        "geothermal_heat_flux": ["hfgeou"],
        "sea_surface_fraction": ["sea_surface_fraction"],
    }
)


class HasOceanDepthIntegral(Protocol):
    def __len__(self) -> int: ...

    def get_mask(self) -> torch.Tensor: ...

    def get_mask_level(self, level: int) -> torch.Tensor: ...

    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None: ...

    def get_idepth(self) -> torch.Tensor: ...

    def depth_integral(
        self,
        integrand: torch.Tensor,
    ) -> torch.Tensor: ...

    def build_output_masker(self) -> Callable[[TensorMapping], TensorDict]: ...

    def to(self, device: str) -> "HasOceanDepthIntegral": ...


class OceanData:
    """Container for ocean data for accessing variables and providing
    torch.Tensor views on data with multiple depth levels.
    """

    def __init__(
        self,
        ocean_data: TensorMapping,
        depth_coordinate: HasOceanDepthIntegral | None = None,
        ocean_field_name_prefixes: Mapping[str, list[str]] = OCEAN_FIELD_NAME_PREFIXES,
    ):
        """
        Initializes the instance based on the provided data and prefixes.

        Args:
            ocean_data: Mapping from field names to tensors.
            depth_coordinate: The depth coordinate of the model.
            ocean_field_name_prefixes: Mapping which defines the correspondence
                between an arbitrary set of "standard" names (e.g.,
                "potential_temperature" or "salinity") and lists of possible
                names or prefix variants (e.g., ["thetao_"] or
                ["zos"]) found in the data.
        """
        self._data = dict(ocean_data)
        self._prefix_map = ocean_field_name_prefixes
        self._depth_coordinate = depth_coordinate
        self._stacker = Stacker(ocean_field_name_prefixes)

    @property
    def data(self) -> TensorDict:
        """Mapping from field names to tensors."""
        return self._data

    def __getitem__(self, name: str):
        return getattr(self, name)

    def _get_prefix(self, prefix):
        return self.data[prefix]

    def _set(self, name, value):
        for prefix in self._prefix_map[name]:
            if prefix in self.data.keys():
                self._set_prefix(prefix, value)
                return
        raise KeyError(name)

    def _set_prefix(self, prefix, value):
        self.data[prefix] = value

    def _get(self, name):
        for prefix in self._prefix_map[name]:
            if prefix in self.data.keys():
                return self._get_prefix(prefix)
        raise KeyError(name)

    @property
    def sea_water_potential_temperature(self) -> torch.Tensor:
        """Returns all depth levels of potential temperature."""
        return self._stacker("sea_water_potential_temperature", self.data)

    @property
    def sea_water_salinity(self) -> torch.Tensor:
        """Returns all depth levels of salinity."""
        return self._stacker("sea_water_salinity", self.data)

    @property
    def sea_water_x_velocity(self) -> torch.Tensor:
        """Returns all depth levels of x-velocity."""
        return self._stacker("sea_water_x_velocity", self.data)

    @property
    def sea_water_y_velocity(self) -> torch.Tensor:
        """Returns all depth levels of y-velocity."""
        return self._stacker("sea_water_y_velocity", self.data)

    @property
    def sea_surface_temperature(self) -> torch.Tensor:
        """Returns surface temperature."""
        return self._get("sea_surface_temperature")

    @property
    def sea_surface_height_above_geoid(self) -> torch.Tensor:
        """Returns sea surface height above geoid."""
        return self._get("sea_surface_height_above_geoid")

    @property
    def ocean_heat_content(self) -> torch.Tensor:
        """Returns column-integrated ocean heat content."""
        if self._depth_coordinate is None:
            raise ValueError(
                "Depth coordinate must be provided to compute column-integrated "
                "ocean heat content."
            )
        return self._depth_coordinate.depth_integral(
            self.sea_water_potential_temperature
            * SPECIFIC_HEAT_OF_WATER_CM4
            * DENSITY_OF_WATER_CM4
        )

    @property
    def sea_surface_fraction(self) -> torch.Tensor:
        """Returns the sea surface fraction."""
        return self._get("sea_surface_fraction")

    @property
    def net_downward_surface_heat_flux(self) -> torch.Tensor:
        """Net heat flux downward across the ocean surface (below the sea-ice)."""
        return self._get("net_downward_surface_heat_flux")

    @property
    def geothermal_heat_flux(self) -> torch.Tensor:
        """Geothermal heat flux."""
        return self._get("geothermal_heat_flux")

    @property
    def sea_ice_fraction(self) -> torch.Tensor:
        """Returns the sea ice fraction."""
        try:
            return self._get("sea_ice_fraction")
        except KeyError:
            land_fraction = self.land_fraction
            ocean_sea_ice_fraction = self.ocean_sea_ice_fraction
            return ocean_sea_ice_fraction * (1 - land_fraction)

    @property
    def land_fraction(self) -> torch.Tensor:
        """Returns the land fraction."""
        return self._get("land_fraction")

    @property
    def ocean_sea_ice_fraction(self) -> torch.Tensor:
        """Returns the sea ice fraction as a proportion of the sea surface."""
        return self._get("ocean_sea_ice_fraction")

    @property
    def ocean_fraction(self) -> torch.Tensor:
        """Returns the dynamic ocean fraction, computed from the sea ice
        fraction and land fraction.
        """
        return 1 - self.land_fraction - self.sea_ice_fraction
