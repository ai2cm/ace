import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Protocol

import torch

from fme.core.constants import (
    DENSITY_OF_WATER_CM4,
    REFERENCE_SALINITY_PSU,
    SPECIFIC_HEAT_OF_WATER_CM4,
)
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
        "sea_ice_thickness": ["HI"],
        "sea_ice_volume": ["sea_ice_volume"],
        "ocean_sea_ice_fraction": ["ocean_sea_ice_fraction"],
        "land_fraction": ["land_fraction"],
        "net_downward_surface_heat_flux": ["hfds"],
        "net_downward_surface_heat_flux_total_area": ["hfds_total_area"],
        "geothermal_heat_flux": ["hfgeou"],
        "water_flux_into_sea_water": ["wfo"],
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


class HasCellAreaInMetersSquared(Protocol):
    """Protocol for objects that can provide cell areas in square meters."""

    @property
    def area_weights_m2(self) -> torch.Tensor: ...


class OceanData:
    """Container for ocean data for accessing variables and providing
    torch.Tensor views on data with multiple depth levels.
    """

    def __init__(
        self,
        ocean_data: TensorMapping,
        depth_coordinate: HasOceanDepthIntegral | None = None,
        ocean_field_name_prefixes: Mapping[str, list[str]] = OCEAN_FIELD_NAME_PREFIXES,
        cell_area_provider: HasCellAreaInMetersSquared | None = None,
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
            cell_area_provider: An object providing cell areas in square meters
                via the ``area_weights_m2`` property. Used by derived variables
                that need cell area information (e.g. sea ice thickness).
        """
        self._data = dict(ocean_data)
        self._prefix_map = ocean_field_name_prefixes
        self._depth_coordinate = depth_coordinate
        self._stacker = Stacker(ocean_field_name_prefixes)
        self._cell_area_provider = cell_area_provider

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
    def ocean_salt_content(self) -> torch.Tensor:
        """Returns column-integrated ocean salt content."""
        if self._depth_coordinate is None:
            raise ValueError(
                "Depth coordinate must be provided to compute column-integrated "
                "ocean salt content."
            )
        return self._depth_coordinate.depth_integral(
            self.sea_water_salinity * DENSITY_OF_WATER_CM4
        )

    @property
    def water_flux_into_sea_water(self) -> torch.Tensor:
        """Returns water flux into sea water (wfo)."""
        return self._get("water_flux_into_sea_water")

    @property
    def sea_surface_fraction(self) -> torch.Tensor:
        """Returns the sea surface fraction."""
        try:
            return self._get("sea_surface_fraction")
        except KeyError:
            return 1 - self.land_fraction

    @property
    def net_downward_surface_heat_flux(self) -> torch.Tensor:
        """Net heat flux downward across the ocean surface (below the sea-ice)."""
        try:
            return self._get("net_downward_surface_heat_flux")
        except KeyError:
            # derive from the sea-surface-fraction-weighted version
            return (
                self.net_downward_surface_heat_flux_total_area
                / self.sea_surface_fraction
            )

    @property
    def net_downward_surface_heat_flux_total_area(self) -> torch.Tensor:
        """Net heat flux downward across the ocean surface (below the sea-ice),
        normalized by total grid cell area.
        """
        return self._get("net_downward_surface_heat_flux_total_area")

    @property
    def geothermal_heat_flux(self) -> torch.Tensor:
        """Geothermal heat flux."""
        try:
            return self._get("geothermal_heat_flux")
        except KeyError:
            return torch.zeros_like(self.sea_surface_fraction)

    @property
    def net_energy_flux_into_ocean(self) -> torch.Tensor:
        return (
            self.net_downward_surface_heat_flux + self.geothermal_heat_flux
        ) * self.sea_surface_fraction

    @property
    def net_virtual_salt_flux_into_ocean(self) -> torch.Tensor:
        """Virtual salt flux into the ocean column (g/m2/s).

        Positive wfo (freshwater in) dilutes salt, giving a negative salt flux.
        Uses a fixed reference salinity for diagnostic purposes.
        """
        return (
            -REFERENCE_SALINITY_PSU
            * self.water_flux_into_sea_water
            * (self.sea_surface_fraction)
        )

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

    @property
    def area_weights_m2(self) -> torch.Tensor:
        """Returns cell areas in square meters.

        Raises:
            ValueError: If a cell area provider was not provided.
        """
        if self._cell_area_provider is None:
            raise ValueError(
                "A cell area provider must be provided to access cell area information."
            )
        return self._cell_area_provider.area_weights_m2

    @property
    def sea_ice_thickness(self) -> torch.Tensor:
        """Returns the sea ice thickness."""
        try:
            return self._get("sea_ice_thickness")
        except KeyError:
            sfrac = self.sea_surface_fraction
            sea_ice_vol = self.sea_ice_volume
            try:
                sea_ice_frac = self.ocean_sea_ice_fraction * sfrac
            except KeyError:
                # assumes that sea_ice_fraction comes from compute_coupled_sea_ice
                # in scripts/data_process/coupled_dataset_utils.py
                lfrac = self.land_fraction
                sea_ice_frac = self.sea_ice_fraction * sfrac / (1 - lfrac)
            cell_area = self.area_weights_m2
            return torch.where(
                torch.isnan(sea_ice_vol),
                float("nan"),
                torch.nan_to_num(
                    torch.exp(
                        9 * math.log(10)
                        + torch.log(sea_ice_vol)
                        - torch.log(cell_area)
                        - torch.log(sea_ice_frac)
                    )
                ),
            )

    @property
    def sea_ice_volume(self) -> torch.Tensor:
        """Returns the sea ice volume."""
        return self._get("sea_ice_volume")
