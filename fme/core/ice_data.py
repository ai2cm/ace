from collections.abc import Mapping
from types import MappingProxyType

import torch

from fme.core.typing_ import TensorDict, TensorMapping

ICE_FIELD_NAME_PREFIXES = MappingProxyType(
    {
        "sea_ice_concentration": ["siconc"],
        "sea_ice_mass": ["simass"],
        "snow_mass": ["sisnmass"],
        "ice_mass_growth": ["LSRCi"],
        "ice_mass_melt": ["LSNKi"],
        "snow_mass_growth": ["LSRCs"],
        "snow_mass_melt": ["LSNKs"],
        "ice_surface_temperature": ["TS"],
        "ice_to_ocean_salt_flux": ["SALTF"],
        "ice_to_ocean_mass_flux": ["ice_to_ocean_mass_flux"],
        "ice_to_ocean_energy_flux": ["BMELT"],
        "ice_to_atmosphere_energy_flux": ["TMELT"],
        "upward_shortwave_radiation": ["SWUP"],
        "sea_surface_fraction": ["sea_surface_fraction"],
        "land_fraction": ["land_fraction"],
        "ocean_fraction": ["ocean_fraction"],
        "ocean_sea_ice_fraction": ["ocean_sea_ice_fraction"],
        "sea_ice_fraction": ["sea_ice_fraction"],
    }
)


class IceData:
    """Container for ice data for accessing variables."""

    def __init__(
        self,
        ice_data: TensorMapping,
        ice_field_name_prefixes: Mapping[str, list[str]] = ICE_FIELD_NAME_PREFIXES,
    ):
        """
        Initializes the instance based on the provided data and prefixes.

        Args:
            ice_data: Mapping from field names to tensors.
            ice_field_name_prefixes: Mapping which defines the correspondence
                between an arbitrary set of "standard" names (e.g.,
                "ice_to_ocean_salt_flux") and lists of possible
                names or prefix variants (e.g., ["SALTF"]) found in the data.
        """
        self._data = dict(ice_data)
        self._prefix_map = ice_field_name_prefixes

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
    def land_fraction(self) -> torch.Tensor:
        """Returns the land fraction."""
        return self._get("land_fraction")
    
    @property
    def sea_surface_fraction(self) -> torch.Tensor:
        """Returns the sea surface fraction."""
        return 1 - self.land_fraction

    @property
    def ocean_fraction(self) -> torch.Tensor:
        """Returns the dynamic ocean fraction, computed from the sea ice
        fraction and land fraction.
        """
        return (1 - self.sea_ice_concentration) * self.sea_surface_fraction

    @property
    def sea_ice_concentration(self) -> torch.Tensor:
        """Returns sea ice concentration."""
        return self._get("sea_ice_concentration")

    @property
    def ocean_sea_ice_fraction(self) -> torch.Tensor:
        """Returns sea ice concentration."""
        return self._get("sea_ice_concentration")

    @property
    def sea_ice_fraction(self) -> torch.Tensor:
        """Returns sea ice fraction."""
        try:
            return self._get("sea_ice_fraction")
        except KeyError:
            sea_frac = self.sea_surface_fraction
            return sea_frac * self.sea_ice_concentration

    @property
    def sea_ice_mass(self) -> torch.Tensor:
        """Returns sea ice mass."""
        return self._get("sea_ice_mass")

    @property
    def snow_mass(self) -> torch.Tensor:
        """Returns snow mass."""
        return self._get("snow_mass")

    @property
    def ice_mass_growth(self) -> torch.Tensor:
        """Returns ice mass growth rate."""
        return self._get("ice_mass_growth")

    @property
    def ice_mass_melt(self) -> torch.Tensor:
        """Returns ice mass melt rate."""
        return self._get("ice_mass_melt")

    @property
    def ice_surface_temperature(self) -> torch.Tensor:
        """Returns ice or snow surface skin temperature."""
        return self._get("ice_surface_temperature")

    @property
    def ice_to_ocean_salt_flux(self) -> torch.Tensor:
        """Returns ice to ocean salt flux."""
        return self._get("ice_to_ocean_salt_flux")

    @property
    def ice_to_ocean_mass_flux(self) -> torch.Tensor:
        """Returns ice to ocean mass flux."""
        try:
            return self._get("ice_to_ocean_mass_flux")
        except KeyError:
            return self.ice_mass_melt + self.ice_mass_growth

    @property
    def ice_to_ocean_energy_flux(self) -> torch.Tensor:
        """Returns ice to ocean melt/growth energy flux."""
        return self._get("ice_to_ocean_energy_flux")

    @property
    def ice_to_atmosphere_energy_flux(self) -> torch.Tensor:
        """Returns ice to atmosphere melting energy flux."""
        return self._get("ice_to_atmosphere_energy_flux")
    
    @property
    def upward_shortwave_radiation(self) -> torch.Tensor:
        """Returns the upward surface shortwave."""
        return self._get("upward_shortwave_radiation")
