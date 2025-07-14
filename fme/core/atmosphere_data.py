from collections.abc import Mapping
from typing import Protocol

import torch

from fme.core import metrics
from fme.core.constants import (
    GRAVITY,
    LATENT_HEAT_OF_VAPORIZATION,
    RDGAS,
    RVGAS,
    SPECIFIC_HEAT_OF_DRY_AIR_CONST_PRESSURE,
    SPECIFIC_HEAT_OF_DRY_AIR_CONST_VOLUME,
)
from fme.core.stacker import Stacker
from fme.core.typing_ import TensorDict, TensorMapping

ATMOSPHERE_FIELD_NAME_PREFIXES = {
    "specific_total_water": ["specific_total_water_"],
    "surface_pressure": ["PRESsfc", "PS"],
    "surface_height": ["HGTsfc"],
    "surface_geopotential": ["PHIS"],
    "tendency_of_total_water_path_due_to_advection": [
        "tendency_of_total_water_path_due_to_advection"
    ],
    "latent_heat_flux": ["LHTFLsfc", "LHFLX"],
    "sensible_heat_flux": ["SHTFLsfc", "SHFLX"],
    "precipitation_rate": ["PRATEsfc", "surface_precipitation_rate"],
    "sfc_down_sw_radiative_flux": ["DSWRFsfc", "FSDS"],
    "sfc_up_sw_radiative_flux": ["USWRFsfc", "surface_upward_shortwave_flux"],
    "sfc_down_lw_radiative_flux": ["DLWRFsfc", "FLDS"],
    "sfc_up_lw_radiative_flux": ["ULWRFsfc", "surface_upward_longwave_flux"],
    "toa_up_lw_radiative_flux": ["ULWRFtoa", "FLUT"],
    "toa_up_sw_radiative_flux": ["USWRFtoa", "top_of_atmos_upward_shortwave_flux"],
    "toa_down_sw_radiative_flux": ["DSWRFtoa", "SOLIN"],
    "air_temperature": ["air_temperature_", "T_"],
    "frozen_precipitation_rate": ["total_frozen_precipitation_rate"],
    "eastward_wind_at_10m": ["UGRD10m"],
    "northward_wind_at_10m": ["VGRD10m"],
}


class HasAtmosphereVerticalIntegral(Protocol):
    def vertical_integral(
        self,
        field: torch.Tensor,
        surface_pressure: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def interface_pressure(self, surface_pressure: torch.Tensor) -> torch.Tensor:
        pass

    def get_ak(self) -> torch.Tensor:
        pass

    def get_bk(self) -> torch.Tensor:
        pass


class AtmosphereData:
    """Container for atmospheric data for accessing variables and providing
    torch.Tensor views on data with multiple vertical levels.
    """

    def __init__(
        self,
        atmosphere_data: TensorMapping,
        vertical_coordinate: HasAtmosphereVerticalIntegral | None = None,
        atmosphere_field_name_prefixes: Mapping[str, list[str]] | None = None,
    ):
        """
        Initializes the instance based on the provided data and prefixes.

        Args:
            atmosphere_data: Mapping from field names to tensors.
            vertical_coordinate: The vertical coordinate of the model. If not provided,
                then variables which require vertical integration will raise an error.
            atmosphere_field_name_prefixes: Mapping which defines the correspondence
                between an arbitrary set of "standard" names (e.g., "surface_pressure"
                or "air_temperature") and lists of possible names or prefix variants
                (e.g., ["PRESsfc", "PS"] or ["air_temperature_", "T_"]) found in the
                data.
        """
        if atmosphere_field_name_prefixes is None:
            atmosphere_field_name_prefixes = ATMOSPHERE_FIELD_NAME_PREFIXES.copy()
        self._data = dict(atmosphere_data)
        self._prefix_map = atmosphere_field_name_prefixes
        self._vertical_coordinate = vertical_coordinate
        self._stacker = Stacker(atmosphere_field_name_prefixes)

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
    def air_temperature(self) -> torch.Tensor:
        """Returns all vertical levels of air_temperature, e.g. a tensor of
        shape `(..., vertical_level)`.
        """
        return self._stacker("air_temperature", self.data)

    @property
    def specific_total_water(self) -> torch.Tensor:
        """Returns all vertical levels of specific total water, e.g. a tensor of
        shape `(..., vertical_level)`.
        """
        return self._stacker("specific_total_water", self.data)

    @property
    def surface_height(self) -> torch.Tensor:
        try:
            return self._get("surface_height")
        except KeyError:
            # E3SM saves geopotential not surface height so need to convert
            # by using g value from e3sm
            GRAVITY_E3SM = 9.80616
            return self._get("surface_geopotential") / GRAVITY_E3SM

    @property
    def surface_pressure(self) -> torch.Tensor:
        return self._get("surface_pressure")

    def set_surface_pressure(self, value: torch.Tensor):
        self._set("surface_pressure", value)

    @property
    def toa_down_sw_radiative_flux(self) -> torch.Tensor:
        return self._get("toa_down_sw_radiative_flux")

    @property
    def toa_up_sw_radiative_flux(self) -> torch.Tensor:
        return self._get("toa_up_sw_radiative_flux")

    @property
    def toa_up_lw_radiative_flux(self) -> torch.Tensor:
        return self._get("toa_up_lw_radiative_flux")

    @property
    def surface_pressure_due_to_dry_air(self) -> torch.Tensor:
        if self._vertical_coordinate is None:
            raise ValueError("Vertical coordinate must be provided to compute dry air.")
        return metrics.surface_pressure_due_to_dry_air(
            self.surface_pressure,
            self.total_water_path,
        )

    @property
    def total_water_path(self) -> torch.Tensor:
        if self._vertical_coordinate is None:
            raise ValueError(
                "Vertical coordinate must be provided to compute total water path."
            )
        return self._vertical_coordinate.vertical_integral(
            self.specific_total_water,
            self.surface_pressure,
        )

    @property
    def frozen_precipitation_rate(self) -> torch.Tensor:
        # Return zero if any necessary fields are missing
        try:
            return self._get("frozen_precipitation_rate")
        except KeyError:
            try:
                return (
                    self._data["ICEsfc"]
                    + self._data["GRAUPELsfc"]
                    + self._data["SNOWsfc"]
                )
            except KeyError:
                return torch.zeros_like(self.surface_pressure)

    @property
    def net_surface_energy_flux_without_frozen_precip(self) -> torch.Tensor:
        return metrics.net_surface_energy_flux(
            self._get("sfc_down_lw_radiative_flux"),
            self._get("sfc_up_lw_radiative_flux"),
            self._get("sfc_down_sw_radiative_flux"),
            self._get("sfc_up_sw_radiative_flux"),
            self._get("latent_heat_flux"),
            self._get("sensible_heat_flux"),
        )

    @property
    def net_surface_energy_flux(self) -> torch.Tensor:
        return metrics.net_surface_energy_flux(
            self._get("sfc_down_lw_radiative_flux"),
            self._get("sfc_up_lw_radiative_flux"),
            self._get("sfc_down_sw_radiative_flux"),
            self._get("sfc_up_sw_radiative_flux"),
            self._get("latent_heat_flux"),
            self._get("sensible_heat_flux"),
            frozen_precipitation_rate=self.frozen_precipitation_rate,
        )

    @property
    def net_top_of_atmosphere_energy_flux(self) -> torch.Tensor:
        return metrics.net_top_of_atmosphere_energy_flux(
            self._get("toa_down_sw_radiative_flux"),
            self._get("toa_up_sw_radiative_flux"),
            self._get("toa_up_lw_radiative_flux"),
        )

    @property
    def net_energy_flux_into_atmosphere(self) -> torch.Tensor:
        return self.net_top_of_atmosphere_energy_flux - self.net_surface_energy_flux

    @property
    def precipitation_rate(self) -> torch.Tensor:
        """
        Precipitation rate in kg m-2 s-1.
        """
        return self._get("precipitation_rate")

    def set_precipitation_rate(self, value: torch.Tensor):
        self._set("precipitation_rate", value)

    @property
    def latent_heat_flux(self) -> torch.Tensor:
        """
        Latent heat flux in W m-2.
        """
        return self._get("latent_heat_flux")

    @property
    def evaporation_rate(self) -> torch.Tensor:
        """
        Evaporation rate in kg m-2 s-1.
        """
        lhf = self._get("latent_heat_flux")  # W/m^2
        # (W/m^2) / (J/kg) = (J s^-1 m^-2) / (J/kg) = kg/m^2/s
        return lhf / LATENT_HEAT_OF_VAPORIZATION

    def set_evaporation_rate(self, value: torch.Tensor):
        self._set("latent_heat_flux", value * LATENT_HEAT_OF_VAPORIZATION)

    @property
    def tendency_of_total_water_path_due_to_advection(self) -> torch.Tensor:
        """
        Tendency of total water path due to advection in kg m-2 s-1.
        """
        return self._get("tendency_of_total_water_path_due_to_advection")

    def set_tendency_of_total_water_path_due_to_advection(self, value: torch.Tensor):
        self._set("tendency_of_total_water_path_due_to_advection", value)

    def height_at_log_midpoint(self) -> torch.Tensor:
        """
        Compute vertical height at layer log midpoints.
        """
        if self._vertical_coordinate is None:
            raise ValueError(
                "Vertical coordinate must be provided to compute height at log midpoint"
            )
        interface_pressure = self._vertical_coordinate.interface_pressure(
            self.surface_pressure
        )
        layer_thickness = compute_layer_thickness(
            pressure_at_interface=interface_pressure,
            air_temperature=self.air_temperature,
            specific_total_water=self.specific_total_water,
        )
        height_at_interface = _height_at_interface(layer_thickness, self.surface_height)
        return (height_at_interface[..., :-1] * height_at_interface[..., 1:]) ** 0.5

    @property
    def height_at_midpoint(self) -> torch.Tensor:
        """Compute vertical height at layer midpoints with linear interpolation."""
        if self._vertical_coordinate is None:
            raise ValueError(
                "Vertical coordinate must be provided to compute height at mmidpoint"
            )
        interface_pressure = self._vertical_coordinate.interface_pressure(
            self.surface_pressure
        )
        layer_thickness = compute_layer_thickness(
            pressure_at_interface=interface_pressure,
            air_temperature=self.air_temperature,
            specific_total_water=self.specific_total_water,
        )
        height_at_interface = _height_at_interface(layer_thickness, self.surface_height)
        return 0.5 * (height_at_interface[..., :-1] + height_at_interface[..., 1:])

    @property
    def moist_static_energy(self) -> torch.Tensor:
        """
        Compute moist static energy.
        """
        # ACE does not currently prognose specific humidity, so here we closely
        # approximate this using specific total water (<0.01% effect on total MSE).
        return (
            self.air_temperature * SPECIFIC_HEAT_OF_DRY_AIR_CONST_PRESSURE
            + self.specific_total_water * LATENT_HEAT_OF_VAPORIZATION
            + self.height_at_midpoint * GRAVITY
        )

    @property
    def total_energy_ace2(self) -> torch.Tensor:
        """
        Compute the total energy, following some assumptions used for ACE2 models.

        Namely, we ignore kinetic energy, use hydrostatic balance to compute the
        geoportential energy, and approximate specific humidity with specific total
        water. We also ignore the ice water contribution to total energy.
        """
        return (
            self.air_temperature * SPECIFIC_HEAT_OF_DRY_AIR_CONST_VOLUME
            + self.specific_total_water * LATENT_HEAT_OF_VAPORIZATION
            + self.height_at_midpoint * GRAVITY
        )

    @property
    def total_energy_ace2_path(self) -> torch.Tensor:
        """Compute vertical integral of total energy."""
        if self._vertical_coordinate is None:
            raise ValueError(
                "Vertical coordinate must be provided to compute total energy ACE2 path"
            )
        return self._vertical_coordinate.vertical_integral(
            self.total_energy_ace2, self.surface_pressure
        )

    @property
    def windspeed_at_10m(self) -> torch.Tensor:
        """Compute the windspeed at 10m above surface."""
        return torch.sqrt(
            self._get("eastward_wind_at_10m") ** 2
            + self._get("northward_wind_at_10m") ** 2
        )


def compute_layer_thickness(
    pressure_at_interface: torch.Tensor,
    air_temperature: torch.Tensor,
    specific_total_water: torch.Tensor,
) -> torch.Tensor:
    """
    Computes vertical thickness of each layer assuming hydrostatic equilibrium.
    ACE does not currently prognose specific humidity, so here we closely
    approximate this using specific total water.
    """
    tv = air_temperature * (1 + (RVGAS / RDGAS - 1.0) * specific_total_water)
    # Enforce min log(p) = 0 so that geopotential energy calculation is finite.
    # This is equivalent to setting the TOA pressure to 1 Pa if it is less than that.
    # The ERA5 data has a TOA pressure of 0.0 Pa which causes issues otherwise.
    dlogp = torch.clamp(torch.log(pressure_at_interface), min=0.0).diff(dim=-1)
    return dlogp * RDGAS * tv / GRAVITY


def _height_at_interface(
    layer_thickness: torch.Tensor, surface_height: torch.Tensor
) -> torch.Tensor:
    """
    Computes height at layer interfaces from layer thickness and surface height.
    Vertical coordinate is the last tensor dimension.
    """
    cumulative_thickness = torch.cumsum(layer_thickness.flip(dims=(-1,)), dim=-1).flip(
        dims=(-1,)
    )
    # Sometimes surface height data has negative values, which are filled with 0.
    hsfc = torch.where(surface_height < 0.0, 0, surface_height).reshape(
        *surface_height.shape, 1
    )
    return torch.concat(
        [
            (cumulative_thickness + hsfc.broadcast_to(cumulative_thickness.shape)),
            hsfc,
        ],
        dim=-1,
    )
