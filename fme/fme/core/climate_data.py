from types import MappingProxyType
from typing import Callable, List, Mapping

import torch

from fme.core import metrics
from fme.core.constants import (
    GRAVITY,
    LATENT_HEAT_OF_VAPORIZATION,
    RDGAS,
    RVGAS,
    SPECIFIC_HEAT_OF_DRY_AIR_CONST_PRESSURE,
)
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.stacker import Stacker
from fme.core.typing_ import TensorDict, TensorMapping

CLIMATE_FIELD_NAME_PREFIXES = MappingProxyType(
    {
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
    }
)


class ClimateData:
    """Container for climate data for accessing variables and providing
    torch.Tensor views on data with multiple vertical levels.
    """

    def __init__(
        self,
        climate_data: TensorMapping,
        climate_field_name_prefixes: Mapping[
            str, List[str]
        ] = CLIMATE_FIELD_NAME_PREFIXES,
    ):
        """
        Initializes the instance based on the climate data and prefixes.

        Args:
            climate_data: Mapping from field names to tensors.
            climate_field_name_prefixes: Mapping which defines the correspondence
                between an arbitrary set of "standard" names (e.g., "surface_pressure"
                or "air_temperature") and lists of possible names or prefix variants
                (e.g., ["PRESsfc", "PS"] or ["air_temperature_", "T_"]) found in the
                data.
        """
        self._data = dict(climate_data)
        self._prefix_map = climate_field_name_prefixes
        self._stacker = Stacker(climate_field_name_prefixes)

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

    @surface_pressure.setter
    def surface_pressure(self, value: torch.Tensor):
        self._set("surface_pressure", value)

    @property
    def toa_down_sw_radiative_flux(self) -> torch.Tensor:
        return self._get("toa_down_sw_radiative_flux")

    @toa_down_sw_radiative_flux.setter
    def toa_down_sw_radiative_flux(self, value: torch.Tensor):
        self._set("toa_down_sw_radiative_flux", value)

    @property
    def toa_up_sw_radiative_flux(self) -> torch.Tensor:
        return self._get("toa_up_sw_radiative_flux")

    @toa_up_sw_radiative_flux.setter
    def toa_up_sw_radiative_flux(self, value: torch.Tensor):
        self._set("toa_up_sw_radiative_flux", value)

    @property
    def toa_up_lw_radiative_flux(self) -> torch.Tensor:
        return self._get("toa_up_lw_radiative_flux")

    @toa_up_lw_radiative_flux.setter
    def toa_up_lw_radiative_flux(self, value: torch.Tensor):
        self._set("toa_up_lw_radiative_flux", value)

    def surface_pressure_due_to_dry_air(
        self, vertical_coordinate: HybridSigmaPressureCoordinate
    ) -> torch.Tensor:
        return metrics.surface_pressure_due_to_dry_air(
            self.specific_total_water,
            self.surface_pressure,
            vertical_coordinate,
        )

    def total_water_path(
        self, vertical_coordinate: HybridSigmaPressureCoordinate
    ) -> torch.Tensor:
        return vertical_coordinate.vertical_integral(
            self.specific_total_water,
            self.surface_pressure,
        )

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
    def precipitation_rate(self) -> torch.Tensor:
        """
        Precipitation rate in kg m-2 s-1.
        """
        return self._get("precipitation_rate")

    @precipitation_rate.setter
    def precipitation_rate(self, value: torch.Tensor):
        self._set("precipitation_rate", value)

    @property
    def latent_heat_flux(self) -> torch.Tensor:
        """
        Latent heat flux in W m-2.
        """
        return self._get("latent_heat_flux")

    @latent_heat_flux.setter
    def latent_heat_flux(self, value: torch.Tensor):
        self._set("latent_heat_flux", value)

    @property
    def evaporation_rate(self) -> torch.Tensor:
        """
        Evaporation rate in kg m-2 s-1.
        """
        lhf = self._get("latent_heat_flux")  # W/m^2
        # (W/m^2) / (J/kg) = (J s^-1 m^-2) / (J/kg) = kg/m^2/s
        return lhf / LATENT_HEAT_OF_VAPORIZATION

    @evaporation_rate.setter
    def evaporation_rate(self, value: torch.Tensor):
        self._set("latent_heat_flux", value * LATENT_HEAT_OF_VAPORIZATION)

    @property
    def tendency_of_total_water_path_due_to_advection(self) -> torch.Tensor:
        """
        Tendency of total water path due to advection in kg m-2 s-1.
        """
        return self._get("tendency_of_total_water_path_due_to_advection")

    @tendency_of_total_water_path_due_to_advection.setter
    def tendency_of_total_water_path_due_to_advection(self, value: torch.Tensor):
        self._set("tendency_of_total_water_path_due_to_advection", value)

    def height_at_log_midpoint(
        self, vertical_coordinate: HybridSigmaPressureCoordinate
    ) -> torch.Tensor:
        """
        Compute vertical height at layer log midpoints.
        """
        interface_pressure = vertical_coordinate.interface_pressure(
            self.surface_pressure
        )
        layer_thickness = _layer_thickness(
            pressure_at_interface=interface_pressure,
            air_temperature=self.air_temperature,
            specific_total_water=self.specific_total_water,
        )
        height_at_interface = _height_at_interface(layer_thickness, self.surface_height)
        return (height_at_interface[..., :-1] * height_at_interface[..., 1:]) ** 0.5

    def moist_static_energy(
        self, vertical_coordinate: HybridSigmaPressureCoordinate
    ) -> torch.Tensor:
        """
        Compute moist static energy.
        """
        # ACE does not currently prognose specific humidity, so here we closely
        # approximate this using specific total water (<0.01% effect on total MSE).
        return (
            self.air_temperature * SPECIFIC_HEAT_OF_DRY_AIR_CONST_PRESSURE
            + self.specific_total_water * LATENT_HEAT_OF_VAPORIZATION
            + self.height_at_log_midpoint(vertical_coordinate) * GRAVITY
        )


def compute_dry_air_absolute_differences(
    climate_data: ClimateData,
    area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
    vertical_coordinate: HybridSigmaPressureCoordinate,
) -> torch.Tensor:
    """
    Computes the absolute value of the dry air tendency of each time step.

    Args:
        climate_data: ClimateData object.
        area_weighted_mean: Function which returns an area-weighted mean.
        vertical_coordinate: The vertical coordinate of the model.

    Returns:
        A tensor of shape (time,) of the absolute value of the dry air tendency
            of each time step.
    """
    try:
        water = climate_data.specific_total_water
        pressure = climate_data.surface_pressure
    except KeyError:
        return torch.tensor([torch.nan])
    ps_dry = metrics.surface_pressure_due_to_dry_air(
        water, pressure, vertical_coordinate
    )
    ps_dry_mean = area_weighted_mean(ps_dry)
    return ps_dry_mean.diff(dim=-1).abs().mean(dim=0)


def _layer_thickness(
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
    # Enforce min log(p) = 0 so that geopotential energy calculation is finite
    dlogp = torch.clamp(torch.log(pressure_at_interface), min=0.0).diff(dim=-1)
    return dlogp * RDGAS * tv / GRAVITY


def _height_at_interface(
    layer_thickness: torch.tensor, surface_height: torch.tensor
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
