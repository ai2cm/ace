import re
from types import MappingProxyType
from typing import Callable, List, Mapping, Union

import torch

from fme.core import metrics
from fme.core.constants import (
    GRAVITY,
    LATENT_HEAT_OF_VAPORIZATION,
    RDGAS,
    RVGAS,
    SPECIFIC_HEAT_OF_DRY_AIR_CONST_PRESSURE,
)
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.typing_ import TensorDict, TensorMapping

CLIMATE_FIELD_NAME_PREFIXES = MappingProxyType(
    {
        "specific_total_water": ["specific_total_water_"],
        "surface_pressure": ["PRESsfc", "PS"],
        "surface_height": ["HGTsfc"],
        "tendency_of_total_water_path_due_to_advection": [
            "tendency_of_total_water_path_due_to_advection"
        ],
        "latent_heat_flux": ["LHTFLsfc", "LHFLX"],
        "sensible_heat_flux": ["SHTFLsfc"],
        "precipitation_rate": ["PRATEsfc", "surface_precipitation_rate"],
        "sfc_down_sw_radiative_flux": ["DSWRFsfc"],
        "sfc_up_sw_radiative_flux": ["USWRFsfc"],
        "sfc_down_lw_radiative_flux": ["DLWRFsfc"],
        "sfc_up_lw_radiative_flux": ["ULWRFsfc"],
        "air_temperature": ["air_temperature_"],
    }
)


def natural_sort(alist: List[str]) -> List[str]:
    """Sort to alphabetical order but with numbers sorted
    numerically, e.g. a11 comes after a2. See [1] and [2].

    [1] https://stackoverflow.com/questions/11150239/natural-sorting
    [2] https://en.wikipedia.org/wiki/Natural_sort_order
    """

    def convert(text: str) -> Union[str, int]:
        if text.isdigit():
            return int(text)
        else:
            return text.lower()

    def alphanum_key(item: str) -> List[Union[str, int]]:
        return [convert(c) for c in re.split("([0-9]+)", item)]

    return sorted(alist, key=alphanum_key)


LEVEL_PATTERN = re.compile(r"_(\d+)$")


class ClimateData:
    """Container for climate data for accessing variables and providing
    torch.Tensor views on data with multiple vertical levels."""

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
            climate_field_name_prefixes: Mapping from field name prefixes (e.g.
                "specific_total_water_") to standardized prefixes, e.g. "PRESsfc" →
                "surface_pressure".
        """
        self._data = dict(climate_data)
        self._prefixes = climate_field_name_prefixes

    def _extract_levels(self, name: List[str]) -> torch.Tensor:
        for prefix in name:
            try:
                return self._extract_prefix_levels(prefix)
            except KeyError:
                pass
        raise KeyError(name)

    def _extract_prefix_levels(self, prefix: str) -> torch.Tensor:
        names = [
            field_name for field_name in self._data if field_name.startswith(prefix)
        ]

        levels = []
        for name in names:
            match = LEVEL_PATTERN.search(name)
            if match is None:
                raise ValueError(
                    f"Invalid field name {name}, is a prefix variable "
                    "but does not end in _(number)."
                )
            levels.append(int(match.group(1)))

        for i, level in enumerate(sorted(levels)):
            if i != level:
                raise KeyError(f"Missing level {i} in {prefix} levels {levels}.")

        if len(names) == 0:
            raise KeyError(prefix)

        names = natural_sort(names)
        return torch.stack([self._data[name] for name in names], dim=-1)

    def _get(self, name):
        for prefix in self._prefixes[name]:
            if prefix in self._data.keys():
                return self._get_prefix(prefix)
        raise KeyError(name)

    def _get_prefix(self, prefix):
        return self._data[prefix]

    def _set(self, name, value):
        for prefix in self._prefixes[name]:
            if prefix in self._data.keys():
                self._set_prefix(prefix, value)
                return
        raise KeyError(name)

    def _set_prefix(self, prefix, value):
        self._data[prefix] = value

    @property
    def data(self) -> TensorDict:
        """Mapping from field names to tensors."""
        return self._data

    @property
    def air_temperature(self) -> torch.Tensor:
        """Returns all vertical levels of air_temperature, e.g. a tensor of
        shape `(..., vertical_level)`."""
        prefix = self._prefixes["air_temperature"]
        return self._extract_levels(prefix)

    @property
    def specific_total_water(self) -> torch.Tensor:
        """Returns all vertical levels of specific total water, e.g. a tensor of
        shape `(..., vertical_level)`."""
        prefix = self._prefixes["specific_total_water"]
        return self._extract_levels(prefix)

    @property
    def surface_height(self) -> torch.Tensor:
        return self._get("surface_height")

    @property
    def surface_pressure(self) -> torch.Tensor:
        return self._get("surface_pressure")

    @surface_pressure.setter
    def surface_pressure(self, value: torch.Tensor):
        self._set("surface_pressure", value)

    def surface_pressure_due_to_dry_air(
        self, sigma_coordinates: SigmaCoordinates
    ) -> torch.Tensor:
        return metrics.surface_pressure_due_to_dry_air(
            self.specific_total_water,
            self.surface_pressure,
            sigma_coordinates.ak,
            sigma_coordinates.bk,
        )

    def total_water_path(self, sigma_coordinates: SigmaCoordinates) -> torch.Tensor:
        return metrics.vertical_integral(
            self.specific_total_water,
            self.surface_pressure,
            sigma_coordinates.ak,
            sigma_coordinates.bk,
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
        self, sigma_coordinates: SigmaCoordinates
    ) -> torch.Tensor:
        """
        Compute vertical height at layer log midpoints
        """
        pressure_interfaces = _pressure_at_interface(
            sigma_coordinates.ak, sigma_coordinates.bk, self.surface_pressure
        )
        layer_thickness = _layer_thickness(
            pressure_at_interface=pressure_interfaces,
            air_temperature=self.air_temperature,
            specific_total_water=self.specific_total_water,
        )
        height_at_interface = _height_at_interface(layer_thickness, self.surface_height)
        return (height_at_interface[..., :-1] * height_at_interface[..., 1:]) ** 0.5

    def moist_static_energy(self, sigma_coordinates: SigmaCoordinates) -> torch.Tensor:
        """
        Compute moist static energy.
        """
        # ACE does not currently prognose specific humidity, so here we closely
        # approximate this using specific total water (<0.01% effect on total MSE).
        return (
            self.air_temperature * SPECIFIC_HEAT_OF_DRY_AIR_CONST_PRESSURE
            + self.specific_total_water * LATENT_HEAT_OF_VAPORIZATION
            + self.height_at_log_midpoint(sigma_coordinates) * GRAVITY
        )


def compute_dry_air_absolute_differences(
    climate_data: ClimateData,
    area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
    sigma_coordinates: SigmaCoordinates,
) -> torch.Tensor:
    """
    Computes the absolute value of the dry air tendency of each time step.

    Args:
        climate_data: ClimateData object.
        area: Area of each grid cell as a [lat, lon] tensor, in m^2.
        sigma_coordinates: The sigma coordinates of the model.

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
        water,  # (sample, time, y, x, level)
        pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
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
    approximate this using specific total water."""
    tv = air_temperature * (1 + (RVGAS / RDGAS - 1.0) * specific_total_water)
    dlogp = torch.clamp(torch.log(pressure_at_interface), min=-100.0).diff(dim=-1)
    return dlogp * RDGAS * tv / GRAVITY


def _pressure_at_interface(
    ak: torch.tensor, bk: torch.tensor, surface_pressure: torch.tensor
) -> torch.Tensor:
    """
    Computes pressure at layer interfaces from sigma coefficients.
    Vertical coordinate is the last tensor dimension.
    """
    return torch.stack(
        [ak[i] + bk[i] * surface_pressure for i in range(ak.shape[-1])],
        dim=-1,
    )


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
