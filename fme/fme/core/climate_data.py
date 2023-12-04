from types import MappingProxyType
from typing import Dict, Mapping

import torch

from fme.core import metrics
from fme.core.constants import LATENT_HEAT_OF_VAPORIZATION
from fme.core.data_loading.typing import SigmaCoordinates

CLIMATE_FIELD_NAME_PREFIXES = MappingProxyType(
    {
        "specific_total_water": "specific_total_water_",
        "surface_pressure": "PRESsfc",
        "tendency_of_total_water_path_due_to_advection": "tendency_of_total_water_path_due_to_advection",  # noqa: E501
        "latent_heat_flux": "LHTFLsfc",
        "precipitation_rate": "PRATEsfc",
    }
)


class ClimateData:
    """Container for climate data for accessing variables and providing
    torch.Tensor views on data with multiple vertical levels."""

    def __init__(
        self,
        climate_data: Mapping[str, torch.Tensor],
        climate_field_name_prefixes: Mapping[str, str] = CLIMATE_FIELD_NAME_PREFIXES,
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

    def _extract_levels(self, prefix) -> torch.Tensor:
        names = [
            field_name for field_name in self._data if field_name.startswith(prefix)
        ]

        if len(names) == 0:
            raise KeyError(prefix)
        elif len(names) > 10:
            raise NotImplementedError("No support for > 10 vertical levels.")

        names = sorted(names)
        return torch.stack([self._data[name] for name in names], dim=-1)

    def _get(self, name):
        return self._data[self._prefixes[name]]

    @property
    def data(self) -> Dict[str, torch.Tensor]:
        """Mapping from field names to tensors."""
        return self._data

    @property
    def specific_total_water(self) -> torch.Tensor:
        """Returns all vertical levels of specific total water, e.g. a tensor of
        shape `(..., vertical_level)`."""
        prefix = self._prefixes["specific_total_water"]
        return self._extract_levels(prefix)

    @property
    def surface_pressure(self) -> torch.Tensor:
        return self._get("surface_pressure")

    @surface_pressure.setter
    def surface_pressure(self, value: torch.Tensor):
        self._data[self._prefixes["surface_pressure"]] = value

    def surface_pressure_due_to_dry_air(
        self, sigma_coordinates: SigmaCoordinates
    ) -> torch.Tensor:
        return metrics.surface_pressure_due_to_dry_air(
            self.specific_total_water,
            self.surface_pressure,
            sigma_coordinates.ak,
            sigma_coordinates.bk,
        )

    @property
    def precipitation_rate(self) -> torch.Tensor:
        """
        Precipitation rate in kg m-2 s-1.
        """
        return self._get("precipitation_rate")

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
        lhf = self.latent_heat_flux  # W/m^2
        # (W/m^2) / (J/kg) = (J s^-1 m^-2) / (J/kg) = kg/m^2/s
        return lhf / LATENT_HEAT_OF_VAPORIZATION

    @property
    def tendency_of_total_water_path_due_to_advection(self) -> torch.Tensor:
        """
        Tendency of total water path due to advection in kg m-2 s-1.
        """
        return self._get("tendency_of_total_water_path_due_to_advection")

    @tendency_of_total_water_path_due_to_advection.setter
    def tendency_of_total_water_path_due_to_advection(self, value: torch.Tensor):
        self._data[
            self._prefixes["tendency_of_total_water_path_due_to_advection"]
        ] = value


def compute_dry_air_absolute_differences(
    climate_data: ClimateData, area: torch.Tensor, sigma_coordinates: SigmaCoordinates
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
    return (
        metrics.weighted_mean(
            metrics.surface_pressure_due_to_dry_air(
                water,  # (sample, time, y, x, level)
                pressure,
                sigma_coordinates.ak,
                sigma_coordinates.bk,
            ),
            area,
            dim=(2, 3),
        )
        .diff(dim=-1)
        .abs()
        .mean(dim=0)
    )
