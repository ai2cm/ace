from types import MappingProxyType
from typing import Dict, Mapping

import torch

from fme.core import metrics
from fme.core.data_loading.typing import SigmaCoordinates

CLIMATE_FIELD_NAME_PREFIXES = MappingProxyType(
    {
        "specific_total_water": "specific_total_water_",
        "surface_pressure": "PRESsfc",
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

    def _get(self, name):
        return self._data[self._prefixes[name]]
