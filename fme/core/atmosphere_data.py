from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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

if TYPE_CHECKING:
    from fme.core.corrector.output import CorrectorOutput

ATMOSPHERE_FIELD_NAME_PREFIXES = {
    "specific_total_water": ["specific_total_water_", "STW_"],
    "surface_pressure": ["PRESsfc", "PS"],
    "surface_height": ["HGTsfc"],
    "surface_geopotential": ["PHIS"],
    "tendency_of_total_water_path_due_to_advection": [
        "tendency_of_total_water_path_due_to_advection",
        "DTENDTTW",
    ],
    "latent_heat_flux": ["LHTFLsfc", "LHFLX"],
    "sensible_heat_flux": ["SHTFLsfc", "SHFLX"],
    "precipitation_rate": ["PRATEsfc", "surface_precipitation_rate", "PRECT"],
    "sfc_down_sw_radiative_flux": ["DSWRFsfc", "FSDS"],
    "sfc_up_sw_radiative_flux": ["USWRFsfc", "surface_upward_shortwave_flux", "FSUS"],
    "sfc_down_lw_radiative_flux": ["DLWRFsfc", "FLDS"],
    "sfc_up_lw_radiative_flux": ["ULWRFsfc", "surface_upward_longwave_flux", "FLUS"],
    "toa_up_lw_radiative_flux": ["ULWRFtoa", "FLUT"],
    "toa_up_sw_radiative_flux": [
        "USWRFtoa",
        "top_of_atmos_upward_shortwave_flux",
        "FSUTOA",
    ],
    "toa_down_sw_radiative_flux": ["DSWRFtoa", "SOLIN"],
    "air_temperature": ["air_temperature_", "T_"],
    "frozen_precipitation_rate": ["total_frozen_precipitation_rate"],
    "eastward_wind_at_10m": ["UGRD10m", "Uat10m"],
    "northward_wind_at_10m": ["VGRD10m", "Vat10m"],
}


@runtime_checkable
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
        # Concrete data keys written through this instance's ``set_*`` methods.
        self._modified_keys: set[str] = set()
        # CorrectorOutput wrapping (set by ``for_correction``).
        self._corrector_output: CorrectorOutput | None = None
        self._pending_deltas: TensorDict = {}
        self._pending_diagnosed: TensorDict = {}

    @classmethod
    def for_correction(
        cls,
        accumulated_output: CorrectorOutput,
        vertical_coordinate: HasAtmosphereVerticalIntegral | None = None,
        atmosphere_field_name_prefixes: Mapping[str, list[str]] | None = None,
    ) -> AtmosphereData:
        """Create an ``AtmosphereData`` for applying corrections.

        Reads go through the accumulated output's ``corrected`` dict.
        Corrections are tracked internally and applied to the
        ``CorrectorOutput`` by calling :meth:`result`.
        """
        obj = cls(
            accumulated_output.corrected,
            vertical_coordinate,
            atmosphere_field_name_prefixes,
        )
        obj._corrector_output = accumulated_output
        return obj

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
        self._modified_keys.add(prefix)

    @property
    def modified_data(self) -> TensorDict:
        """Return the data keys written through this instance's setters.

        The returned tensors are references into this instance's data (not
        clones).
        """
        return {key: self._data[key] for key in self._modified_keys}

    # ---- correction verbs (require ``for_correction``) --------------------

    def _correct(self, name: str, value: torch.Tensor) -> None:
        """Record a corrective delta for *name* (a standard name)."""
        for prefix in self._prefix_map[name]:
            if prefix in self._data:
                self._correct_prefix(prefix, value)
                return
        raise KeyError(name)

    def _correct_prefix(self, prefix: str, value: torch.Tensor) -> None:
        """Record a corrective delta for a concrete data key."""
        assert self._corrector_output is not None
        original = self._corrector_output.corrected[prefix]
        self._pending_deltas[prefix] = value - original
        self._data[prefix] = value

    def correct_surface_pressure(self, value: torch.Tensor) -> None:
        self._correct("surface_pressure", value)

    def correct_precipitation_rate(self, value: torch.Tensor) -> None:
        self._correct("precipitation_rate", value)

    def correct_evaporation_rate(self, value: torch.Tensor) -> None:
        self._correct("latent_heat_flux", value * LATENT_HEAT_OF_VAPORIZATION)

    def correct_tendency_of_total_water_path_due_to_advection(
        self, value: torch.Tensor
    ) -> None:
        self._correct("tendency_of_total_water_path_due_to_advection", value)

    def correct_frozen_precipitation_rate(self, value: torch.Tensor) -> None:
        self._correct("frozen_precipitation_rate", value)

    def _diagnose(self, name: str, value: torch.Tensor) -> None:
        """Record a diagnosis (wholesale replacement) for *name*."""
        for prefix in self._prefix_map[name]:
            if prefix in self._data:
                self._diagnose_prefix(prefix, value)
                return
        raise KeyError(name)

    def _diagnose_prefix(self, prefix: str, value: torch.Tensor) -> None:
        """Record a diagnosis for a concrete data key."""
        assert self._corrector_output is not None
        self._pending_diagnosed[prefix] = value
        self._data[prefix] = value

    def diagnose_tendency_of_total_water_path_due_to_advection(
        self, value: torch.Tensor
    ) -> None:
        self._diagnose("tendency_of_total_water_path_due_to_advection", value)

    def diagnose_frozen_precipitation_rate(self, value: torch.Tensor) -> None:
        self._diagnose("frozen_precipitation_rate", value)

    def diagnose_all_levels(self, standard_name: str, value: torch.Tensor) -> None:
        """Record a diagnosis for a multi-level (Stacker) variable."""
        names = self.get_all_vertical_level_names(standard_name)
        for k, name in enumerate(names):
            self._diagnose_prefix(name, value[..., k])

    def correct_all_levels(self, standard_name: str, value: torch.Tensor) -> None:
        """Apply corrective deltas for a multi-level (Stacker) variable.

        Args:
            standard_name: The standard name (e.g. ``"air_temperature"``).
            value: Tensor with shape ``(..., n_levels)``.
        """
        names = self.get_all_vertical_level_names(standard_name)
        for k, name in enumerate(names):
            self._correct_prefix(name, value[..., k])

    def result(self) -> CorrectorOutput:
        """Return the ``CorrectorOutput`` with all pending corrections applied.

        Raises ``AssertionError`` when called on an instance not created via
        :meth:`for_correction`.
        """
        from fme.core.corrector.output import CorrectorOutput  # noqa: F811, F401

        assert self._corrector_output is not None
        return self._corrector_output.apply_correction(
            diagnosed=self._pending_diagnosed,
            deltas=self._pending_deltas,
        )

    # ---- read accessors ---------------------------------------------------

    def _get(self, name):
        for prefix in self._prefix_map[name]:
            if prefix in self.data.keys():
                return self._get_prefix(prefix)
        raise KeyError(name)

    def get_all_vertical_level_names(self, standard_name: str) -> list[str]:
        """Return names of all vertical levels for a given standard name."""
        return self._stacker.get_all_level_names(standard_name, self.data)

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

    def set_frozen_precipitation_rate(self, value: torch.Tensor):
        self._set("frozen_precipitation_rate", value)

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
    # Clamp the minimum pressure to 1 Pa to ensure that log(p) is finite and
    # greater than or equal to 0.0. The ERA5 data has a TOA pressure of 0.0
    # Pa which causes issues otherwise. It is important to clamp and then take
    # the log rather than vice versa to ensure that we can backpropagate through
    # this function in circumstances where clamping is necessary.
    dlogp = torch.log(torch.clamp(pressure_at_interface, min=1.0)).diff(dim=-1)
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
