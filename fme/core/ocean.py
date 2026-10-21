import abc
import dataclasses
import datetime
from collections.abc import Collection
from typing import Literal

import torch

from fme.core.typing_ import TensorDict, TensorMapping

from .atmosphere_data import ATMOSPHERE_FIELD_NAME_PREFIXES, AtmosphereData
from .constants import DENSITY_OF_WATER, SPECIFIC_HEAT_OF_WATER
from .prescriber import Prescriber


@dataclasses.dataclass(frozen=True)
class SlabOceanConfig:
    """
    Configuration for a slab ocean model.

    Parameters:
        mixed_layer_depth_name: Name of the mixed layer depth field.
        q_flux_name: Name of the heat flux field.
        surface_flux_source: Where the net surface energy flux driving the slab
            comes from. "prediction" uses the model's predicted surface fluxes;
            "data" reads the same flux variables from the forcing data instead
            (e.g. to drive the slab with the reference model's fluxes).
    """

    mixed_layer_depth_name: str
    q_flux_name: str
    surface_flux_source: Literal["prediction", "data"] = "prediction"

    def __post_init__(self):
        if self.surface_flux_source not in ("prediction", "data"):
            raise ValueError(
                "surface_flux_source must be 'prediction' or 'data', got "
                f"'{self.surface_flux_source}'"
            )

    @property
    def names(self) -> list[str]:
        return [self.mixed_layer_depth_name, self.q_flux_name]


class SurfaceTemperature(abc.ABC):
    """Computes the next-step sea surface temperature for an :class:`Ocean`.

    Each ocean model (prescribed or slab) is a self-contained callable object
    bundling the field names and operators it needs, so :class:`Ocean` applies
    it without reading any config fields itself.
    """

    @abc.abstractmethod
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> torch.Tensor: ...


@dataclasses.dataclass(frozen=True)
class PrescribedSurfaceTemperature(SurfaceTemperature):
    """Next-step surface temperature taken directly from the target data."""

    surface_temperature_name: str

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> torch.Tensor:
        return target_data[self.surface_temperature_name]


@dataclasses.dataclass(frozen=True)
class SlabOceanSurfaceTemperature(SurfaceTemperature):
    """Next-step surface temperature from a slab ocean mixed-layer tendency."""

    surface_temperature_name: str
    q_flux_name: str
    mixed_layer_depth_name: str
    timestep: datetime.timedelta
    surface_flux_source: Literal["prediction", "data"] = "prediction"

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> torch.Tensor:
        flux_data = (
            gen_data if self.surface_flux_source == "prediction" else target_data
        )
        temperature_tendency = mixed_layer_temperature_tendency(
            AtmosphereData(flux_data).net_surface_energy_flux_without_frozen_precip,
            target_data[self.q_flux_name],
            target_data[self.mixed_layer_depth_name],
        )
        return (
            input_data[self.surface_temperature_name]
            + temperature_tendency * self.timestep.total_seconds()
        )


@dataclasses.dataclass(frozen=True)
class OceanConfig:
    """
    Configuration for determining sea surface temperature from an ocean model.

    Parameters:
        surface_temperature_name: Name of the sea surface temperature field.
        ocean_fraction_name: Name of the ocean fraction field.
        interpolate: If True, interpolate between ML-predicted surface temperature and
            ocean-predicted surface temperature according to ocean_fraction. If False,
            only use ocean-predicted surface temperature where ocean_fraction>=0.5.
        slab: If provided, use a slab ocean model to predict surface temperature.
    """

    surface_temperature_name: str
    ocean_fraction_name: str
    interpolate: bool = False
    slab: SlabOceanConfig | None = None

    def build(
        self,
        in_names: Collection[str],
        out_names: Collection[str],
        timestep: datetime.timedelta,
    ) -> "Ocean":
        if not (
            self.surface_temperature_name in in_names
            and self.surface_temperature_name in out_names
        ):
            raise ValueError(
                "To use a surface ocean model, the surface temperature must be present"
                f" in_names and out_names, but {self.surface_temperature_name} is not."
            )
        # resolve the data surface flux names now so a missing one fails at build
        self.target_names(out_names)
        return self._build(timestep)

    def _build(self, timestep: datetime.timedelta) -> "Ocean":
        prescriber = Prescriber(
            prescribed_name=self.surface_temperature_name,
            mask_name=self.ocean_fraction_name,
            mask_value=1,
            interpolate=self.interpolate,
        )
        surface_temperature: SurfaceTemperature
        if self.slab is None:
            surface_temperature = PrescribedSurfaceTemperature(
                self.surface_temperature_name
            )
        else:
            surface_temperature = SlabOceanSurfaceTemperature(
                surface_temperature_name=self.surface_temperature_name,
                q_flux_name=self.slab.q_flux_name,
                mixed_layer_depth_name=self.slab.mixed_layer_depth_name,
                timestep=timestep,
                surface_flux_source=self.slab.surface_flux_source,
            )
        return Ocean(
            surface_temperature=surface_temperature,
            prescriber=prescriber,
            forcing_names=self.forcing_names,
            surface_temperature_name=self.surface_temperature_name,
            ocean_fraction_name=self.ocean_fraction_name,
        )

    @property
    def is_slab(self) -> bool:
        """Whether this config uses a slab ocean model."""
        return self.slab is not None

    @property
    def forcing_names(self) -> list[str]:
        names = [self.ocean_fraction_name]
        if self.slab is None:
            names.append(self.surface_temperature_name)
        else:
            names.extend(self.slab.names)
        return list(set(names))

    def target_names(self, out_names: Collection[str]) -> list[str]:
        """Names the ocean reads from the next-step (target) data.

        These are the forcing names plus, for a slab driven by data surface
        fluxes, the surface energy flux variables among ``out_names``. The fluxes
        are kept out of :attr:`forcing_names`, which steps add to their input
        names: as model outputs they would then become prognostic.

        Args:
            out_names: Output names of the model, used to pick which name each
                surface flux has in the data.
        """
        names = self.forcing_names
        if self.slab is not None and self.slab.surface_flux_source == "data":
            names = names + _surface_energy_flux_names(out_names)
        return names


# The AtmosphereData quantities summed by
# AtmosphereData.net_surface_energy_flux_without_frozen_precip.
_NET_SURFACE_ENERGY_FLUX_QUANTITIES = (
    "sfc_down_lw_radiative_flux",
    "sfc_up_lw_radiative_flux",
    "sfc_down_sw_radiative_flux",
    "sfc_up_sw_radiative_flux",
    "latent_heat_flux",
    "sensible_heat_flux",
)


def _surface_energy_flux_names(out_names: Collection[str]) -> list[str]:
    """Data names of the net surface energy flux terms, as named in out_names."""
    names = []
    for quantity in _NET_SURFACE_ENERGY_FLUX_QUANTITIES:
        candidates = ATMOSPHERE_FIELD_NAME_PREFIXES[quantity]
        matches = [name for name in candidates if name in out_names]
        if len(matches) == 0:
            raise ValueError(
                f"A slab ocean with surface_flux_source='data' needs the {quantity}"
                f" as a model output (one of {candidates}), but none is in out_names."
            )
        names.append(matches[0])
    return names


class Ocean:
    """Overwrite sea surface temperature with that predicted from some ocean model."""

    def __init__(
        self,
        surface_temperature: SurfaceTemperature,
        prescriber: Prescriber,
        forcing_names: list[str],
        surface_temperature_name: str,
        ocean_fraction_name: str,
    ):
        """
        Args:
            surface_temperature: Computes the next-step surface temperature.
            prescriber: Overwrites the surface temperature in the ocean region.
            forcing_names: Variables required from the forcing data.
            surface_temperature_name: Name of the sea surface temperature field.
            ocean_fraction_name: Name of the ocean fraction field.
        """
        self._surface_temperature = surface_temperature
        self.prescriber = prescriber
        self._forcing_names = forcing_names
        self.surface_temperature_name = surface_temperature_name
        self.ocean_fraction_name = ocean_fraction_name

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> TensorDict:
        """
        Args:
            input_data: Denormalized input data for current step.
            gen_data: Denormalized output data for current step.
            target_data: Denormalized data that includes mask and forcing data. Assumed
                to correspond to the same time step as gen_data.

        Returns:
            gen_data with sea surface temperature overwritten by ocean model.
        """
        next_step_temperature = self._surface_temperature(
            input_data, gen_data, target_data
        )
        return self.prescriber(
            target_data,
            gen_data,
            {self.surface_temperature_name: next_step_temperature},
        )

    @property
    def forcing_names(self) -> list[str]:
        """These are the variables required from the forcing data."""
        return self._forcing_names


def mixed_layer_temperature_tendency(
    f_net: torch.Tensor,
    q_flux: torch.Tensor,
    depth: torch.Tensor,
    density=DENSITY_OF_WATER,
    specific_heat=SPECIFIC_HEAT_OF_WATER,
) -> torch.Tensor:
    """
    Args:
        f_net: Net surface energy flux in W/m^2.
        q_flux: Convergence of ocean heat transport in W/m^2.
        depth: Mixed layer depth in m.
        density (optional): Density of water in kg/m^3.
        specific_heat (optional): Specific heat of water in J/kg/K.

    Returns:
        Temperature tendency of mixed layer in K/s.
    """
    return (f_net + q_flux) / (density * depth * specific_heat)
