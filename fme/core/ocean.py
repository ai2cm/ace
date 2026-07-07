import dataclasses
import datetime
from typing import Protocol

import torch

from fme.core.typing_ import TensorDict, TensorMapping

from .atmosphere_data import AtmosphereData
from .constants import DENSITY_OF_WATER, SPECIFIC_HEAT_OF_WATER
from .prescriber import Prescriber


@dataclasses.dataclass
class SlabOceanConfig:
    """
    Configuration for a slab ocean model.

    Parameters:
        mixed_layer_depth_name: Name of the mixed layer depth field.
        q_flux_name: Name of the heat flux field.
    """

    mixed_layer_depth_name: str
    q_flux_name: str

    @property
    def names(self) -> list[str]:
        return [self.mixed_layer_depth_name, self.q_flux_name]


class SurfaceTemperature(Protocol):
    """Computes the next-step sea surface temperature for an :class:`Ocean`.

    Each ocean model (prescribed or slab) is a self-contained callable object
    bundling the field names and operators it needs, so :class:`Ocean` applies
    it without reading any config fields itself.
    """

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> torch.Tensor: ...


@dataclasses.dataclass
class PrescribedSurfaceTemperature:
    """Next-step surface temperature taken directly from the target data."""

    surface_temperature_name: str

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> torch.Tensor:
        return target_data[self.surface_temperature_name]


@dataclasses.dataclass
class SlabSurfaceTemperature:
    """Next-step surface temperature from a slab ocean mixed-layer tendency."""

    surface_temperature_name: str
    q_flux_name: str
    mixed_layer_depth_name: str
    timestep: datetime.timedelta

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> torch.Tensor:
        temperature_tendency = mixed_layer_temperature_tendency(
            AtmosphereData(gen_data).net_surface_energy_flux_without_frozen_precip,
            target_data[self.q_flux_name],
            target_data[self.mixed_layer_depth_name],
        )
        return (
            input_data[self.surface_temperature_name]
            + temperature_tendency * self.timestep.total_seconds()
        )


@dataclasses.dataclass
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
        in_names: list[str],
        out_names: list[str],
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
            surface_temperature = SlabSurfaceTemperature(
                surface_temperature_name=self.surface_temperature_name,
                q_flux_name=self.slab.q_flux_name,
                mixed_layer_depth_name=self.slab.mixed_layer_depth_name,
                timestep=timestep,
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
