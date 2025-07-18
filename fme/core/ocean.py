import dataclasses
import datetime

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
        return Ocean(config=self, timestep=timestep)

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

    def __init__(self, config: OceanConfig, timestep: datetime.timedelta):
        """
        Args:
            config: Configuration for the surface ocean model.
            timestep: Timestep of the model.
        """
        self.surface_temperature_name = config.surface_temperature_name
        self.ocean_fraction_name = config.ocean_fraction_name
        self.prescriber = Prescriber(
            prescribed_name=config.surface_temperature_name,
            mask_name=config.ocean_fraction_name,
            mask_value=1,
            interpolate=config.interpolate,
        )
        self._forcing_names = config.forcing_names
        if config.slab is None:
            self.type = "prescribed"
        else:
            self.type = "slab"
            self.mixed_layer_depth_name = config.slab.mixed_layer_depth_name
            self.q_flux_name = config.slab.q_flux_name
        self.timestep = timestep

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
        if self.type == "prescribed":
            next_step_temperature = target_data[self.surface_temperature_name]
        elif self.type == "slab":
            temperature_tendency = mixed_layer_temperature_tendency(
                AtmosphereData(gen_data).net_surface_energy_flux_without_frozen_precip,
                target_data[self.q_flux_name],
                target_data[self.mixed_layer_depth_name],
            )
            next_step_temperature = (
                input_data[self.surface_temperature_name]
                + temperature_tendency * self.timestep.total_seconds()
            )
        else:
            raise NotImplementedError(f"Ocean type={self.type} is not implemented")

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
