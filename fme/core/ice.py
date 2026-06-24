import dataclasses
import datetime

from fme.core.typing_ import TensorDict, TensorMapping

from .prescriber import Prescriber


@dataclasses.dataclass
class IceConfig:
    """
    Configuration for determining sea ice concentration from an ice model.

    Parameters:
        ice_surface_temperature_name: Name of the ice surface skin temperature field.
        ocean_fraction_name: Name of the ocean fraction field.
        interpolate: If True, interpolate between ML-predicted ice concentration and
            ice-predicted ice concentration according to ocean_fraction. If False,
            only use ice-predicted ice concentration where ocean_fraction>=0.5.
    """

    ice_surface_temperature_name: str
    ocean_fraction_name: str
    interpolate: bool = False

    def build(
        self,
        in_names: list[str],
        out_names: list[str],
        timestep: datetime.timedelta,
    ) -> "Ice":
        if not (
            self.ice_surface_temperature_name in in_names
            and self.ice_surface_temperature_name in out_names
        ):
            raise ValueError(
                "To use a surface ice model, the ice surface temperature must be present"
                f" in_names and out_names, but {self.ice_surface_temperature_name}"
                f" is not."
            )
        return Ice(config=self, timestep=timestep)

    @property
    def forcing_names(self) -> list[str]:
        names = [self.ocean_fraction_name]
        names.append(self.ice_surface_temperature_name)
        return list(set(names))


class Ice:
    """Overwrite ice surface temperature with that predicted from some ice model."""

    def __init__(self, config: IceConfig, timestep: datetime.timedelta):
        """
        Args:
            config: Configuration for the surface ice model.
            timestep: Timestep of the model.
        """
        self.ocean_fraction_name = config.ocean_fraction_name
        self.ice_surface_temperature_name = config.ice_surface_temperature_name
        self.prescriber = Prescriber(
            prescribed_name=config.ice_surface_temperature_name,
            mask_name=config.ocean_fraction_name,
            mask_value=1,
            interpolate=config.interpolate,
        )
        self._forcing_names = config.forcing_names
        self.type = "prescribed"
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
            gen_data with ice surface temperature overwritten by ice model.
        """
        if self.type == "prescribed":
            next_step_temperature = target_data[self.ice_surface_temperature_name]
        else:
            raise NotImplementedError(f"Ice type={self.type} is not implemented")

        return self.prescriber(
            target_data,
            gen_data,
            {self.ice_surface_temperature_name: next_step_temperature},
        )

    @property
    def forcing_names(self) -> list[str]:
        """These are the variables required from the forcing data."""
        return self._forcing_names
