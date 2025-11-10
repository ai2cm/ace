import copy
import dataclasses
import datetime

import torch
import xarray as xr

from fme.ace.requirements import DataRequirements
from fme.ace.stepper.insolation import CM4Insolation
from fme.core.coordinates import HorizontalCoordinates
from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class NameConfig:
    """Configuration for specifying a solar constant name.

    Parameters:
        name: name of a solar constant variable to load from data
            on disk; useful in the case that a time-varying solar
            constant is desired. The computed insolation will share
            the same dtype as the loaded solar constant.
    """

    name: str

    def get(self, tensors: TensorMapping) -> torch.tensor:
        return tensors[self.name]


@dataclasses.dataclass
class ValueConfig:
    """Configuration for specifying a solar constant value.

    Parameters:
        value: scalar solar constant value to use for all time.
        dtype: dtype for solar constant and resulting insolation.
    """

    value: float
    dtype: str = "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        try:
            torch_dtype = getattr(torch, self.dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype '{self.dtype}'")
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"Invalid dtype '{self.dtype}'")
        return torch_dtype

    def get(self, tensors: TensorMapping) -> torch.tensor:
        return torch.tensor(self.value, dtype=self.torch_dtype)


@dataclasses.dataclass
class InsolationConfig:
    """Configuration for computing insolation.

    Currently only supports computing the insolation as in GFDL's CM4
    model.

    Parameters:
        insolation_name: name to assign the computed insolation; must
            be present as an input to your model.
        solar_constant: configuration for setting the solar constant
            to a scalar value or loading a time-varying value from
            disk. Configure as a value to use the same scalar value
            for all time. Configure as a name to load a potentially
            time-varying value from disk. The computed insolation
            will share the same dtype as the solar constant.
        obliquity: angle of the axis of rotation of the Earth with
            the normal to the orbital plane in units of degrees.
        eccentricity: eccentricity of the orbit of the Earth.
        longitude_of_perhelion: orbital angle of perhelion in units
            of degrees, measured relative to the orbital position of
            the autumnal equinox in the Northern Hemisphere.

    Descriptions of the orbital parameters are paraphrased from `a
    PostScript-format technical document in GFDL's Flexible Modeling
    System repository <FMS_postscript_>`_. Definitions align with
    those in `Held (1982) <Held_1982_>`_, with the one minor difference
    that the ``longitude_of_perhelion`` in this case is defined with
    respect to the autumnal equinox rather than the vernal equinox.

    .. _FMS_postscript: https://github.com/NOAA-GFDL/FMS/blob/039d5f73fc4c7ce83117ca555f0b0761caf18e06/astronomy/astronomy.tech.ps
    .. _Held_1982: https://doi.org/10.1016/0019-1035(82)90135-X
    """

    insolation_name: str
    solar_constant: NameConfig | ValueConfig
    obliquity: float = 23.439
    eccentricity: float = 0.0167
    longitude_of_perhelion: float = 102.932

    def build(
        self,
        timestep: datetime.timedelta,
        horizontal_coordinates: HorizontalCoordinates,
    ) -> "Insolation":
        """Build an Insolation instance with the current configuration.

        Args:
            timestep: Timestep over which to average the insolation.
            horizontal_coordinates: Horizontal grid over which to compute
                the insolation.
        """
        return Insolation(self, timestep, horizontal_coordinates)

    def build_insolation_function(self) -> CM4Insolation:
        """Build the insolation function for the current configuration."""
        return CM4Insolation(
            self.obliquity, self.eccentricity, self.longitude_of_perhelion
        )

    def update_requirements(self, requirements: DataRequirements) -> DataRequirements:
        """Add or remove names from the requirements associated with the insolation.

        Args:
            requirements: The requirements to update.
        """
        names = copy.deepcopy(requirements.names)
        if self.insolation_name in names:
            names.remove(self.insolation_name)
            if isinstance(self.solar_constant, NameConfig):
                if self.solar_constant.name not in names:
                    names.append(self.solar_constant.name)
        return DataRequirements(names=names, n_timesteps=requirements.n_timesteps)


class Insolation:
    """A class orchestrating the computation of insolation.

    Parameters:
        config: Configuration for computing the insolation.
        timestep: Timestep over which to average the insolation.
        horizontal_coordinates: Horizontal grid over which to compute
            the insolation.
    """

    def __init__(
        self,
        config: InsolationConfig,
        timestep: datetime.timedelta,
        horizontal_coordinates: HorizontalCoordinates,
    ):
        self.config = config
        self.timestep = timestep
        self.horizontal_coordinates = horizontal_coordinates
        self.insolation_function = config.build_insolation_function()

    def compute(self, time: xr.DataArray, tensors: TensorMapping) -> TensorMapping:
        """Compute the insolation.

        Args:
            time: Times at which to compute the insolation.
            tensors: Dictionary of tensors to update with the insolation; this
                also may contain input data for computing the insolation like
                the solar constant.

        Returns:
            The tensor dictionary updated to include the insolation.
        """
        tensors = dict(tensors)  # Shallow copy to avoid mutating input.
        solar_constant = self.config.solar_constant.get(tensors)
        insolation = self.insolation_function(
            time, self.timestep, self.horizontal_coordinates, solar_constant
        )
        tensors[self.config.insolation_name] = insolation
        return tensors
