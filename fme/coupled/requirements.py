import dataclasses
import datetime

from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements


@dataclasses.dataclass
class CoupledDataRequirements:
    ocean_timestep: datetime.timedelta
    ocean_requirements: DataRequirements
    atmosphere_timestep: datetime.timedelta
    atmosphere_requirements: DataRequirements

    def __post_init__(self):
        if self.atmosphere_timestep > self.ocean_timestep:
            raise ValueError("Atmosphere timestep must be no larger than the ocean's.")
        n_steps_fast = self.ocean_timestep / self.atmosphere_timestep
        if n_steps_fast != int(n_steps_fast):
            raise ValueError(
                f"Expected atmosphere timestep {self.atmosphere_timestep} to be a "
                f"multiple of ocean timestep {self.ocean_timestep}."
            )
        n_steps_fast = int(n_steps_fast)

        # check for misconfigured DataRequirements n_timesteps in the atmosphere
        slow_n_steps = self.ocean_requirements.n_timesteps
        fast_n_steps = (slow_n_steps - 1) * n_steps_fast + 1
        if self.atmosphere_requirements.n_timesteps != fast_n_steps:
            raise ValueError(
                f"Atmosphere dataset timestep is {self.atmosphere_timestep} and "
                f"ocean dataset timestep is {self.ocean_timestep}, "
                f"so we need {n_steps_fast} atmosphere steps for each of the "
                f"{slow_n_steps - 1} ocean steps, giving {fast_n_steps} total "
                "timepoints (including IC) per sample, but atmosphere dataset "
                f"was configured to return {self.atmosphere_requirements.n_timesteps} "
                "steps."
            )
        self._n_steps_fast = n_steps_fast

    @property
    def n_steps_fast(self) -> int:
        return self._n_steps_fast


@dataclasses.dataclass
class CoupledPrognosticStateDataRequirements:
    """
    The requirements for the model's prognostic state.

    """

    ocean: PrognosticStateDataRequirements
    atmosphere: PrognosticStateDataRequirements
