import dataclasses
import datetime

from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements


def _compute_n_steps_fast(
    atmosphere_timestep: datetime.timedelta,
    ocean_timestep: datetime.timedelta,
) -> int:
    """Compute and validate the number of atmosphere steps per ocean step."""
    if atmosphere_timestep > ocean_timestep:
        raise ValueError("Atmosphere timestep must be no larger than the ocean's.")
    n_steps_fast = ocean_timestep / atmosphere_timestep
    if n_steps_fast != int(n_steps_fast):
        raise ValueError(
            f"Expected atmosphere timestep {atmosphere_timestep} to be a "
            f"multiple of ocean timestep {ocean_timestep}."
        )
    return int(n_steps_fast)


def _validate_atmosphere_n_timesteps(
    atmosphere_timestep: datetime.timedelta,
    ocean_timestep: datetime.timedelta,
    ocean_requirements: DataRequirements,
    atmosphere_n_timesteps: int,
) -> None:
    """Validate that the initial value of the atmosphere data requirements
    n_timesteps_schedule matches the value derived from the ocean requirements
    and the timesteps.

    """
    n_steps_fast = _compute_n_steps_fast(
        atmosphere_timestep=atmosphere_timestep,
        ocean_timestep=ocean_timestep,
    )
    slow_n_steps = ocean_requirements.n_timesteps_schedule.get_value(0)
    expected_n_steps = (slow_n_steps - 1) * n_steps_fast + 1
    if atmosphere_n_timesteps != expected_n_steps:
        raise ValueError(
            f"Atmosphere dataset timestep is {atmosphere_timestep} and "
            f"ocean dataset timestep is {ocean_timestep}, "
            f"so we need {n_steps_fast} atmosphere steps for each of the "
            f"{slow_n_steps - 1} ocean steps, giving {expected_n_steps} total "
            f"timepoints (including IC) per sample, but "
            f"atmosphere dataset was configured to return "
            f"{atmosphere_n_timesteps} steps."
        )


@dataclasses.dataclass
class CoupledDataRequirements:
    ocean_timestep: datetime.timedelta
    ocean_requirements: DataRequirements
    atmosphere_timestep: datetime.timedelta
    atmosphere_requirements: DataRequirements

    def __post_init__(self):
        n_steps_fast = _compute_n_steps_fast(
            atmosphere_timestep=self.atmosphere_timestep,
            ocean_timestep=self.ocean_timestep,
        )
        atmosphere_n_timesteps = (
            self.atmosphere_requirements.n_timesteps_schedule.get_value(0)
        )
        _validate_atmosphere_n_timesteps(
            atmosphere_timestep=self.atmosphere_timestep,
            ocean_timestep=self.ocean_timestep,
            ocean_requirements=self.ocean_requirements,
            atmosphere_n_timesteps=atmosphere_n_timesteps,
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


@dataclasses.dataclass
class CoupledTrainDataRequirements:
    """Coupled data requirements for training, with separate atmosphere
    requirements for target variables (loaded for the loss horizon plus the
    initial condition) and forcing variables (loaded for the full rollout).

    The atmosphere data loader is expected to load each variable group at its
    own time horizon and then NaN-pad the shorter (target) group along the
    leading time dimension to match the longer (forcing) group.

    Trailing NaNs are never consumed by the loss (loss is skipped for
    non-optimized steps) or by the forward pass (forcings come from a disjoint
    variable set).

    Parameters:
        ocean_timestep: Ocean timestep size.
        ocean_requirements: Data requirements for the ocean.
        atmosphere_timestep: Atmosphere timestep size.
        atmosphere_target_requirements: Data requirements for the atmosphere
            target variables (short horizon).
        atmosphere_forcing_requirements: Data requirements for the atmosphere
            forcing variables (long horizon, governs the canonical sample
            length).
    """

    ocean_timestep: datetime.timedelta
    ocean_requirements: DataRequirements
    atmosphere_timestep: datetime.timedelta
    atmosphere_target_requirements: DataRequirements
    atmosphere_forcing_requirements: DataRequirements

    def __post_init__(self):
        n_steps_fast = _compute_n_steps_fast(
            atmosphere_timestep=self.atmosphere_timestep,
            ocean_timestep=self.ocean_timestep,
        )

        # check that the atmosphere forcing window is consistent with the ocean window
        atmosphere_forcing_n_timesteps = (
            self.atmosphere_forcing_requirements.n_timesteps_schedule.get_value(0)
        )
        _validate_atmosphere_n_timesteps(
            atmosphere_timestep=self.atmosphere_timestep,
            ocean_timestep=self.ocean_timestep,
            ocean_requirements=self.ocean_requirements,
            atmosphere_n_timesteps=atmosphere_forcing_n_timesteps,
        )

        # check that the atmosphere target window is no longer than the forcing window
        atmosphere_target_n_timesteps = (
            self.atmosphere_target_requirements.n_timesteps_schedule.get_value(0)
        )
        if atmosphere_target_n_timesteps > atmosphere_forcing_n_timesteps:
            raise ValueError(
                f"Atmosphere target requirements n_timesteps "
                f"({atmosphere_target_n_timesteps}) must be no larger than the "
                f"atmosphere forcing requirements n_timesteps "
                f"({atmosphere_forcing_n_timesteps})."
            )

        # check that the atmosphere target and forcing variable names are disjoint
        target_names = set(self.atmosphere_target_requirements.names)
        forcing_names = set(self.atmosphere_forcing_requirements.names)
        overlap = target_names.intersection(forcing_names)
        if overlap:
            raise ValueError(
                "Atmosphere target and forcing variable names must be disjoint, "
                f"but the following names appear in both: {sorted(overlap)}."
            )

        self._n_steps_fast = n_steps_fast

    @property
    def n_steps_fast(self) -> int:
        return self._n_steps_fast
