import dataclasses
import datetime

from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements


def _compute_n_steps_fast(
    atmosphere_timestep: datetime.timedelta | None = None,
    ocean_timestep: datetime.timedelta| None = None,
    ice_timestep: datetime.timedelta | None = None,
) -> int:
    """Compute and validate the number of fast steps per slow step."""
    if atmosphere_timestep is None:
        if ice_timestep > ocean_timestep:
            raise ValueError("Ice timestep must be no larger than the ocean's.")
        n_steps_fast = ocean_timestep / ice_timestep
    elif ice_timestep is None:
        if atmosphere_timestep > ocean_timestep:
            raise ValueError("Atmosphere timestep must be no larger than the ocean's.")
        n_steps_fast = ocean_timestep / atmosphere_timestep
    elif ocean_timestep is None:
        if atmosphere_timestep > ice_timestep:
            raise ValueError("Atmosphere timestep must be no larger than the ice's.")
        n_steps_fast = ice_timestep / atmosphere_timestep
    else:
        if atmosphere_timestep > ocean_timestep:
            raise ValueError("Atmosphere timestep must be no larger than the ocean's.")
        n_steps_fast = ocean_timestep / atmosphere_timestep

    if n_steps_fast != int(n_steps_fast):
        raise ValueError(
            f"Expected fast timestep to be a "
            f"multiple of slow timestep."
        )
    return int(n_steps_fast)


def _validate_fast_n_timesteps(
    atmosphere_timestep: datetime.timedelta | None = None,
    ocean_timestep: datetime.timedelta | None = None,
    ice_timestep: datetime.timedelta | None = None,
    ocean_requirements: DataRequirements | None = None,
    ice_requirements: DataRequirements | None = None,
    fast_n_timesteps: int | None = None,
) -> None:
    """Validate that the initial value of the fast data requirements
    n_timesteps_schedule matches the value derived from the slow requirements
    and the timesteps.

    """
    n_steps_fast = _compute_n_steps_fast(
        atmosphere_timestep=atmosphere_timestep,
        ocean_timestep=ocean_timestep,
        ice_timestep=ice_timestep
    )
    if ocean_timestep is not None:
        slow_n_steps = ocean_requirements.n_timesteps_schedule.get_value(0)
        slow_dt = ocean_timestep
    else:
        slow_n_steps = ice_requirements.n_timesteps_schedule.get_value(0)
        slow_dt = ice_timestep
    expected_n_steps = (slow_n_steps - 1) * n_steps_fast + 1

    if atmosphere_timestep is not None:
        fast_dt = atmosphere_timestep
    else:
        fast_dt = ice_timestep

    if fast_n_timesteps != expected_n_steps:
        raise ValueError(
            f"Fast dataset timestep is {fast_dt} and "
            f"slow dataset timestep is {slow_dt}, "
            f"so we need {n_steps_fast} fast steps for each of the "
            f"{slow_n_steps - 1} slow steps, giving {expected_n_steps} total "
            f"timepoints (including IC) per sample, but "
            f"fast dataset was configured to return "
            f"{fast_n_timesteps} steps."
        )

@dataclasses.dataclass
class CoupledDataRequirements:
    ocean_timestep: datetime.timedelta | None = None
    ocean_requirements: DataRequirements | None = None
    ice_timestep: datetime.timedelta | None = None
    ice_requirements: DataRequirements | None = None
    atmosphere_timestep: datetime.timedelta | None = None
    atmosphere_requirements: DataRequirements | None = None

    def __post_init__(self):
        n_steps_fast = _compute_n_steps_fast(
            atmosphere_timestep=self.atmosphere_timestep,
            ocean_timestep=self.ocean_timestep,
            ice_timestep=self.ice_timestep,
        )

        if self.atmosphere is None:
            fast_n_timesteps = (
                self.ice_requirements.n_timesteps_schedule.get_value(0)
            )
        else:
            fast_n_timesteps = (
                self.atmosphere_requirements.n_timesteps_schedule.get_value(0)
            )

        _validate_fast_n_timesteps(
            atmosphere_timestep=self.atmosphere_timestep,
            ocean_timestep=self.ocean_timestep,
            ice_timestep=self.ice_timestep,
            ocean_requirements=self.ocean_requirements,
            ice_requirements=self.ice_requirements,
            fast_n_timesteps=fast_n_timesteps,
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

    ocean: PrognosticStateDataRequirements | None = None
    atmosphere: PrognosticStateDataRequirements | None = None
    ice: PrognosticStateDataRequirements | None = None


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

    ocean_timestep: datetime.timedelta | None = None
    ocean_requirements: DataRequirements | None = None
    atmosphere_timestep: datetime.timedelta | None = None
    atmosphere_target_requirements: DataRequirements | None = None
    atmosphere_forcing_requirements: DataRequirements | None = None
    ice_timestep: datetime.timedelta | None = None
    ice_target_requirements: DataRequirements | None = None
    ice_forcing_requirements: DataRequirements | None = None

    def __post_init__(self):
        n_steps_fast = _compute_n_steps_fast(
            atmosphere_timestep=self.atmosphere_timestep,
            ocean_timestep=self.ocean_timestep,
            ice_timestep=self.ice_timestep,
        )

        if self.atmosphere_timestep is None:
            # check that the ice forcing window is consistent with the ocean window
            forcing_n_timesteps = (
                self.ice_forcing_requirements.n_timesteps_schedule.get_value(0)
            )
            # check that the ice target window is no longer than the forcing window
            target_n_timesteps = (
                self.ice_target_requirements.n_timesteps_schedule.get_value(0)
            )
            # check that the ice target and forcing variable names are disjoint
            target_names = set(self.ice_target_requirements.names)
            forcing_names = set(self.ice_forcing_requirements.names)
        else:
            # check that the atmosphere forcing window is consistent with the ocean window
            forcing_n_timesteps = (
                self.atmosphere_forcing_requirements.n_timesteps_schedule.get_value(0)
            )
            # check that the atmosphere target window is no longer than the forcing window
            target_n_timesteps = (
                self.atmosphere_target_requirements.n_timesteps_schedule.get_value(0)
            )
            # check that the atmosphere target and forcing variable names are disjoint
            target_names = set(self.atmosphere_target_requirements.names)
            forcing_names = set(self.atmosphere_forcing_requirements.names)

        _validate_fast_n_timesteps(
            atmosphere_timestep=self.atmosphere_timestep,
            ocean_timestep=self.ocean_timestep,
            ice_timestep=self.ice_timestep,
            ocean_requirements=self.ocean_requirements,
            ice_requirements=self.ice_requirements,
            fast_n_timesteps=forcing_n_timesteps,
        )

        if target_n_timesteps > forcing_n_timesteps:
            raise ValueError(
                f"Fast target requirements n_timesteps "
                f"({target_n_timesteps}) must be no larger than the "
                f"fast forcing requirements n_timesteps "
                f"({forcing_n_timesteps})."
            )

        overlap = target_names.intersection(forcing_names)
        if overlap:
            raise ValueError(
                "Fast target and forcing variable names must be disjoint, "
                f"but the following names appear in both: {sorted(overlap)}."
            )

        self._n_steps_fast = n_steps_fast

    @property
    def n_steps_fast(self) -> int:
        return self._n_steps_fast
