import dataclasses

from fme.core.dataset.schedule import IntSchedule


@dataclasses.dataclass
class PrognosticStateDataRequirements:
    """
    The requirements for the model's prognostic state.

    Parameters:
        names: Names of prognostic variables.
        n_timesteps: Number of consecutive timesteps that must be stored.
    """

    names: list[str]
    n_timesteps: int


@dataclasses.dataclass
class DataRequirements:
    """
    ACE's requirements for batches (time windows) of loaded data.

    Parameters:
        names: Names of the variables to load.
        n_timesteps: Number of timesteps to load in each batch window.
    """

    names: list[str]
    n_timesteps: int | IntSchedule

    @property
    def n_timesteps_schedule(self) -> IntSchedule:
        if isinstance(self.n_timesteps, IntSchedule):
            return self.n_timesteps
        return IntSchedule(start_value=self.n_timesteps, milestones=[])


NullDataRequirements = DataRequirements(
    names=[],
    n_timesteps=0,
)
