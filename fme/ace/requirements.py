import dataclasses


@dataclasses.dataclass
class PrognosticStateDataRequirements:
    """
    The requirements for the model's prognostic state.

    Parameters:
        names: Names of prognostic variables.
        n_timesteps: Number of consecutive timesteps that must be stored.
        n_ensembles: Number of IC ensemble members that must be stored.
    """

    names: list[str]
    n_timesteps: int
    n_ensemble: int | None = None


@dataclasses.dataclass
class DataRequirements:
    """
    ACE's requirements for batches (time windows) of loaded data.

    Parameters:
        names: Names of the variables to load.
        n_timesteps: Number of timesteps to load in each batch window.
    """

    names: list[str]
    n_timesteps: int


NullDataRequirements = DataRequirements(
    names=[],
    n_timesteps=0,
)
