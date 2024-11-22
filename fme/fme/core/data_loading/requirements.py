import dataclasses
from typing import List


@dataclasses.dataclass
class DataRequirements:
    """
    The requirements for batches (time windows) of loaded data.

    Attributes:
        names: Names of the variables to load.
        n_timesteps: Number of timesteps to load in each batch window.
    """

    names: List[str]
    n_timesteps: int


@dataclasses.dataclass
class PrognosticStateDataRequirements:
    """
    The requirements for the model's prognostic state.

    Attributes:
        names: Names of prognostic variables.
        n_timesteps: Number of consecutive timesteps that must be stored.
    """

    names: List[str]
    n_timesteps: int
