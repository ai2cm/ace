import dataclasses
from typing import List


@dataclasses.dataclass
class DataRequirements:
    """
    The requirements for batches (time windows) of loaded data.

    Parameters:
        names: Names of the variables to load.
        n_timesteps: Number of timesteps to load in each batch window.
    """

    names: List[str]
    n_timesteps: int
