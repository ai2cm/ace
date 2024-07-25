import dataclasses
from typing import List


@dataclasses.dataclass
class DataRequirements:
    names: List[str]
    n_timesteps: int
