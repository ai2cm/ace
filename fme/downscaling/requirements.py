import dataclasses
from typing import List


@dataclasses.dataclass
class DataRequirements:
    fine_names: List[str]
    coarse_names: List[str]
    n_timesteps: int
