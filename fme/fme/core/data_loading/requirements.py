import dataclasses
from typing import List


@dataclasses.dataclass
class DataRequirements:
    names: List[str]
    # TODO: delete these when validation no longer needs them
    in_names: List[str]
    out_names: List[str]
    n_timesteps: int
