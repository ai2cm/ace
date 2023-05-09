import dataclasses
from typing import List


@dataclasses.dataclass
class DataLoaderParams:
    data_type: str
    batch_size: int
    num_data_workers: int
    dt: float
    n_history: int
    two_step_training: bool
    global_means_path: str
    global_stds_path: str
    time_means_path: str
    normalize: bool
    normalization: str
    in_names: List[str] = dataclasses.field(default_factory=list)
    out_names: List[str] = dataclasses.field(default_factory=list)
