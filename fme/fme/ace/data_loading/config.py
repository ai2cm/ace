import dataclasses
from typing import Optional, Sequence

from fme.core.dataset.config import XarrayDataConfig
from fme.core.distributed import Distributed


@dataclasses.dataclass
class DataLoaderConfig:
    """
    Parameters:
        dataset: A sequence of configurations each defining a dataset
            to be loaded. This sequence of datasets will be concatenated.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        prefetch_factor: how many batches a single data worker will attempt to
            hold in host memory at a given time.
        strict_ensemble: Whether to enforce that the datasets to be concatened
            have the same dimensions and coordinates.
    """

    dataset: Sequence[XarrayDataConfig]
    batch_size: int
    num_data_workers: int = 0
    prefetch_factor: Optional[int] = None
    strict_ensemble: bool = True

    def __post_init__(self):
        dist = Distributed.get_instance()
        if self.batch_size % dist.world_size != 0:
            raise ValueError(
                "batch_size must be divisible by the number of parallel "
                f"workers, got {self.batch_size} and {dist.world_size}"
            )
