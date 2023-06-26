import dataclasses
from typing import Optional


@dataclasses.dataclass
class DataLoaderParams:
    """
    Attributes:
        data_path: Path to the data.
        data_type: Type of data to load.
        batch_size: Batch size.
        num_data_workers: Number of parallel data workers.
        n_samples: Number of samples to load, starting at the beginning of the data.
            If None, load all samples.
    """

    data_path: str
    data_type: str
    batch_size: int
    num_data_workers: int
    n_samples: Optional[int] = None

    def __post_init__(self):
        if self.n_samples is not None and self.batch_size > self.n_samples:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be less than or equal to "
                f"n_samples ({self.n_samples}) or no batches would be produced"
            )
