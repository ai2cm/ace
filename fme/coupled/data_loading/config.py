import dataclasses
from typing import Optional, Sequence

from fme.core.dataset.config import XarrayDataConfig
from fme.core.distributed import Distributed


@dataclasses.dataclass
class CoupledDatasetConfig:
    """
    Parameters:
        ocean: Configuration for the ocean dataset.
        atmosphere: Configuration for the atmosphere dataset.
    """

    ocean: XarrayDataConfig
    atmosphere: XarrayDataConfig

    @property
    def data_configs(self) -> Sequence[XarrayDataConfig]:
        return [self.ocean, self.atmosphere]


@dataclasses.dataclass
class CoupledDataLoaderConfig:
    """
    Parameters:
        dataset: A sequence of configurations each defining a coupled dataset
            to be loaded. This sequence of datasets will be concatenated.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        prefetch_factor: how many batches a single data worker will attempt to
            hold in host memory at a given time.
        strict_ensemble: Whether to enforce that the concatenated ensemble
            members have the same dimensions and coordinates.

    """

    dataset: Sequence[CoupledDatasetConfig]
    batch_size: int
    num_data_workers: int = 1
    prefetch_factor: Optional[int] = None
    strict_ensemble: bool = True

    def __post_init__(self):
        dist = Distributed.get_instance()
        if self.batch_size % dist.world_size != 0:
            raise ValueError(
                "batch_size must be divisible by the number of parallel "
                f"workers, got {self.batch_size} and {dist.world_size}"
            )
        self._zarr_engine_used = any(
            ds.engine == "zarr"
            for ds_coupled in self.dataset
            for ds in ds_coupled.data_configs
        )

    @property
    def zarr_engine_used(self) -> bool:
        """
        Whether the dataset uses the Zarr engine in any of its components.
        """
        return self._zarr_engine_used
