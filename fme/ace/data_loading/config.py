import dataclasses
import warnings
from collections.abc import Sequence

from fme.ace.data_loading.augmentation import AugmentationConfig
from fme.core.dataset.config import (
    ConcatDatasetConfig,
    MergeDatasetConfig,
    XarrayDataConfig,
)
from fme.core.distributed import Distributed


@dataclasses.dataclass
class DataLoaderConfig:
    """
    Parameters:
        dataset: Could be a single dataset configuration,
            or a sequence of datasets to be concatenated using the keyword `concat`,
            or datasets from different sources to be merged using the keyword `merge`.
            For backwards compatibility, it can also be a sequence of
            datasets, which will be concatenated.
            During `merge`, if multiple datasets contain the same data variable,
            the version from the first source is loaded and other sources are ignored.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        prefetch_factor: how many batches a single data worker will attempt to
            hold in host memory at a given time.
        strict_ensemble: Whether to enforce that the datasets to be concatened
            have the same dimensions and coordinates.
        augmentation: Configuration for data augmentation.
    """  # noqa: D415

    dataset: (
        ConcatDatasetConfig
        | MergeDatasetConfig
        | XarrayDataConfig
        | Sequence[XarrayDataConfig]
    )
    batch_size: int
    num_data_workers: int = 0
    prefetch_factor: int | None = None
    strict_ensemble: bool = True
    augmentation: AugmentationConfig = dataclasses.field(
        default_factory=lambda: AugmentationConfig()
    )

    def __post_init__(self):
        dist = Distributed.get_instance()
        if self.batch_size % dist.world_size != 0:
            raise ValueError(
                "batch_size must be divisible by the number of parallel "
                f"workers, got {self.batch_size} and {dist.world_size}"
            )
        # TODO: remove following backwards compatibility code in a future release
        if isinstance(self.dataset, Sequence):
            warnings.warn(
                "Dataset list format is deprecated. "
                "Use `concat` to specify concatenating datasets.",
                DeprecationWarning,
            )
            self.dataset = ConcatDatasetConfig(concat=self.dataset)
        self._zarr_engine_used = self.dataset.zarr_engine_used

    @property
    def zarr_engine_used(self) -> bool:
        """
        Whether any of the configured datasets are using the Zarr engine.
        """
        return self._zarr_engine_used
