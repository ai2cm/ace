import dataclasses
from collections.abc import Mapping, Sequence

from fme.ace.data_loading.augmentation import AugmentationConfig
from fme.core.dataset.config import XarrayDataConfig
from fme.core.distributed import Distributed


@dataclasses.dataclass
class DataLoaderConfig:
    """
    Parameters:
        dataset: A sequence of configurations each defining a dataset
            to be loaded. This sequence of datasets will be concatenated.
            It could also be a mapping of dataset names to sequences of
            configurations, in which each value in the mapping is a
            different source of data to be merged. For example:

            .. code-block:: yaml

                dataset:
                    source1:
                    - data_path: ...
                    source2:
                    - data_path: ...

            If multiple sources contain the same data variable, the version
            from the first source is loaded and other sources are ignored.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        prefetch_factor: how many batches a single data worker will attempt to
            hold in host memory at a given time.
        strict_ensemble: Whether to enforce that the datasets to be concatened
            have the same dimensions and coordinates.
        augmentation: Configuration for data augmentation.
    """  # noqa: D415

    dataset: Sequence[XarrayDataConfig] | Mapping[str, Sequence[XarrayDataConfig]]
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
        self._zarr_engine_used = False
        if isinstance(self.dataset, Sequence):
            self._zarr_engine_used = any(ds.engine == "zarr" for ds in self.dataset)
        elif isinstance(self.dataset, Mapping):
            self._zarr_engine_used = any(
                ds.engine == "zarr"
                for ds_list in self.dataset.values()
                for ds in ds_list
            )

    @property
    def zarr_engine_used(self) -> bool:
        """
        Whether any of the configured datasets are using the Zarr engine.
        """
        return self._zarr_engine_used
