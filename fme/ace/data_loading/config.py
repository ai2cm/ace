import dataclasses
import warnings
from collections.abc import Sequence

import torch

from fme.ace.data_loading.augmentation import AugmentationConfig
from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.merged import MergeDatasetConfig
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.distributed import Distributed


@dataclasses.dataclass
class DataLoaderConfig:
    r"""
    Configuration for a data loader for training/validation.

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
        augmentation: Configuration for data augmentation.
        sample_with_replacement: If provided, the dataset will be
            sampled randomly with replacement to the given size each period,
            instead of retrieving each sample once (either shuffled or not).
        time_buffer: How many more continuous timesteps to load in memory than the
            required number of timesteps for a single batch. Setting this to greater
            than 0 should improve data loading performance, however, it also decreases
            the independence of subsequent batches if shuffled batches are desired.

    Note:
        Setting `time_buffer` to a value greater than 0 results in pre-loading
        samples of length `time_buffer + n_timesteps_required`, where
        `n_timesteps_required` is the number of timesteps required for training
        the model (initial condition(s) plus forward step(s)). These pre-loaded samples
        become a window from which samples of the required length are drawn without
        replacement. The windows will overlap by an amount such that no samples are
        skipped, with exception of the last window, which is dropped if incomplete.
        This is useful for improving data loading throughput and reducing the number of
        reads. There must be enough pre-loaded samples in the dataset to produce at
        least one batch at the configured batch size. Independent data will be seen
        every `time_buffer + 1` batches, i.e., this is the number of samples in each
        pre-loaded window.
    """

    dataset: (
        ConcatDatasetConfig
        | MergeDatasetConfig
        | XarrayDataConfig
        | Sequence[XarrayDataConfig]
    )
    batch_size: int
    num_data_workers: int = 0
    prefetch_factor: int | None = None
    augmentation: AugmentationConfig = dataclasses.field(
        default_factory=lambda: AugmentationConfig()
    )
    sample_with_replacement: int | None = None
    time_buffer: int = 0

    def get_dataset(
        self,
        names: Sequence[str],
        n_timesteps: int,
    ) -> tuple[torch.utils.data.Dataset, DatasetProperties]:
        if isinstance(self.dataset, Sequence):
            raise RuntimeError(
                "Dataset list should have been replaced by a ConcatDatasetConfig "
                "at init time, perhaps the attribute was set post-init?"
            )
        return self.dataset.build(names, n_timesteps)

    def __post_init__(self):
        dist = Distributed.get_instance()
        dist.check_local_batch_size(self.batch_size)
        # TODO: remove following backwards compatibility code in a future release
        if isinstance(self.dataset, Sequence):
            warnings.warn(
                "Dataset list format is deprecated. "
                "Use `concat` to specify concatenating datasets.",
                DeprecationWarning,
            )
            self.dataset = ConcatDatasetConfig(concat=self.dataset)
        self._zarr_engine_used = self.dataset.zarr_engine_used
        if self.time_buffer < 0:
            raise ValueError(
                "time_buffer must be greater than or equal to 0. "
                f"Got {self.time_buffer}"
            )

    @property
    def zarr_engine_used(self) -> bool:
        """
        Whether any of the configured datasets are using the Zarr engine.
        """
        return self._zarr_engine_used
