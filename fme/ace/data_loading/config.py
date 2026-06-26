import dataclasses
from collections.abc import Sequence

from fme.ace.data_loading.augmentation import AugmentationConfig
from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.dataset import DatasetABC
from fme.core.dataset.merged import MergeDatasetConfig
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule, WeightMilestone, WeightSchedule
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.distributed import Distributed


@dataclasses.dataclass
class GroupWeightsConfig:
    """
    Configuration for scheduled group-weighted sampling of a concat dataset.

    Partitions the concat members into consecutive groups and assigns each group
    a sampling weight, optionally varying the weights across training epochs.

    Parameters:
        groups: Number of consecutive concat members in each group. All entries
            must be positive and their sum must equal the number of concat
            members.
        start_value: Initial per-group sampling weight, one per group. Weights
            need not sum to 1; they are normalized at use.
        milestones: Epoch-keyed overrides of the per-group weights.
    """

    groups: list[int]
    start_value: list[float]
    milestones: list[WeightMilestone] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if len(self.groups) != len(self.start_value):
            raise ValueError(
                "groups and start_value must have the same length, got "
                f"{len(self.groups)} and {len(self.start_value)}"
            )
        if any(g <= 0 for g in self.groups):
            raise ValueError(f"All group sizes must be positive, got {self.groups}")
        self.schedule = WeightSchedule(
            start_value=self.start_value, milestones=self.milestones
        )


@dataclasses.dataclass
class DataLoaderConfig:
    r"""
    Configuration for a data loader for training/validation.

    Parameters:
        dataset: Could be a single dataset configuration,
            or a sequence of datasets to be concatenated using the keyword `concat`,
            or datasets from different sources to be merged using the keyword `merge`.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        prefetch_factor: how many batches a single data worker will attempt to
            hold in host memory at a given time.
        augmentation: Configuration for data augmentation.
        sample_with_replacement: If provided, the dataset will be
            sampled randomly with replacement to the given size each period,
            instead of retrieving each sample once (either shuffled or not).
        group_weights: If provided, partition the concat dataset members into
            groups and sample them with scheduled per-group weights. Only
            supported for a ``ConcatDatasetConfig`` on a training loader, and
            mutually exclusive with ``sample_with_replacement``.
        time_buffer: How many more continuous timesteps to load in memory than the
            required number of timesteps for a single batch. Setting this to greater
            than 0 should improve data loading performance, however, it also decreases
            the independence of subsequent batches if shuffled batches are desired.
        time_buffer_pool_size: Number of pre-loaded windows to hold in memory
            simultaneously when ``time_buffer > 0``. With ``time_buffer`` alone,
            consecutive output batches are correlated because they are drawn from
            the same window. Increasing ``time_buffer_pool_size`` reduces this
            correlation by interleaving output batches across multiple windows.
            Each pool slot holds one window of
            ``batch_size * (time_buffer + n_timesteps)`` tensors in memory.
            Requires ``time_buffer > 0``.

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

    dataset: ConcatDatasetConfig | MergeDatasetConfig | XarrayDataConfig
    batch_size: int
    num_data_workers: int = 0
    prefetch_factor: int | None = None
    augmentation: AugmentationConfig = dataclasses.field(
        default_factory=lambda: AugmentationConfig()
    )
    sample_with_replacement: int | None = None
    time_buffer: int = 0
    time_buffer_pool_size: int = 1
    group_weights: GroupWeightsConfig | None = None

    @property
    def using_labels(self) -> bool:
        return self.available_labels is not None

    def get_dataset(
        self,
        names: Sequence[str],
        n_timesteps: IntSchedule,
        allow_missing_variables: bool = False,
    ) -> tuple[DatasetABC, DatasetProperties]:
        return self.dataset.build(
            names, n_timesteps, allow_missing_variables=allow_missing_variables
        )

    @property
    def available_labels(self) -> set[str] | None:
        """
        Return the labels that are available in the dataset.
        """
        return self.dataset.available_labels

    def __post_init__(self):
        dist = Distributed.get_instance()
        if self.batch_size % dist.total_data_parallel_ranks != 0:
            raise ValueError(
                "batch_size must be divisible by the number of data-parallel "
                f"workers, got {self.batch_size} and "
                f"{dist.total_data_parallel_ranks}"
            )
        self._zarr_engine_used = self.dataset.zarr_engine_used
        if self.time_buffer < 0:
            raise ValueError(
                "time_buffer must be greater than or equal to 0. "
                f"Got {self.time_buffer}"
            )
        if self.time_buffer_pool_size < 1:
            raise ValueError(
                "time_buffer_pool_size must be greater than or equal to 1. "
                f"Got {self.time_buffer_pool_size}"
            )
        if self.time_buffer == 0 and self.time_buffer_pool_size > 1:
            raise ValueError(
                "time_buffer_pool_size > 1 requires time_buffer > 0. "
                f"Got time_buffer={self.time_buffer}, "
                f"time_buffer_pool_size={self.time_buffer_pool_size}"
            )
        if self.group_weights is not None:
            if self.sample_with_replacement is not None:
                raise ValueError(
                    "group_weights cannot be combined with "
                    "sample_with_replacement; both are with-replacement "
                    "sampler choices."
                )
            if not isinstance(self.dataset, ConcatDatasetConfig):
                raise ValueError(
                    "group_weights requires the dataset to be a "
                    "ConcatDatasetConfig (use the `concat` keyword)."
                )
            n_members = len(self.dataset.concat)
            if sum(self.group_weights.groups) != n_members:
                raise ValueError(
                    "sum(group_weights.groups) must equal the number of concat "
                    f"members ({n_members}), got "
                    f"{sum(self.group_weights.groups)}"
                )

    @property
    def zarr_engine_used(self) -> bool:
        """
        Whether any of the configured datasets are using the Zarr engine.
        """
        return self._zarr_engine_used
