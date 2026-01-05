import dataclasses
from collections.abc import Sequence

import xarray as xr

from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.config import DatasetConfigABC
from fme.core.dataset.dataset import DatasetABC, DatasetItem
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.dataset.utils import accumulate_labels
from fme.core.dataset.xarray import XarrayDataConfig, get_raw_paths
from fme.core.typing_ import Slice, TensorDict


class MergedXarrayDataset(DatasetABC):
    def __init__(self, datasets: Sequence[DatasetABC]):
        self.datasets = datasets

        combined_names = [
            item for dataset in self.datasets for item in dataset[0][0].keys()
        ]
        if len(combined_names) != len(set(combined_names)):
            duplicates = list(
                {item for item in combined_names if combined_names.count(item) > 1}
            )
            raise ValueError(
                f"Variable names must be unique across merged datasets. \
                    \nDuplicates found: {duplicates}"
            )
        self._sample_start_times = self.datasets[0].sample_start_times
        self._sample_n_times = self.datasets[0].sample_n_times
        for dataset in self.datasets:
            if not dataset.sample_start_times.equals(self._sample_start_times):
                raise ValueError(
                    "All datasets in a merged dataset must have the same sample "
                    "start times."
                )
            if not dataset.sample_n_times == self._sample_n_times:
                raise ValueError(
                    "All datasets in the merged datasets \
                         must have the same number of steps per sample item."
                )

    def __getitem__(self, idx: int) -> DatasetItem:
        tensors: TensorDict = {}
        labels = None
        epochs = []
        for dataset in self.datasets:
            ds_tensors, time, ds_labels, ds_epoch = dataset[idx]
            if labels is None:
                labels = ds_labels
            else:
                if ds_labels is not None:
                    labels = labels.union(ds_labels)
            tensors.update(ds_tensors)
            epochs.append(ds_epoch)
        if not all(epoch == epochs[0] for epoch in epochs):
            raise ValueError(
                "All datasets in a merged dataset must have the same epoch."
            )
        return tensors, time, labels, epochs[0]

    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        tensors: TensorDict = {}
        for dataset in self.datasets:
            ds_tensors, time, labels, epoch = dataset.get_sample_by_time_slice(
                time_slice
            )
            tensors.update(ds_tensors)
        return tensors, time, labels, epoch

    @property
    def all_times(self) -> xr.CFTimeIndex:
        """
        Like sample_start_times, but includes all times in the dataset, including
        final times which are not valid as a start index.

        This is relevant for inference, where we may use get_sample_by_time_slice to
        retrieve time windows directly.

        If this dataset does not support inference,
        this will raise a NotImplementedError.
        """
        return self.datasets[0].all_times

    @property
    def sample_start_times(self):
        return self._sample_start_times

    @property
    def sample_n_times(self) -> int:
        return self._sample_n_times

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        for dataset in self.datasets:
            dataset.validate_inference_length(max_start_index, max_window_len)

    @property
    def properties(self) -> DatasetProperties:
        data_properties = None
        for dataset in self.datasets:
            if data_properties is None:
                data_properties = dataset.properties
            else:
                data_properties.update_merged_dataset(dataset.properties)
        if data_properties is None:
            raise ValueError("No dataset available to determine properties")
        return data_properties

    def set_epoch(self, epoch: int):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)


@dataclasses.dataclass
class MergeDatasetConfig(DatasetConfigABC):
    """
    Configuration for merging multiple datasets. Merging means combining
    variables from multiple datasets, each of which must have the same
    time coordinate. If multiple datasets contain the same data variable, the version
    from the first source is loaded and other sources are ignored.

    Parameters:
        merge: List of dataset configurations to merge.
    """

    merge: Sequence[ConcatDatasetConfig | XarrayDataConfig]

    def __post_init__(self):
        self.zarr_engine_used = False
        for ds in self.merge:
            if isinstance(ds, ConcatDatasetConfig):
                if ds.zarr_engine_used:
                    self.zarr_engine_used = ds.zarr_engine_used
                    break
            elif isinstance(ds, XarrayDataConfig):
                if ds.engine == "zarr":
                    self.zarr_engine_used = True
                    break

    def build(
        self,
        names: Sequence[str],
        n_timesteps: IntSchedule,
    ):
        return get_merged_datasets(
            self,
            names,
            n_timesteps,
        )

    @property
    def available_labels(self) -> set[str] | None:
        """
        Return the labels that are available in the dataset.
        """
        return accumulate_labels([ds.available_labels for ds in self.merge])


@dataclasses.dataclass
class MergeNoConcatDatasetConfig(DatasetConfigABC):
    """
    Configuration for merging multiple datasets. Merging means combining
    variables from multiple datasets, each of which must have the same
    time coordinate.  If multiple datasets contain the same data variable, the version
    from the first source is loaded and other sources are ignored. For
    `MergeNoConcatDatasetConfig`, the datasets being merged may not be concatenated
    datasets.

    Parameters:
        merge: List of dataset configurations to merge.
    """

    merge: Sequence[XarrayDataConfig]

    def __post_init__(self):
        self.zarr_engine_used = False
        for ds in self.merge:
            if ds.engine == "zarr":
                self.zarr_engine_used = True
                break

    def update_subset(self, subset: Slice | TimeSlice | RepeatedInterval):
        for ds in self.merge:
            ds.update_subset(subset)

    @property
    def subset(self) -> Slice | TimeSlice | RepeatedInterval:
        return self.merge[0].subset

    def build(
        self,
        names: Sequence[str],
        n_timesteps: IntSchedule,
    ) -> tuple[MergedXarrayDataset, DatasetProperties]:
        return get_merged_datasets(
            MergeDatasetConfig(merge=self.merge),
            names,
            n_timesteps,
        )

    @property
    def available_labels(self) -> set[str] | None:
        """
        Return the labels that are available in the dataset.
        """
        return accumulate_labels([ds.available_labels for ds in self.merge])


def get_merged_datasets(
    merged_config: MergeDatasetConfig | MergeNoConcatDatasetConfig,
    names: Sequence[str],
    n_timesteps: IntSchedule,
) -> tuple[MergedXarrayDataset, DatasetProperties]:
    merged_xarray_datasets = []
    merged_properties: DatasetProperties | None = None
    per_dataset_names = get_per_dataset_names(merged_config, names)
    config_counter = 0
    for config in merged_config.merge:
        (
            current_source_xarray_dataset,
            current_source_properties,
        ) = config.build(
            per_dataset_names[config_counter],
            n_timesteps,
        )
        merged_xarray_datasets.append(current_source_xarray_dataset)

        if merged_properties is None:
            merged_properties = current_source_properties
        else:
            merged_properties.update_merged_dataset(current_source_properties)
        config_counter += 1

    if merged_properties is None:
        raise ValueError("At least one dataset must be provided.")
    merged_datasets = MergedXarrayDataset(datasets=merged_xarray_datasets)
    return merged_datasets, merged_properties


def _infer_available_variables(config: XarrayDataConfig):
    """
    Infer the available variables from a XarrayDataset.
    """
    raw_paths = get_raw_paths(config.data_path, config.file_pattern)
    if len(raw_paths) == 0:
        raise ValueError(
            f"No files found matching '{config.data_path}/{config.file_pattern}'."
        )
    dataset = xr.open_dataset(
        raw_paths[0],
        decode_times=False,
        decode_timedelta=False,
        engine=config.engine,
        chunks=None,
    )
    return dataset.data_vars


def get_per_dataset_names(
    merged_config: MergeDatasetConfig | MergeNoConcatDatasetConfig,
    names: Sequence[str],
) -> list[list[str]]:
    merged_required_names = list(names)
    per_dataset_names = []
    for config in merged_config.merge:
        if isinstance(config, XarrayDataConfig):
            current_source_variables = _infer_available_variables(config)
        elif isinstance(config, ConcatDatasetConfig):
            current_source_variables = _infer_available_variables(config.concat[0])
        current_source_names = [
            name for name in merged_required_names if name in current_source_variables
        ]
        per_dataset_names.append(current_source_names)
        for name in current_source_names:
            merged_required_names.remove(name)
    return per_dataset_names
