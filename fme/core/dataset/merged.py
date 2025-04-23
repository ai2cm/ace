from typing import Sequence, Tuple, Union

import xarray as xr

from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.subset import XarraySubset
from fme.core.dataset.xarray import XarrayDataset
from fme.core.typing_ import TensorDict


class MergedXarrayDataset:
    def __init__(
        self, datasets: Sequence[Union[XarrayDataset, XarraySubset, XarrayConcat]]
    ):
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
        for dataset in self.datasets:
            if not dataset.sample_start_times.equals(
                self.datasets[0].sample_start_times
            ):
                raise ValueError(
                    "All datasets in a merged dataset must have the same sample "
                    "start times."
                )
            if not dataset.sample_n_times == self.datasets[0].sample_n_times:
                raise ValueError(
                    "All datasets in the merged datasets \
                         must have the same number of steps per sample item."
                )

    def __getitem__(self, idx: int) -> Tuple[TensorDict, xr.DataArray]:
        tensors = {}
        for dataset in self.datasets:
            ds_tensors, time = dataset[idx]
            tensors.update(ds_tensors)
        return tensors, time

    def __len__(self) -> int:
        return len(self.datasets[0])

    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> Tuple[TensorDict, xr.DataArray]:
        tensors: TensorDict = {}
        for dataset in self.datasets:
            ds_tensors, time = dataset.get_sample_by_time_slice(time_slice)
            tensors.update(ds_tensors)
        return tensors, time

    @property
    def all_times(self) -> xr.CFTimeIndex:
        return self.datasets[0].all_times

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

    @property
    def total_timesteps(self) -> int:
        return self.datasets[0].total_timesteps
