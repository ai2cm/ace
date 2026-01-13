import dataclasses
import warnings
from collections.abc import Sequence

import torch
import xarray as xr

from fme.core.dataset.config import DatasetConfigABC
from fme.core.dataset.dataset import DatasetABC, DatasetItem
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.utils import accumulate_labels
from fme.core.dataset.xarray import XarrayDataConfig, get_xarray_datasets


class XarrayConcat(DatasetABC):
    def __init__(self, datasets: Sequence[DatasetABC], strict: bool = True):
        self._dataset = torch.utils.data.ConcatDataset(datasets)
        self._wrapped_datasets = datasets
        sample_start_times = datasets[0].sample_start_times
        for dataset in datasets[1:]:
            sample_start_times = sample_start_times.append(dataset.sample_start_times)
            assert dataset.sample_n_times == datasets[0].sample_n_times
            if not dataset.sample_n_times == datasets[0].sample_n_times:
                raise ValueError(
                    "All concatenated datasets \
                         must have the same number of steps per sample item."
                )
        self._sample_start_times = sample_start_times
        assert len(self._dataset) == len(sample_start_times)
        self._sample_n_times = datasets[0].sample_n_times
        self._properties = datasets[0].properties.copy()
        for dataset in datasets[1:]:
            if strict:
                self._properties.update(dataset.properties)
            else:
                try:
                    self._properties.update(dataset.properties)
                except ValueError as e:
                    warnings.warn(
                        f"Metadata for each ensemble member are not the same: {e}"
                    )

    def __getitem__(self, idx: int) -> DatasetItem:
        return self._dataset[idx]

    @property
    def sample_start_times(self):
        return self._sample_start_times

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
        raise NotImplementedError("Concat datasets do not support inference.")

    @property
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        return self._sample_n_times

    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        raise NotImplementedError(
            "Concat datasets do not support getting samples by time slice, "
            "and should not be configurable for inference. Is there a bug?."
        )

    @property
    def properties(self) -> DatasetProperties:
        return self._properties

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        raise ValueError("Concat datasets do not support inference.")

    def set_epoch(self, epoch: int):
        for dataset in self._wrapped_datasets:
            dataset.set_epoch(epoch)


def get_dataset(
    dataset_configs: Sequence[XarrayDataConfig],
    names: Sequence[str],
    n_timesteps: IntSchedule,
    strict: bool = True,
) -> tuple[XarrayConcat, DatasetProperties]:
    datasets, properties = get_xarray_datasets(
        dataset_configs, names, n_timesteps, strict=strict
    )
    ensemble = XarrayConcat(datasets, strict=strict)
    return ensemble, properties


@dataclasses.dataclass
class ConcatDatasetConfig(DatasetConfigABC):
    """
    Configuration for concatenating multiple datasets across time.

    Parameters:
        concat: List of XarrayDataConfig objects to concatenate.
        strict: Whether to enforce that the datasets to be concatenated
            have the same dimensions and spatial coordinates.
    """

    concat: Sequence[XarrayDataConfig]
    strict: bool = True

    def __post_init__(self):
        self.zarr_engine_used = any(ds.engine == "zarr" for ds in self.concat)

    def build(
        self,
        names: Sequence[str],
        n_timesteps: IntSchedule,
    ) -> tuple[DatasetABC, DatasetProperties]:
        return get_dataset(
            self.concat,
            names,
            n_timesteps,
            strict=self.strict,
        )

    @property
    def available_labels(self) -> set[str] | None:
        """
        Return the labels that are available in the dataset.
        """
        return accumulate_labels([ds.available_labels for ds in self.concat])
