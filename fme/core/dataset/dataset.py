import abc
from typing import Any, final

import xarray as xr

from fme.core.dataset.properties import DatasetProperties
from fme.core.generics.dataset import GenericDataset
from fme.core.typing_ import TensorDict

DatasetItem = tuple[TensorDict, xr.DataArray, set[str] | None, int | None]


class DatasetABC(GenericDataset[DatasetItem], abc.ABC):
    @property
    @abc.abstractmethod
    def sample_start_times(self) -> xr.CFTimeIndex:
        pass

    @property
    @abc.abstractmethod
    def all_times(self) -> xr.CFTimeIndex:
        """
        Like sample_start_times, but includes all times in the dataset, including
        final times which are not valid as a start index.

        This is relevant for inference, where we may use get_sample_by_time_slice to
        retrieve time windows directly.

        If this dataset does not support inference,
        this will raise a NotImplementedError.
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, index) -> DatasetItem:
        pass

    @final
    def __len__(self) -> int:
        return len(self.sample_start_times)

    @property
    @final
    def first_time(self) -> Any:
        return self.sample_start_times[0]

    @property
    @final
    def last_time(self) -> Any:
        return self.sample_start_times[-1]

    @property
    @abc.abstractmethod
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        pass

    @abc.abstractmethod
    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        pass

    @property
    @abc.abstractmethod
    def properties(self) -> DatasetProperties:
        pass

    @abc.abstractmethod
    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        """
        Validate that this dataset supports inference, and that the
        number of forward steps is valid for the dataset.
        Raises a ValueError if the number of forward steps is not valid or
        if the dataset does not support inference.

        Parameters:
            max_start_index: The maximum valid start index for inference.
            max_window_len: The maximum window length including the
                start index for inference.
        """
        pass
