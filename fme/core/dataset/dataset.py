import abc

import torch
import xarray as xr

from fme.core.dataset.properties import DatasetProperties
from fme.core.typing_ import TensorDict


class DatasetABC(abc.ABC, torch.utils.data.Dataset):
    @property
    @abc.abstractmethod
    def sample_start_times(self) -> xr.CFTimeIndex:
        pass

    @property
    @abc.abstractmethod
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        pass

    @abc.abstractmethod
    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> tuple[TensorDict, xr.DataArray, set[str]]:
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
