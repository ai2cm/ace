import xarray as xr

from fme.core.dataset.dataset import DatasetABC
from fme.core.dataset.properties import DatasetProperties


class SubsetDataset(DatasetABC):
    def __init__(self, dataset: DatasetABC, indices: list[int]):
        """
        A dataset that represents a subset of another dataset.

        Parameters:
            dataset: The original dataset to subset.
            indices: List of indices to include in the subset.
        """
        self._dataset = dataset
        self._indices = indices

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        """Return cftime index corresponding to start time of each sample."""
        return self._dataset.sample_start_times[self._indices]

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
        raise NotImplementedError("SubsetDataset does not support inference.")

    @property
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        return self._dataset.sample_n_times

    def get_sample_by_time_slice(self, time_slice: slice):
        raise NotImplementedError(
            "SubsetDataset does not support getting samples by time slice."
        )

    @property
    def properties(self) -> DatasetProperties:
        return self._dataset.properties

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        raise ValueError("SubsetDataset does not support inference.")

    def set_epoch(self, epoch):
        self._dataset.set_epoch(epoch)

    def __getitem__(self, idx: int):
        actual_idx = self._indices[idx]
        return self._dataset[actual_idx]
