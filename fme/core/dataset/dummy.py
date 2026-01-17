import datetime

import cftime
import torch
import xarray as xr

from fme.core.coordinates import HorizontalCoordinates, NullVerticalCoordinate
from fme.core.dataset.dataset import DatasetABC, DatasetItem
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.xarray import _get_timestep
from fme.core.mask_provider import MaskProvider


class DummyDataset(DatasetABC):
    def __init__(
        self,
        start_time: cftime.datetime,
        end_time: cftime.datetime,
        timestep: datetime.timedelta,
        n_timesteps: IntSchedule,
        horizontal_coordinates: HorizontalCoordinates,
        labels: set[str] | None = None,
    ):
        """
        Parameters:
            start_time: Initial time for the dummy dataset.
            end_time: End time for the dummy dataset.
            timestep: Timestep between each time in the dataset.
            n_timesteps: Number of contiguous timesteps to provide in each item.
            horizontal_coordinates: Horizontal coordinates for the dummy dataset.
            labels: Set of labels attached to samples in the dataset.
        """
        self.timestep = timestep
        self._n_timesteps_schedule = n_timesteps
        calendar = start_time.calendar
        self._all_times = xr.date_range(
            start=start_time,
            end=end_time,
            freq=self.timestep,
            calendar=calendar,
            use_cftime=True,
        )
        self._apply_sample_n_times(self._n_timesteps_schedule.get_value(0))
        self._timestep = _get_timestep(self._all_times)
        self._horizontal_coordinates = horizontal_coordinates
        self._horizontal_size = horizontal_coordinates.loaded_sizes
        shape = tuple(s.size for s in self._horizontal_size)
        full_shape = (self.sample_n_times,) + shape
        self._dummy_dict = {
            "__dummy__": torch.zeros(full_shape, device=torch.device("cpu"))
        }
        self._labels = labels
        self._epoch: int | None = None

    @property
    def all_times(self) -> xr.CFTimeIndex:
        """Time index of all available times in the data."""
        return self._all_times

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        """Return cftime index corresponding to start time of each sample."""
        return self._sample_start_times

    def _apply_sample_n_times(self, sample_n_times: int):
        total_timesteps = len(self._all_times)
        self._sample_n_times = sample_n_times
        self._n_initial_conditions = total_timesteps - self._sample_n_times + 1
        self._sample_start_times = self._all_times[: self._n_initial_conditions]

    @property
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        return self._sample_n_times

    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        raise NotImplementedError(
            "Dummy datasets do not support getting samples by time slice, "
            "is this a bug?."
        )

    @property
    def properties(self) -> DatasetProperties:
        return DatasetProperties(
            variable_metadata={},
            vertical_coordinate=NullVerticalCoordinate(),
            horizontal_coordinates=self._horizontal_coordinates,
            mask_provider=MaskProvider(),
            timestep=self.timestep,
            is_remote=False,
            all_labels=self._labels,
        )

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        pass

    def __getitem__(self, idx: int) -> DatasetItem:
        """Return a sample of data spanning the timesteps
        [idx, idx + self.sample_n_times).

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of a dummy data of zeros and its corresponding time coordinate.
        """
        time_slice = slice(idx, idx + self.sample_n_times)
        time = xr.DataArray(self.all_times[time_slice].values, dims=["time"])
        return (self._dummy_dict, time, self._labels, self._epoch)

    def set_epoch(self, epoch: int):
        self._apply_sample_n_times(self._n_timesteps_schedule.get_value(epoch))
        self._epoch = epoch
