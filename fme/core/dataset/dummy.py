import datetime

import cftime
import torch
import xarray as xr

from fme.core.coordinates import HorizontalCoordinates
from fme.core.dataset.xarray import _get_timestep
from fme.core.typing_ import TensorDict


class DummyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        start_time: cftime.datetime,
        end_time: cftime.datetime,
        timestep: datetime.timedelta,
        n_timesteps: int,
        horizontal_coordinates: HorizontalCoordinates,
    ):
        """
        Parameters:
            start_time: Initial time for the dummy dataset.
            end_time: End time for the dummy dataset.
            timestep: Timestep between each time in the dataset.
            n_timesteps: Number of contiguous timesteps to provide in each item.
            horizontal_coordinates: Horizontal coordinates for the dummy dataset.
        """
        self.timestep = timestep
        self.sample_n_times = n_timesteps
        calendar = start_time.calendar
        self._all_times = xr.date_range(
            start=start_time,
            end=end_time,
            freq=self.timestep,
            calendar=calendar,
            use_cftime=True,
        )
        total_timesteps = len(self._all_times)
        self._n_initial_conditions = total_timesteps - self.sample_n_times + 1
        self._sample_start_times = self._all_times[: self._n_initial_conditions]
        self._timestep = _get_timestep(self._all_times)
        self._horizontal_size = horizontal_coordinates.loaded_sizes
        self._labels: set[str] = set()
        shape = tuple(s.size for s in self._horizontal_size)
        full_shape = (self.sample_n_times,) + shape
        self._dummy_dict = {
            "__dummy__": torch.zeros(full_shape, device=torch.device("cpu"))
        }

    @property
    def all_times(self) -> xr.CFTimeIndex:
        """Time index of all available times in the data."""
        return self._all_times

    def __len__(self):
        return self._n_initial_conditions

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        """Return cftime index corresponding to start time of each sample."""
        return self._sample_start_times

    def __getitem__(self, idx: int) -> tuple[TensorDict, xr.DataArray, set[str]]:
        """Return a sample of data spanning the timesteps
        [idx, idx + self.sample_n_times).

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of a dummy data of zeros and its corresponding time coordinate.
        """
        time_slice = slice(idx, idx + self.sample_n_times)
        time = xr.DataArray(self.all_times[time_slice].values, dims=["time"])
        return (self._dummy_dict, time, self._labels)
