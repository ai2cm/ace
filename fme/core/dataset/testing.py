from typing import Self

import cftime
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates, NullVerticalCoordinate
from fme.core.dataset.dataset import DatasetABC, DatasetItem
from fme.core.dataset.properties import DatasetProperties
from fme.core.mask_provider import MaskProvider
from fme.core.typing_ import TensorMapping


class TestingDataset(DatasetABC):
    START_DATE = "2000-01-01"
    START_DATE_CF = cftime.DatetimeGregorian(2000, 1, 1)
    CALENDAR = "gregorian"

    @classmethod
    def time_to_int(cls, time: cftime.DatetimeGregorian) -> int:
        """Convert a cftime index to integer representation (days since START_DATE)."""
        ref = cls.START_DATE_CF
        delta = time - ref
        return delta.days

    def __init__(
        self,
        time: xr.CFTimeIndex,
        data: TensorMapping,
        sample_n_times: int,
        labels: set[str] | None = None,
        properties: DatasetProperties | None = None,
        initial_epoch: int | None = None,
    ):
        self.labels = labels
        self.time = time
        self.data = data
        self._sample_n_times = sample_n_times
        if properties is not None:
            self._properties = properties
        else:
            self._properties = DatasetProperties(
                variable_metadata={},
                vertical_coordinate=NullVerticalCoordinate(),
                horizontal_coordinates=LatLonCoordinates(
                    lon=torch.arange(0, 1), lat=torch.arange(0, 1)
                ),
                mask_provider=MaskProvider(),
                timestep=None,
                is_remote=False,
                all_labels=self.labels,
            )
        self.epoch = initial_epoch

    @classmethod
    def new(
        cls,
        n_times: int,
        varnames: list[str],
        sample_n_times: int,
        labels: set[str] | None = None,
        properties: DatasetProperties | None = None,
        initial_epoch: int | None = None,
    ) -> Self:
        time = xr.date_range(
            cls.START_DATE,
            periods=n_times,
            freq="D",
            calendar=cls.CALENDAR,
            use_cftime=True,
        )
        data = {varname: torch.arange(n_times) for varname in varnames}
        return cls(
            time=time,
            data=data,
            sample_n_times=sample_n_times,
            labels=labels,
            properties=properties,
            initial_epoch=initial_epoch,
        )

    def __getitem__(self, index) -> DatasetItem:
        time_slice = slice(index, index + self._sample_n_times)
        return self.get_sample_by_time_slice(time_slice)

    @property
    def all_times(self) -> xr.CFTimeIndex:
        return self.time

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        return self.time[: len(self.time) - self._sample_n_times + 1]

    @property
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        return self._sample_n_times

    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        data = {k: v[time_slice] for k, v in self.data.items()}
        time = xr.DataArray(self.time[time_slice], dims=["time"])
        return data, time, self.labels, self.epoch

    @property
    def properties(self) -> DatasetProperties:
        return self._properties

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
        if max_start_index + max_window_len > len(self):
            raise ValueError("Inference length exceeds dataset length.")

    def set_epoch(self, epoch: int):
        self.epoch = epoch
