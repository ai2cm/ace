import dataclasses
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr


@dataclasses.dataclass
class TimeSlice:
    """
    Configuration of a slice of times. Step is an integer-valued index step.

    Note: start_time and stop_time may be provided as partial time strings and the
        stop_time will be included in the slice. See more details in `Xarray docs`_.

    Parameters:
        start_time: Start time of the slice.
        stop_time: Stop time of the slice.
        step: Step of the slice.

    .. _Xarray docs:
       https://docs.xarray.dev/en/latest/user-guide/weather-climate.html#non-standard-calendars-and-dates-outside-the-nanosecond-precision-range
    """  # noqa: E501

    start_time: str | None = None
    stop_time: str | None = None
    step: int | None = None

    def as_raw_slice(self) -> slice:
        """
        Return the raw slice object without applying it to a time index.
        E.g., directly as a selection method for an xarray object.
        """
        return slice(self.start_time, self.stop_time, self.step)

    def slice(self, time: xr.CFTimeIndex) -> slice:
        """
        Return a slice object with indexing based on the provided time index.
        """
        return time.slice_indexer(self.start_time, self.stop_time, self.step)


def _convert_interval_to_int(
    interval: pd.Timedelta,
    timestep: timedelta,
):
    """Convert interval to integer number of timesteps."""
    if interval % timestep != timedelta(0):
        raise ValueError(
            f"Requested interval length {interval} is not a "
            f"multiple of the timestep {timestep}."
        )

    return interval // timestep


@dataclasses.dataclass
class RepeatedInterval:
    """
    Configuration for a repeated interval within a block.  This configuration
    is used to generate a boolean mask for a dataset that will return values
    within the interval and repeat that throughout the dataset.

    Parameters:
        interval_length: Length of the interval to return values from
        start: Start position of the interval within the repeat block.
        block_length: Total length of the block to be repeated over the length of
            the dataset, including the interval length.

    Note:
        The interval_length, start, and block_length can be provided as either
        all integers or all strings representing timedeltas of the block.
        If provided as strings, the timestep must be provided when calling
        `get_boolean_mask`.

    Examples:
        To return values from the first 3 items of every 6 items, use:

        >>> fme.ace.RepeatedInterval(interval_length=3, block_length=6, start=0)  # doctest: +IGNORE_OUTPUT

        To return a days worth of values starting after 2 days from every 7-day
        block, use:

        >>> fme.ace.RepeatedInterval(interval_length="1d", block_length="7d", start="2d")  # doctest: +IGNORE_OUTPUT
    """  # noqa: E501

    interval_length: int | str
    start: int | str
    block_length: int | str

    def __post_init__(self):
        types = {type(self.interval_length), type(self.block_length), type(self.start)}
        if len(types) > 1:
            raise ValueError(
                "All attributes of RepeatedInterval must be of the "
                "same type (either all int or all str)."
            )

        self._is_time_delta_str = isinstance(self.interval_length, str)

        if self._is_time_delta_str:
            self.interval_length = pd.Timedelta(self.interval_length)
            self.block_length = pd.Timedelta(self.block_length)
            self.start = pd.Timedelta(self.start)

    def get_boolean_mask(
        self, length: int, timestep: timedelta | None = None
    ) -> np.ndarray:
        """
        Return a boolean mask for the repeated interval.

        Args:
            length: Length of the dataset.
            timestep: Timestep of the dataset.
        """
        if self._is_time_delta_str:
            if timestep is None:
                raise ValueError(
                    "Timestep must be provided when using time deltas "
                    "for RepeatedInterval."
                )

            interval_length = _convert_interval_to_int(self.interval_length, timestep)
            block_length = _convert_interval_to_int(self.block_length, timestep)
            start = _convert_interval_to_int(self.start, timestep)
        else:
            interval_length = self.interval_length
            block_length = self.block_length
            start = self.start

        if start + interval_length > block_length:
            raise ValueError(
                "The interval (with start point) must fit within the repeat block."
            )

        block = np.zeros(block_length, dtype=bool)
        block[start : start + interval_length] = True
        num_blocks = length // block_length + 1
        mask = np.tile(block, num_blocks)[:length]
        return mask
