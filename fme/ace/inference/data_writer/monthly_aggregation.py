"""Format-agnostic pieces of a monthly-mean output dataset.

This module owns the month indexing, the running-mean update, the valid-time
calendar arithmetic, and the names those quantities are stored under.
``monthly.py`` and ``monthly_zarr.py`` own the netCDF and zarr
representations built on them.
"""

import datetime
from collections.abc import Sequence

import cftime
import numpy as np
import numpy.typing as npt
import xarray as xr

LEAD_TIME_DIM = "time"
LEAD_TIME_UNITS = "months"
ENSEMBLE_DIM = "sample"
INIT_TIME = "init_time"
VALID_TIME = "valid_time"
TIME_UNITS = "days since 1970-01-01 00:00:00"
COUNTS = "counts"


class MonthIndexer:
    """
    Maps batch times to indices along a monthly lead-time axis.

    Month index 0 is the calendar month of each sample's first output time.
    Samples are indexed independently, so ensemble members with different
    initial times each start at their own month 0.
    """

    def __init__(self, first_output_times: Sequence[cftime.datetime]):
        """
        Args:
            first_output_times: The first time each sample will write, one per
                sample. The inference loop drops the initial condition step, so
                this is each sample's initial condition time plus one timestep.
        """
        self._init_years, self._init_months = _years_and_months(first_output_times)

    @property
    def init_years(self) -> np.ndarray:
        """Calendar year of month 0, per sample."""
        return self._init_years.copy()

    @property
    def init_months(self) -> np.ndarray:
        """Zero-indexed calendar month of month 0, per sample."""
        return self._init_months.copy()

    def month_indices(self, batch_time: xr.DataArray) -> np.ndarray:
        """
        Month index of each time in a batch.

        Args:
            batch_time: Time coordinate for each sample in the batch, of shape
                [sample, time].

        Returns:
            Month indices for the batch, of shape [sample, time]. Negative
            where a time precedes its sample's month 0.
        """
        years = batch_time.dt.year.values
        # datetime months are 1-indexed, we want 0-indexed
        months = batch_time.dt.month.values - 1
        return 12 * (years - self._init_years[:, None]) + (
            months - self._init_months[:, None]
        )

    def n_months_through(self, times: Sequence[cftime.datetime]) -> int:
        """
        Length of the lead-time axis needed to hold data up to ``times``.

        Args:
            times: One time per sample, the last time that sample will write.

        Returns:
            Number of months from month 0 through the latest of ``times``,
            inclusive, maximized over samples.
        """
        years, months = _years_and_months(times)
        elapsed = 12 * (years - self._init_years) + (months - self._init_months)
        return int(np.max(elapsed)) + 1


def _years_and_months(
    times: Sequence[cftime.datetime],
) -> tuple[np.ndarray, np.ndarray]:
    """Calendar years and zero-indexed calendar months of ``times``."""
    years = np.array([time.year for time in times], dtype=int)
    months = np.array([time.month - 1 for time in times], dtype=int)
    return years, months


def add_data(
    *,
    target: np.ndarray,
    target_start_counts: np.ndarray,
    source: np.ndarray,
    months_elapsed: np.ndarray,
):
    """
    Add source data to target monthly mean data, aggregating by month.

    All operations are performed independently on each batch member [b, ...].

    Args:
        target: Array of monthly mean data to add to, of shape
            [b, month].
        target_start_counts: Array of counts for each month, of shape
            [b, month]. This array does not get updated.
        source: Array of values to add into the monthly aggregates, of shape [b, time].
        months_elapsed: Elapsed months of source since the start of the data,
            of shape [b, time],
            corresponding to an index of the target array for each value in source.
            Assumed to be monotonically increasing.
    """
    for i_sample in range(source.shape[0]):
        i_time = 0
        while i_time < source.shape[1]:
            month_index = months_elapsed[i_sample, i_time]
            i_month_boundary = i_time + find_boundary(
                months_elapsed[i_sample, i_time:], month_index
            )
            # Calculate sum of new data for the current month
            new_data_sum = np.sum(source[i_sample, i_time:i_month_boundary], axis=0)
            new_samples_count = i_month_boundary - i_time

            # Update target mean for the month
            old_mean = target[i_sample, month_index]
            old_count = target_start_counts[i_sample, month_index]
            new_mean = (old_mean * old_count + new_data_sum) / (
                old_count + new_samples_count
            )

            target[i_sample, month_index] = new_mean

            i_time = i_month_boundary


def find_boundary(month_array, start_month) -> int:
    """
    Assuming month_array is an ordered array of months,
    find the index of the first month that is not start_month.
    """
    return np.searchsorted(month_array, start_month, side="right")


def get_valid_times(
    init_years: np.ndarray,
    init_months: np.ndarray,
    n_months: int,
    calendar: str,
    month_offset: int = 0,
) -> npt.NDArray[np.int64]:
    """
    Valid time of each month of the lead-time axis, in ``TIME_UNITS``.

    The 15th of each month is used, which is 14 days into the month.

    Args:
        init_years: Calendar year of month 0, per sample.
        init_months: Zero-indexed calendar month of month 0, per sample.
        n_months: Number of months to compute.
        calendar: Calendar to use.
        month_offset: Month index the returned range starts at (default 0).

    Returns:
        Days since the ``TIME_UNITS`` reference date, of shape
        [n_samples, n_months].
    """
    reference_date = cftime.datetime(1970, 1, 1, calendar=calendar)
    months_elapsed = np.arange(month_offset, month_offset + n_months)
    calendar_month = (init_months[:, None] + months_elapsed[None, :]) % 12
    calendar_year = (
        init_years[:, None] + (init_months[:, None] + months_elapsed[None, :]) // 12
    )
    days_since_reference = np.zeros_like(calendar_month, dtype=np.int64)
    for i in range(calendar_month.shape[0]):
        dates_sample = xr.date_range(
            cftime.datetime(
                calendar_year[i, 0], calendar_month[i, 0] + 1, 1, calendar=calendar
            ),
            cftime.datetime(
                calendar_year[i, -1], calendar_month[i, -1] + 1, 1, calendar=calendar
            ),
            freq="MS",
            calendar=calendar,
            use_cftime=True,
        )
        days_since_reference[i, :] = (
            dates_sample.values - reference_date
        ) // datetime.timedelta(days=1)
    return days_since_reference + 14
