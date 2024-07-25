import datetime
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cftime
import numpy as np
import torch
import xarray as xr
from netCDF4 import Dataset

from fme.ace.inference.data_writer.utils import get_all_names
from fme.core.data_loading.data_typing import VariableMetadata

LEAD_TIME_DIM = "time"
LEAD_TIME_UNITS = "months"
ENSEMBLE_DIM = "sample"
INIT_TIME = "init_time"
VALID_TIME = "valid_time"
TIME_UNITS = "days since 1970-01-01 00:00:00"
COUNTS = "counts"


def months_for_timesteps(n_timesteps: int, timestep: datetime.timedelta) -> int:
    steps_per_day = datetime.timedelta(days=1) / timestep
    return ceil(n_timesteps * (12.0 / (365.24 * steps_per_day))) + 2


class PairedMonthlyDataWriter:
    """
    Wrapper over MonthlyDataWriter to write both target and prediction data
    to the same file.

    Gives the same interface as for our other writers.
    """

    def __init__(
        self,
        path: str,
        n_samples: int,
        n_timesteps: int,
        timestep: datetime.timedelta,
        save_names: Optional[Sequence[str]],
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
    ):
        n_months = months_for_timesteps(n_timesteps, timestep)
        self._target_writer = MonthlyDataWriter(
            path=path,
            label="target",
            n_samples=n_samples,
            n_months=n_months,
            save_names=save_names,
            metadata=metadata,
            coords=coords,
        )
        self._prediction_writer = MonthlyDataWriter(
            path=path,
            label="predictions",
            n_samples=n_samples,
            n_months=n_months,
            save_names=save_names,
            metadata=metadata,
            coords=coords,
        )

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        batch_times: xr.DataArray,
    ):
        self._target_writer.append_batch(
            data=target, start_timestep=start_timestep, batch_times=batch_times
        )
        self._prediction_writer.append_batch(
            data=prediction, start_timestep=start_timestep, batch_times=batch_times
        )

    def flush(self):
        self._target_writer.flush()
        self._prediction_writer.flush()


class MonthlyDataWriter:
    """
    Write monthly total data and sample counts to a netCDF file.

    Allows computing the mean afterwards by dividing the total by the counts.
    """

    def __init__(
        self,
        path: str,
        label: str,
        n_samples: int,
        n_months: int,
        save_names: Optional[Sequence[str]],
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
    ):
        """
        Args:
            path: Directory to write netCDF file(s).
            label: Label to append to the filename.
            n_samples: Number of samples to write to the file, each sample being
                an ensemble member.
            n_months: Number of months to write to the file.
            save_names: Names of variables to save in the predictions netcdf file.
                If None, all predicted variables will be saved.
            metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
        """
        if label != "":
            label = "_" + label
        filename = str(Path(path) / f"monthly_mean{label}.nc")
        self._save_names = save_names
        self.metadata = metadata
        self.coords = coords
        self.dataset = Dataset(filename, "w", format="NETCDF4")
        self.dataset.createDimension(LEAD_TIME_DIM, n_months)
        self.dataset.createVariable(LEAD_TIME_DIM, "i8", (LEAD_TIME_DIM,))
        self.dataset.variables[LEAD_TIME_DIM].units = LEAD_TIME_UNITS
        self.dataset.variables[LEAD_TIME_DIM][:] = np.arange(n_months)
        self.dataset.createDimension(ENSEMBLE_DIM, n_samples)
        self.dataset.createVariable(INIT_TIME, "i8", (ENSEMBLE_DIM,))
        self.dataset.variables[INIT_TIME].units = TIME_UNITS
        self.dataset.createVariable(COUNTS, "i8", (ENSEMBLE_DIM, LEAD_TIME_DIM))
        self.dataset.createVariable(
            VALID_TIME,
            "i8",
            (
                ENSEMBLE_DIM,
                LEAD_TIME_DIM,
            ),
        )
        self.dataset.variables[VALID_TIME].units = TIME_UNITS
        self.dataset.variables[COUNTS][:] = 0
        self._n_lat: Optional[int] = None
        self._n_lon: Optional[int] = None
        self._init_years = np.full([n_samples], -1, dtype=int)
        self._init_months = np.full([n_samples], -1, dtype=int)

    def _get_initial_year_and_month(
        self,
        years: np.ndarray,
        months: np.ndarray,
        calendar: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        reference_date = cftime.datetime(1970, 1, 1, calendar=calendar)
        if self._init_years[0] == -1:
            self._init_years[:] = years
            self._init_months[:] = months
            n_months = self.dataset.variables[LEAD_TIME_DIM].shape[0]
            days_since_reference = get_days_since_reference(
                years=years,
                months=months,
                n_months=n_months,
                reference_date=reference_date,
                calendar=calendar,
            )
            self.dataset.variables[INIT_TIME][:] = days_since_reference[:, 0]
            # use the 15th of each month, which is 14 days into the month
            self.dataset.variables[VALID_TIME][:, :] = days_since_reference + 14
        return (self._init_years, self._init_months)

    def _get_month_indices(self, batch_times: xr.DataArray) -> np.ndarray:
        """
        Get the month indices for the batch of data.

        If this is the first time called, the times are assumed to be the
        initial times for the data, and are stored for determining the month
        indices in this and future calls.

        Args:
            batch_times: Time coordinates for each sample in the batch, of shape
                [ensemble_member, lead_time].

        Returns:
            Month indices for the batch of data.
        """
        calendar = batch_times.dt.calendar
        years = batch_times.dt.year.values
        # datetime months are 1-indexed, we want 0-indexed
        months = batch_times.dt.month.values - 1
        init_years, init_months = self._get_initial_year_and_month(
            years=years[:, 0], months=months[:, 0], calendar=calendar
        )
        return 12 * (years - init_years[:, None]) + (months - init_months[:, None])

    def _get_variable_names_to_save(
        self, *data_varnames: Iterable[str]
    ) -> Iterable[str]:
        return get_all_names(*data_varnames, allowlist=self._save_names)

    def append_batch(
        self,
        data: Dict[str, torch.Tensor],
        start_timestep: int,
        batch_times: xr.DataArray,
    ):
        """
        Append a batch of data to the file.

        Args:
            data: Values to store.
            start_timestep: Timestep index for the start of the batch, unused.
            batch_times: Time coordinates for each sample in the batch.
        """
        del start_timestep  # unused
        n_samples_data = list(data.values())[0].shape[0]
        n_samples_time = batch_times.sizes["sample"]
        if n_samples_data != n_samples_time:
            raise ValueError(
                f"Batch size mismatch, data has {n_samples_data} samples "
                f"and times has {n_samples_time} samples."
            )
        n_times_data = list(data.values())[0].shape[1]
        n_times_time = batch_times.sizes["time"]
        if n_times_data != n_times_time:
            raise ValueError(
                f"Batch time dimension mismatch, data has {n_times_data} times "
                f"and times has {n_times_time} times."
            )

        if self._n_lat is None:
            self._n_lat = data[next(iter(data.keys()))].shape[-2]
            self.dataset.createDimension("lat", self._n_lat)
            if "lat" in self.coords:
                self.dataset.createVariable("lat", "f4", ("lat",))
                self.dataset.variables["lat"][:] = self.coords["lat"]
        if self._n_lon is None:
            self._n_lon = data[next(iter(data.keys()))].shape[-1]
            self.dataset.createDimension("lon", self._n_lon)
            if "lon" in self.coords:
                self.dataset.createVariable("lon", "f4", ("lon",))
                self.dataset.variables["lon"][:] = self.coords["lon"]

        dims = (ENSEMBLE_DIM, LEAD_TIME_DIM, "lat", "lon")
        save_names = self._get_variable_names_to_save(data.keys())
        months = self._get_month_indices(batch_times)
        month_min = np.min(months)
        month_range = np.max(months) - month_min + 1
        count_data = self.dataset.variables[COUNTS][
            :, month_min : month_min + month_range
        ]
        for variable_name in save_names:
            # define the variable if it doesn't exist
            if variable_name not in self.dataset.variables:
                self.dataset.createVariable(
                    variable_name,
                    "f4",
                    dims,
                    fill_value=np.nan,
                )
                self.dataset.variables[variable_name][:] = 0.0
                if variable_name in self.metadata:
                    self.dataset.variables[variable_name].units = self.metadata[
                        variable_name
                    ].units
                    self.dataset.variables[variable_name].long_name = self.metadata[
                        variable_name
                    ].long_name
                self.dataset.variables[variable_name].coordinates = " ".join(
                    [INIT_TIME, VALID_TIME]
                )

            array = data[variable_name].cpu().numpy()

            # Add the data to the variable totals
            # Have to extract the data and write it back as `.at` does not play nicely
            # with netCDF4
            # We pull just the month subset we need for speed reasons
            month_data = self.dataset.variables[variable_name][
                :, month_min : month_min + month_range
            ]
            add_data(
                target=month_data,
                target_start_counts=count_data,
                source=array,
                months_elapsed=months - month_min,
            )
            self.dataset.variables[variable_name][
                :, month_min : month_min + month_range
            ] = month_data
        # counts must be added after data, as we use the base counts when updating means
        for i_sample in range(n_samples_data):
            self.dataset.variables[COUNTS][i_sample] += np.bincount(
                months[i_sample], minlength=self.dataset.variables[COUNTS].shape[1]
            )

        self.dataset.sync()

    def flush(self):
        """
        Flush the data to disk.
        """
        self.dataset.sync()


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


def get_days_since_reference(
    years: np.ndarray,
    months: np.ndarray,
    reference_date: cftime.datetime,
    n_months: int,
    calendar: str,
) -> np.ndarray:
    """
    Get the days since a reference date for each month.

    Args:
        years: Array of years, of shape [n_samples].
        months: Array of months, of shape [n_samples], zero-indexed.
        reference_date: Reference date for the calendar.
        n_months: Number of months to compute starting at each sample (year, month).
        calendar: Calendar to use.
    """
    months_elapsed = np.arange(n_months)
    calendar_month = (months[:, None] + months_elapsed[None, :]) % 12
    calendar_year = years[:, None] + (months[:, None] + months_elapsed[None, :]) // 12
    days_since_reference = np.zeros_like(calendar_month, dtype=np.int64)
    for i in range(calendar_month.shape[0]):
        dates_sample = xr.cftime_range(
            cftime.datetime(
                calendar_year[i, 0], calendar_month[i, 0] + 1, 1, calendar=calendar
            ),
            cftime.datetime(
                calendar_year[i, -1], calendar_month[i, -1] + 1, 1, calendar=calendar
            ),
            freq="MS",
            calendar=calendar,
        )
        days_since_reference[i, :] = (dates_sample - reference_date).days
    return days_since_reference
