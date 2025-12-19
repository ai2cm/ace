import copy
import datetime
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import cftime
import numpy as np
import torch
import xarray as xr
from netCDF4 import Dataset

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.utils import (
    DIM_INFO_HEALPIX,
    DIM_INFO_LATLON,
    get_all_names,
)
from fme.core.dataset.data_typing import VariableMetadata

LEAD_TIME_DIM = "time"
LEAD_TIME_UNITS = "months"
ENSEMBLE_DIM = "sample"
INIT_TIME = "init_time"
VALID_TIME = "valid_time"
TIME_UNITS = "days since 1970-01-01 00:00:00"
COUNTS = "counts"


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
        save_names: Sequence[str] | None,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ):
        self._target_writer = MonthlyDataWriter(
            path=path,
            label="monthly_mean_target",
            n_samples=n_samples,
            save_names=save_names,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )
        self._prediction_writer = MonthlyDataWriter(
            path=path,
            label="monthly_mean_predictions",
            n_samples=n_samples,
            save_names=save_names,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )

    def append_batch(
        self,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        batch_time: xr.DataArray,
    ):
        self._target_writer.append_batch(data=target, batch_time=batch_time)
        self._prediction_writer.append_batch(data=prediction, batch_time=batch_time)

    def flush(self):
        self._target_writer.flush()
        self._prediction_writer.flush()

    def finalize(self):
        self._target_writer.finalize()
        self._prediction_writer.finalize()


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
        save_names: Sequence[str] | None,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
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
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
        """
        filename = str(Path(path) / f"{label}.nc")
        self._save_names = save_names
        self.variable_metadata = variable_metadata
        self.coords = coords
        self.dataset = Dataset(filename, "w", format="NETCDF4")
        self.dataset.createDimension(LEAD_TIME_DIM, None)  # unlimited dimension
        self.dataset.createVariable(LEAD_TIME_DIM, "i8", (LEAD_TIME_DIM,))
        self.dataset.variables[LEAD_TIME_DIM].units = LEAD_TIME_UNITS
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
        dataset_metadata = copy.copy(dataset_metadata)
        dataset_metadata.title = f"ACE {label.replace('_', ' ')} data file"
        for key, value in dataset_metadata.as_flat_str_dict().items():
            self.dataset.setncattr(key, value)
        self._init_years = np.full([n_samples], -1, dtype=int)
        self._init_months = np.full([n_samples], -1, dtype=int)
        self._dataset_dims_created = False

    def _get_initial_year_and_month(
        self,
        years: np.ndarray,
        months: np.ndarray,
        calendar: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._init_years[0] == -1:
            self._init_years[:] = years
            self._init_months[:] = months
            reference_date = cftime.datetime(1970, 1, 1, calendar=calendar)
            n_months = 1
            days_since_reference = get_days_since_reference(
                years=years,
                months=months,
                n_months=n_months,
                reference_date=reference_date,
                calendar=calendar,
            )
            self.dataset.variables[INIT_TIME][:] = days_since_reference[:, 0]
            self.dataset.variables[INIT_TIME].calendar = calendar
            self.dataset.variables[VALID_TIME].calendar = calendar
        return (self._init_years, self._init_months)

    def _get_month_indices(self, batch_time: xr.DataArray) -> np.ndarray:
        """
        Get the month indices for the batch of data.

        If this is the first time called, the times are assumed to be the
        initial times for the data, and are stored for determining the month
        indices in this and future calls.

        Args:
            batch_time: Time coordinate for each sample in the batch, of shape
                [ensemble_member, lead_time].

        Returns:
            Month indices for the batch of data.
        """
        calendar = batch_time.dt.calendar
        years = batch_time.dt.year.values
        # datetime months are 1-indexed, we want 0-indexed
        months = batch_time.dt.month.values - 1
        init_years, init_months = self._get_initial_year_and_month(
            years=years[:, 0], months=months[:, 0], calendar=calendar
        )
        return 12 * (years - init_years[:, None]) + (months - init_months[:, None])

    def _get_variable_names_to_save(
        self, *data_varnames: Iterable[str]
    ) -> Iterable[str]:
        return get_all_names(*data_varnames, allowlist=self._save_names)

    def _extend_lead_time(self, old_size: int, new_size: int):
        init_time = self.dataset.variables[LEAD_TIME_DIM]
        init_time[old_size:new_size] = np.arange(old_size, new_size)

    def _extend_valid_time(self, old_size: int, new_size: int):
        n_months = new_size - old_size
        if n_months > 0:
            valid_time = self.dataset.variables[VALID_TIME]
            calendar = valid_time.calendar
            reference_date = cftime.datetime(1970, 1, 1, calendar=calendar)
            days_since_reference = get_days_since_reference(
                years=self._init_years,
                months=self._init_months,
                n_months=n_months,
                reference_date=reference_date,
                calendar=calendar,
                month_offset=old_size,
            )
            # use the 15th of each month, which is 14 days into the month
            valid_time[:, old_size:new_size] = days_since_reference + 14

    def _extend_variable(
        self,
        variable_name: str,
        old_size: int,
        new_size: int,
        initial_value: int | float,
    ):
        variable = self.dataset.variables[variable_name]
        variable[:, old_size:new_size] = initial_value

    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        batch_time: xr.DataArray,
    ):
        """
        Append a batch of data to the file.

        Args:
            data: Values to store.
            batch_time: Time coordinate for each sample in the batch.
        """
        n_samples_data = list(data.values())[0].shape[0]
        n_samples_time = batch_time.sizes["sample"]
        if n_samples_data != n_samples_time:
            raise ValueError(
                f"Batch size mismatch, data has {n_samples_data} samples "
                f"and batch_time has {n_samples_time} samples."
            )
        n_times_data = list(data.values())[0].shape[1]
        n_times_time = batch_time.sizes["time"]
        if n_times_data != n_times_time:
            raise ValueError(
                f"Batch time dimension mismatch, data has {n_times_data} times "
                f"and batch_time has {n_times_time} times."
            )

        if not self._dataset_dims_created:
            _dim_info = DIM_INFO_HEALPIX if "face" in self.coords else DIM_INFO_LATLON
            _ordered_names = []
            for dim in _dim_info:
                dim_size = data[next(iter(data.keys()))].shape[dim.index]
                self.dataset.createDimension(dim.name, dim_size)
                if dim.name in self.coords:
                    self.dataset.createVariable(dim.name, "f4", (dim.name,))
                    self.dataset.variables[dim.name][:] = self.coords[dim.name]
                _ordered_names.append(dim.name)
            dims = (ENSEMBLE_DIM, LEAD_TIME_DIM, *_ordered_names)
            self._dataset_dims_created = True

        save_names = self._get_variable_names_to_save(data.keys())
        months = self._get_month_indices(batch_time)
        month_min = np.min(months)
        month_range = np.max(months) - month_min + 1

        old_size = self.dataset.variables[LEAD_TIME_DIM].size
        new_size = month_min + month_range

        self._extend_lead_time(old_size, new_size)
        self._extend_valid_time(old_size, new_size)
        self._extend_variable(COUNTS, old_size, new_size, initial_value=0)

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
                if variable_name in self.variable_metadata:
                    self.dataset.variables[
                        variable_name
                    ].units = self.variable_metadata[variable_name].units
                    self.dataset.variables[
                        variable_name
                    ].long_name = self.variable_metadata[variable_name].long_name
                self.dataset.variables[variable_name].coordinates = " ".join(
                    [INIT_TIME, VALID_TIME, COUNTS]
                )

            array = data[variable_name].detach().cpu().numpy()

            # Add the data to the variable totals
            # Have to extract the data and write it back as `.at` does not play nicely
            # with netCDF4
            # We pull just the month subset we need for speed reasons
            self._extend_variable(variable_name, old_size, new_size, initial_value=0.0)
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

    def finalize(self):
        self.flush()
        self.dataset.close()


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
    month_offset: int = 0,
) -> np.ndarray:
    """
    Get the days since a reference date for each month.

    Args:
        years: Array of years, of shape [n_samples].
        months: Array of months, of shape [n_samples], zero-indexed.
        reference_date: Reference date for the calendar.
        n_months: Number of months to compute starting at each sample (year, month).
        calendar: Calendar to use.
        month_offset: Optional offset to enable computing days since the reference for
            a range of elapsed months that does not start at zero (default 0).
    """
    months_elapsed = np.arange(month_offset, month_offset + n_months)
    calendar_month = (months[:, None] + months_elapsed[None, :]) % 12
    calendar_year = years[:, None] + (months[:, None] + months_elapsed[None, :]) // 12
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
    return days_since_reference
