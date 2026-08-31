import copy
import datetime
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import cftime
import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from netCDF4 import Dataset

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.monthly_aggregation import (
    COUNTS,
    ENSEMBLE_DIM,
    INIT_TIME,
    LEAD_TIME_DIM,
    LEAD_TIME_UNITS,
    TIME_UNITS,
    VALID_TIME,
    MonthIndexer,
    add_data,
    get_valid_times,
)
from fme.ace.inference.data_writer.raw import infer_calendar
from fme.ace.inference.data_writer.utils import (
    DIM_INFO_HEALPIX,
    DIM_INFO_LATLON,
    get_all_names,
)
from fme.core.cloud import is_local
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.writer import DATETIME_ENCODING_UNITS


class PairedMonthlyDataWriter:
    """
    Wrapper over MonthlyDataWriter to write both target and prediction data
    to the same file.

    Gives the same interface as for our other writers.
    """

    def __init__(
        self,
        path: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
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
            initial_condition_times=initial_condition_times,
            save_names=save_names,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )
        self._prediction_writer = MonthlyDataWriter(
            path=path,
            label="monthly_mean_predictions",
            initial_condition_times=initial_condition_times,
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
    Write monthly mean data and sample counts to a netCDF file.

    Each batch is folded into the stored means using the stored counts, so the
    values on disk are means over however many timesteps have been written for
    that calendar month, and the counts say how many that is.
    """

    def __init__(
        self,
        path: str,
        label: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        save_names: Sequence[str] | None,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ):
        """
        Args:
            path: Directory to write netCDF file(s).
            label: Label to append to the filename.
            initial_condition_times: 1D array of initial condition times
                (start time for each inference run).
            n_months: Number of months to write to the file.
            save_names: Names of variables to save in the predictions netcdf file.
                If None, all predicted variables will be saved.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
        """
        if not is_local(path):
            raise ValueError("MonthlyDataWriter only supports local file systems.")
        filename = str(Path(path) / f"{label}.nc")
        n_initial_conditions = len(initial_condition_times)
        calendar = infer_calendar(initial_condition_times)
        self._save_names = save_names
        self.variable_metadata = variable_metadata
        self.coords = coords
        self.dataset = Dataset(filename, "w", format="NETCDF4")
        self.dataset.createDimension(LEAD_TIME_DIM, None)  # unlimited dimension
        self.dataset.createVariable(LEAD_TIME_DIM, "i8", (LEAD_TIME_DIM,))
        self.dataset.variables[LEAD_TIME_DIM].units = LEAD_TIME_UNITS
        self.dataset.createDimension(ENSEMBLE_DIM, n_initial_conditions)
        self.dataset.createVariable(INIT_TIME, "i8", (ENSEMBLE_DIM,))
        self.dataset.variables[INIT_TIME].units = DATETIME_ENCODING_UNITS
        self.dataset.variables[INIT_TIME].calendar = calendar
        self.dataset.variables[INIT_TIME][:] = cftime.date2num(
            initial_condition_times,
            units=self.dataset.variables[INIT_TIME].units,
            calendar=self.dataset.variables[INIT_TIME].calendar,
        )
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
        self.dataset.variables[VALID_TIME].calendar = calendar
        dataset_metadata = copy.copy(dataset_metadata)
        dataset_metadata.title = f"ACE {label.replace('_', ' ')} data file"
        for key, value in dataset_metadata.as_flat_str_dict().items():
            self.dataset.setncattr(key, value)
        self._month_indexer = MonthIndexer(n_initial_conditions)
        self._dataset_dims_created = False

    def _get_variable_names_to_save(
        self, *data_varnames: Iterable[str]
    ) -> Iterable[str]:
        return get_all_names(*data_varnames, allowlist=self._save_names)

    def _extend_lead_time(self, old_size: int, new_size: int):
        lead_time = self.dataset.variables[LEAD_TIME_DIM]
        lead_time[old_size:new_size] = np.arange(old_size, new_size)

    def _extend_valid_time(self, old_size: int, new_size: int):
        n_months = new_size - old_size
        if n_months > 0:
            valid_time = self.dataset.variables[VALID_TIME]
            valid_time[:, old_size:new_size] = get_valid_times(
                init_years=self._month_indexer.init_years,
                init_months=self._month_indexer.init_months,
                n_months=n_months,
                calendar=valid_time.calendar,
                month_offset=old_size,
            )

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
        months = self._month_indexer.month_indices(batch_time)
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
                    for attr, val in (
                        self.variable_metadata[variable_name].as_attrs().items()
                    ):
                        setattr(self.dataset.variables[variable_name], attr, val)
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
