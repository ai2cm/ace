import copy
import dataclasses
import datetime
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import cftime
import numpy as np
import numpy.typing as npt
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
LEAD_TIME_UNITS = "microseconds"
IC_DIM = "sample"
INIT_TIME = "init_time"
INIT_TIME_UNITS = "microseconds since 1970-01-01 00:00:00"
VALID_TIME = "valid_time"


@dataclasses.dataclass
class NetCDFWriterConfig:
    name: Literal["netcdf"] = "netcdf"  # defined for yaml+dacite ease of use


class PairedRawDataWriter:
    """
    Wrapper over RawDataWriter to write both target and prediction data.

    Gives the same interface as for our other writers.
    """

    def __init__(
        self,
        path: str,
        n_initial_conditions: int,
        save_names: Sequence[str] | None,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ):
        self._target_writer = RawDataWriter(
            path=path,
            label="autoregressive_target.nc",
            n_initial_conditions=n_initial_conditions,
            save_names=save_names,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )
        self._prediction_writer = RawDataWriter(
            path=path,
            label="autoregressive_predictions.nc",
            n_initial_conditions=n_initial_conditions,
            save_names=save_names,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )

    def append_batch(
        self,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        self._target_writer.append_batch(
            data=target,
            start_timestep=start_timestep,
            batch_time=batch_time,
        )
        self._prediction_writer.append_batch(
            data=prediction,
            start_timestep=start_timestep,
            batch_time=batch_time,
        )

    def flush(self):
        self._target_writer.flush()
        self._prediction_writer.flush()

    def finalize(self):
        self._target_writer.finalize()
        self._prediction_writer.finalize()


class RawDataWriter:
    """
    Write raw data to a netCDF file.
    """

    def __init__(
        self,
        path: str,
        label: str,
        n_initial_conditions: int,
        save_names: Sequence[str] | None,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ):
        """
        Args:
            path: Directory within which to write the file.
            label: Name of the file to write.
            n_initial_conditions: Number of initial conditions / timeseries
                to write to the file.
            save_names: Names of variables to save in the output file.
                If None, all provided variables will be saved.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
        """
        filename = str(Path(path) / label)
        self._save_names = save_names
        self.variable_metadata = variable_metadata
        self.coords = coords
        self.dataset = Dataset(filename, "w", format="NETCDF4")
        self.dataset.createDimension(LEAD_TIME_DIM, None)  # unlimited dimension
        self.dataset.createVariable(LEAD_TIME_DIM, "i8", (LEAD_TIME_DIM,))
        self.dataset.variables[LEAD_TIME_DIM].units = LEAD_TIME_UNITS
        self.dataset.createDimension(IC_DIM, n_initial_conditions)
        self.dataset.createVariable(INIT_TIME, "i8", (IC_DIM,))
        self.dataset.variables[INIT_TIME].units = INIT_TIME_UNITS
        self.dataset.createVariable(VALID_TIME, "i8", (IC_DIM, LEAD_TIME_DIM))
        self.dataset.variables[VALID_TIME].units = INIT_TIME_UNITS
        self._dataset_dims_created = False
        dataset_metadata = copy.copy(dataset_metadata)
        dataset_metadata.title = (
            f"ACE {label.removesuffix('.nc').replace('_', ' ')} data file"
        )
        for key, value in dataset_metadata.as_flat_str_dict().items():
            self.dataset.setncattr(key, value)

    def _get_variable_names_to_save(
        self, *data_varnames: Iterable[str]
    ) -> Iterable[str]:
        return get_all_names(*data_varnames, allowlist=self._save_names)

    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        """
        Append a batch of data to the file.

        Args:
            data: Data to be written to file.
            start_timestep: Timestep (lead time dim) at which to start writing.
            batch_time: Time coordinate for each sample in the batch.
        """
        if self.dataset is None:
            return
        n_samples_data = list(data.values())[0].shape[0]
        n_samples_time = batch_time.sizes["sample"]
        if n_samples_data != n_samples_time:
            raise ValueError(
                f"Batch size mismatch, data has {n_samples_data} samples "
                f"and times has {n_samples_time} samples."
            )
        n_times_data = list(data.values())[0].shape[1]
        n_times_time = batch_time.sizes["time"]
        if n_times_data != n_times_time:
            raise ValueError(
                f"Batch time dimension mismatch, data has {n_times_data} times "
                f"and time has {n_times_time} times."
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
            dims = (IC_DIM, LEAD_TIME_DIM, *_ordered_names)
            self._dataset_dims_created = True

        save_names = self._get_variable_names_to_save(data.keys())
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
                    [INIT_TIME, VALID_TIME]
                )

            data_numpy = data[variable_name].detach().cpu().numpy()
            # Append the data to the variables
            self.dataset.variables[variable_name][
                :,
                start_timestep : start_timestep + data_numpy.shape[1],
                :,
            ] = data_numpy

        # handle time dimensions
        if not hasattr(self.dataset.variables[INIT_TIME], "calendar"):
            self.dataset.variables[INIT_TIME].calendar = batch_time.dt.calendar
        if not hasattr(self.dataset.variables[VALID_TIME], "calendar"):
            self.dataset.variables[VALID_TIME].calendar = batch_time.dt.calendar

        if start_timestep == 0:
            init_times: np.ndarray = batch_time.isel(time=0).values
            init_times_numeric: np.ndarray = cftime.date2num(
                init_times,
                units=self.dataset.variables[INIT_TIME].units,
                calendar=self.dataset.variables[INIT_TIME].calendar,
            )
            self.dataset.variables[INIT_TIME][:] = init_times_numeric
        else:
            init_times_numeric = self.dataset.variables[INIT_TIME][:]
            init_times_numeric = (
                init_times_numeric.filled()
            )  # convert masked array to ndarray
            init_times = cftime.num2date(
                init_times_numeric,
                units=self.dataset.variables[INIT_TIME].units,
                calendar=self.dataset.variables[INIT_TIME].calendar,
            )
        lead_time_microseconds = get_batch_lead_time_microseconds(
            init_times,
            batch_time.values,
        )
        self.dataset.variables[LEAD_TIME_DIM][
            start_timestep : start_timestep + lead_time_microseconds.shape[0]
        ] = lead_time_microseconds

        valid_times_numeric: np.ndarray = cftime.date2num(
            batch_time.values,
            units=self.dataset.variables[VALID_TIME].units,
            calendar=self.dataset.variables[VALID_TIME].calendar,
        )
        self.dataset.variables[VALID_TIME][
            :,
            start_timestep : start_timestep + lead_time_microseconds.shape[0],
        ] = valid_times_numeric

        self.dataset.sync()  # Flush the data to disk

    def flush(self):
        """
        Flush the data to disk.
        """
        self.dataset.sync()

    def finalize(self):
        self.flush()
        self.dataset.close()


def get_batch_lead_time_microseconds(
    init_time: npt.NDArray[cftime.datetime], batch_time: npt.NDArray[cftime.datetime]
) -> npt.NDArray[np.int64]:
    """
    Get the lead time in seconds for the batch.
    Assert that they are the same for each sample.

    Args:
        init_time: Initialization time for each sample in the batch.
        batch_time: Array of time coordinates for each sample in the batch.

    Returns:
        Lead time in microseconds for the batch
    """
    if init_time.shape[0] != batch_time.shape[0]:
        raise ValueError(
            f"Number of init times ({len(init_time)}) must "
            f"match number of batch times ({len(batch_time)})"
        )
    # Carry out timedelta arithmetic in NumPy arrays to avoid xarray's automatic
    # casting of datetime.timedelta objects to timedelta64[ns] values, which would
    # unnecessarily limit the lead time coordinate to containing values less than
    # ~292 years. See
    # https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
    # for more details on the limits of various precision timedeltas.
    lead_time: npt.NDArray[datetime.timedelta] = (  # type: ignore
        batch_time - init_time[:, None]
    )
    lead_time_microseconds: npt.NDArray[np.int64] = (
        lead_time // datetime.timedelta(microseconds=1)
    ).astype(np.int64)
    if not np.all(lead_time_microseconds == lead_time_microseconds[0, :]):
        raise ValueError("Lead times are not the same for each sample in the batch.")
    return lead_time_microseconds[0, :]
