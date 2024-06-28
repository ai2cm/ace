import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import cftime
import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from netCDF4 import Dataset

from fme.ace.inference.data_writer.utils import get_all_names
from fme.core.data_loading.data_typing import VariableMetadata

LEAD_TIME_DIM = "time"
LEAD_TIME_UNITS = "microseconds"
SAMPLE_DIM = "sample"
INIT_TIME = "init_time"
INIT_TIME_UNITS = "microseconds since 1970-01-01 00:00:00"
VALID_TIME = "valid_time"


class PairedRawDataWriter:
    """
    Wrapper over RawDataWriter to write both target and prediction data.

    Gives the same interface as for our other writers.
    """

    def __init__(
        self,
        path: str,
        n_samples: int,
        save_names: Optional[Sequence[str]],
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
    ):
        self._target_writer = RawDataWriter(
            path=path,
            label="autoregressive_target.nc",
            n_samples=n_samples,
            save_names=save_names,
            metadata=metadata,
            coords=coords,
        )
        self._prediction_writer = RawDataWriter(
            path=path,
            label="autoregressive_predictions.nc",
            n_samples=n_samples,
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
            data=target,
            start_timestep=start_timestep,
            batch_times=batch_times,
        )
        self._prediction_writer.append_batch(
            data=prediction,
            start_timestep=start_timestep,
            batch_times=batch_times,
        )

    def flush(self):
        self._target_writer.flush()
        self._prediction_writer.flush()


class RawDataWriter:
    """
    Write raw data to a netCDF file.
    """

    def __init__(
        self,
        path: str,
        label: str,
        n_samples: int,
        save_names: Optional[Sequence[str]],
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
    ):
        """
        Args:
            filename: Path to write netCDF file(s).
            n_samples: Number of samples to write to the file. This might correspond
                to a number of initial conditions, or some other grouping of samples.
            save_names: Names of variables to save in the output file.
                If None, all provided variables will be saved.
            metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
        """
        filename = str(Path(path) / label)
        self._save_names = save_names
        self.metadata = metadata
        self.coords = coords
        self.dataset = Dataset(filename, "w", format="NETCDF4")
        self.dataset.createDimension(LEAD_TIME_DIM, None)  # unlimited dimension
        self.dataset.createVariable(LEAD_TIME_DIM, "i8", (LEAD_TIME_DIM,))
        self.dataset.variables[LEAD_TIME_DIM].units = LEAD_TIME_UNITS
        self.dataset.createDimension(SAMPLE_DIM, n_samples)
        self.dataset.createVariable(INIT_TIME, "i8", (SAMPLE_DIM,))
        self.dataset.variables[INIT_TIME].units = INIT_TIME_UNITS
        self.dataset.createVariable(VALID_TIME, "i8", (SAMPLE_DIM, LEAD_TIME_DIM))
        self.dataset.variables[VALID_TIME].units = INIT_TIME_UNITS
        self._n_lat: Optional[int] = None
        self._n_lon: Optional[int] = None

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
            data: Data to be written to file.
            start_timestep: Timestep (lead time dim) at which to start writing.
            batch_times: Time coordinates for each sample in the batch.
        """
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

        dims = (SAMPLE_DIM, LEAD_TIME_DIM, "lat", "lon")
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

            data_numpy = data[variable_name].cpu().numpy()
            # Append the data to the variables
            self.dataset.variables[variable_name][
                :,
                start_timestep : start_timestep + data_numpy.shape[1],
                :,
            ] = data_numpy

        # handle time dimensions
        if not hasattr(self.dataset.variables[INIT_TIME], "calendar"):
            self.dataset.variables[INIT_TIME].calendar = batch_times.dt.calendar
        if not hasattr(self.dataset.variables[VALID_TIME], "calendar"):
            self.dataset.variables[VALID_TIME].calendar = batch_times.dt.calendar

        if start_timestep == 0:
            init_times: np.ndarray = batch_times.isel(time=0).values
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
        lead_times_microseconds = get_batch_lead_times_microseconds(
            init_times,
            batch_times.values,
        )
        self.dataset.variables[LEAD_TIME_DIM][
            start_timestep : start_timestep + lead_times_microseconds.shape[0]
        ] = lead_times_microseconds

        valid_times_numeric: np.ndarray = cftime.date2num(
            batch_times.values,
            units=self.dataset.variables[VALID_TIME].units,
            calendar=self.dataset.variables[VALID_TIME].calendar,
        )
        self.dataset.variables[VALID_TIME][
            :,
            start_timestep : start_timestep + lead_times_microseconds.shape[0],
        ] = valid_times_numeric

        self.dataset.sync()  # Flush the data to disk

    def flush(self):
        """
        Flush the data to disk.
        """
        self.dataset.sync()


def get_batch_lead_times_microseconds(
    init_times: npt.NDArray[cftime.datetime], batch_times: npt.NDArray[cftime.datetime]
) -> npt.NDArray[np.int64]:
    """
    Get the lead times in seconds for the batch.
    Assert that they are the same for each sample.

    Args:
        init_times: Initialization time for each sample in the batch.
        batch_times: Full array of times for each sample in the batch.

    Returns:
        Lead times in microseconds for the batch
    """
    if init_times.shape[0] != batch_times.shape[0]:
        raise ValueError(
            f"Number of init times ({len(init_times)}) must "
            f"match number of batch times ({len(batch_times)})"
        )
    # Carry out timedelta arithmetic in NumPy arrays to avoid xarray's automatic
    # casting of datetime.timedelta objects to timedelta64[ns] values, which would
    # unnecessarily limit the lead time coordinate to containing values less than
    # ~292 years. See
    # https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
    # for more details on the limits of various precision timedeltas.
    lead_times: npt.NDArray[datetime.timedelta] = (  # type: ignore
        batch_times - init_times[:, None]
    )
    lead_times_microseconds: npt.NDArray[np.int64] = (
        lead_times // datetime.timedelta(microseconds=1)
    ).astype(np.int64)
    if not np.all(lead_times_microseconds == lead_times_microseconds[0, :]):
        raise ValueError("Lead times are not the same for each sample in the batch.")
    return lead_times_microseconds[0, :]
