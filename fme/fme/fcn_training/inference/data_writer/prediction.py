import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

import cftime
import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from netCDF4 import Dataset

from fme.core.data_loading.data_typing import VariableMetadata

LEAD_TIME_DIM = "lead"
LEAD_TIME_UNITS = "microseconds"
SAMPLE_DIM = "sample"
INIT_TIME = "init"
INIT_TIME_UNITS = "microseconds since 1970-01-01 00:00:00"
VALID_TIME = "valid_time"


class PredictionDataWriter:
    """
    Write raw prediction data to a netCDF file.
    """

    def __init__(
        self,
        path: str,
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
            n_timesteps: Number of timesteps to write to the file. Note that the
                timesteps dimensions is encoded as "lead" time, i.e., time since
                initialization. See
                https://climpred.readthedocs.io/en/stable/setting-up-data.html
                for the justification of the "lead" dim name and units.
            save_names: Names of variables to save in the predictions netcdf file.
                If None, all predicted variables will be saved.
            metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.


        """
        self.path = path
        filename = str(Path(path) / "autoregressive_predictions.nc")
        self._save_names = save_names
        self.metadata = metadata
        self.coords = coords
        self.dataset = Dataset(filename, "w", format="NETCDF4")
        self.dataset.createDimension("source", 2)
        self.dataset.createDimension(LEAD_TIME_DIM, None)  # unlimited dimension
        self.dataset.createVariable(LEAD_TIME_DIM, "i8", (LEAD_TIME_DIM,))
        self.dataset.variables[LEAD_TIME_DIM].units = LEAD_TIME_UNITS
        self.dataset.createDimension(SAMPLE_DIM, n_samples)
        self.dataset.createVariable(INIT_TIME, "i8", (SAMPLE_DIM,))
        self.dataset.variables[INIT_TIME].units = INIT_TIME_UNITS
        self.dataset.createVariable(VALID_TIME, "i8", (SAMPLE_DIM, LEAD_TIME_DIM))
        self.dataset.variables[VALID_TIME].units = INIT_TIME_UNITS
        self.dataset.createVariable("source", "str", ("source",))
        self.dataset.variables["source"][:] = np.array(["target", "prediction"])
        self._n_lat: Optional[int] = None
        self._n_lon: Optional[int] = None

    def _get_variable_names_to_save(
        self, *data_varnames: Iterable[str]
    ) -> Iterable[str]:
        variables: Set[str] = set()
        for varnames in data_varnames:
            variables = variables.union(set(varnames))
        if self._save_names is None:
            return variables
        else:
            return variables.intersection(set(self._save_names))

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        start_sample: int,
        batch_times: xr.DataArray,
    ):
        """
        Append a batch of data to the file.

        Args:
            target: Target data.
            prediction: Prediction data.
            start_timestep: Timestep (lead time dim) at which to start writing.
            start_sample: Sample (initialization time dim) at which to start writing.
            batch_times: Time coordinates for each sample in the batch.
        """
        n_samples_data = list(target.values())[0].shape[0]
        n_samples_time = batch_times.sizes["sample"]
        if n_samples_data != n_samples_time:
            raise ValueError(
                f"Batch size mismatch, data has {n_samples_data} samples "
                f"and times has {n_samples_time} samples."
            )
        n_times_data = list(target.values())[0].shape[1]
        n_times_time = batch_times.sizes["time"]
        if n_times_data != n_times_time:
            raise ValueError(
                f"Batch time dimension mismatch, data has {n_times_data} times "
                f"and times has {n_times_time} times."
            )

        if self._n_lat is None:
            self._n_lat = target[next(iter(target.keys()))].shape[-2]
            self.dataset.createDimension("lat", self._n_lat)
            if "lat" in self.coords:
                self.dataset.createVariable("lat", "f4", ("lat",))
                self.dataset.variables["lat"][:] = self.coords["lat"]
        if self._n_lon is None:
            self._n_lon = target[next(iter(target.keys()))].shape[-1]
            self.dataset.createDimension("lon", self._n_lon)
            if "lon" in self.coords:
                self.dataset.createVariable("lon", "f4", ("lon",))
                self.dataset.variables["lon"][:] = self.coords["lon"]

        dims = ("source", SAMPLE_DIM, LEAD_TIME_DIM, "lat", "lon")
        save_names = self._get_variable_names_to_save(target.keys(), prediction.keys())
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

            # Target and prediction may not have the same variables.
            # The netCDF contains a "source" dimension for all variables
            # and will have NaN for missing data.
            if variable_name in target:
                target_numpy = target[variable_name].cpu().numpy()
            else:
                target_numpy = np.full(
                    shape=target[next(iter(target.keys()))].shape, fill_value=np.nan
                )
            if variable_name in prediction:
                prediction_numpy = prediction[variable_name].cpu().numpy()
            else:
                prediction_numpy = np.full(
                    shape=prediction[next(iter(prediction.keys()))].shape,
                    fill_value=np.nan,
                )

            # Broadcast the corresponding dimension to match with the
            # 'source' dimension of the variable in the netCDF file
            target_numpy = np.expand_dims(target_numpy, dims.index("source"))
            prediction_numpy = np.expand_dims(prediction_numpy, dims.index("source"))

            n_samples_total = self.dataset.variables[variable_name].shape[1]
            if start_sample + target_numpy.shape[1] > n_samples_total:
                raise ValueError(
                    f"Batch size {target_numpy.shape[1]} starting at sample "
                    f"{start_sample} "
                    "is too large to fit in the netCDF file with sample "
                    f"dimension of length {n_samples_total}."
                )
            # Append the data to the variables
            self.dataset.variables[variable_name][
                :,
                start_sample : start_sample + target_numpy.shape[1],
                start_timestep : start_timestep + target_numpy.shape[2],
                :,
            ] = np.concatenate([target_numpy, prediction_numpy], axis=0)

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
            self.dataset.variables[INIT_TIME][
                start_sample : start_sample + batch_times.sizes["sample"]
            ] = init_times_numeric
        else:
            init_times_numeric = self.dataset.variables[INIT_TIME][
                start_sample : start_sample + batch_times.sizes["sample"]
            ]
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
            start_sample : start_sample + batch_times.sizes["sample"],
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
    lead_times: npt.NDArray[datetime.timedelta] = batch_times - init_times[:, None]
    lead_times_microseconds: npt.NDArray[np.int64] = (
        lead_times // datetime.timedelta(microseconds=1)
    ).astype(np.int64)
    if not np.all(lead_times_microseconds == lead_times_microseconds[0, :]):
        raise ValueError("Lead times are not the same for each sample in the batch.")
    return lead_times_microseconds[0, :]
