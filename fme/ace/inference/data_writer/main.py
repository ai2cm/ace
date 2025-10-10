import dataclasses
import datetime
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.generics.writer import WriterABC

from .dataset_metadata import DatasetMetadata
from .file_writer import FileWriter, FileWriterConfig, PairedFileWriter
from .monthly import MonthlyDataWriter, PairedMonthlyDataWriter, months_for_timesteps
from .raw import PairedRawDataWriter, RawDataWriter
from .time_coarsen import PairedTimeCoarsen, TimeCoarsen, TimeCoarsenConfig

PairedSubwriter: TypeAlias = (
    PairedRawDataWriter | PairedTimeCoarsen | PairedMonthlyDataWriter | PairedFileWriter
)

Subwriter: TypeAlias = MonthlyDataWriter | RawDataWriter | TimeCoarsen | FileWriter


@dataclasses.dataclass
class DataWriterConfig:
    """
    Configuration for inference data writers.

    Parameters:
        save_prediction_files: Whether to enable writing of netCDF files
            containing the predictions and target values.
        save_monthly_files: Whether to enable writing of netCDF files
            containing the monthly predictions and target values.
        names: Names of variables to save in the prediction and monthly
            netCDF files.
        time_coarsen: Configuration for time coarsening of written outputs to the
            raw data writer.
        files: Configuration for a sequence of individual data writers.
    """

    save_prediction_files: bool = True
    save_monthly_files: bool = True
    names: Sequence[str] | None = None
    time_coarsen: TimeCoarsenConfig | None = None
    files: list[FileWriterConfig] | None = None

    def __post_init__(self):
        if (
            not any([self.save_prediction_files, self.save_monthly_files])
            and self.names is not None
        ):
            warnings.warn(
                "names provided but all options to "
                "save subsettable output files are False."
            )

    def build_paired(
        self,
        experiment_dir: str,
        n_initial_conditions: int,
        n_timesteps: int,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> "PairedDataWriter":
        return PairedDataWriter(
            path=experiment_dir,
            n_initial_conditions=n_initial_conditions,
            n_timesteps=n_timesteps,
            timestep=timestep,
            variable_metadata=variable_metadata,
            coords=coords,
            enable_prediction_netcdfs=self.save_prediction_files,
            enable_monthly_netcdfs=self.save_monthly_files,
            save_names=self.names,
            time_coarsen=self.time_coarsen,
            dataset_metadata=dataset_metadata,
            files=self.files,
        )

    def build(
        self,
        experiment_dir: str,
        n_initial_conditions: int,
        n_timesteps: int,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> "DataWriter":
        return DataWriter(
            path=experiment_dir,
            n_initial_conditions=n_initial_conditions,
            n_timesteps=n_timesteps,
            variable_metadata=variable_metadata,
            coords=coords,
            timestep=timestep,
            enable_prediction_netcdfs=self.save_prediction_files,
            enable_monthly_netcdfs=self.save_monthly_files,
            save_names=self.names,
            time_coarsen=self.time_coarsen,
            dataset_metadata=dataset_metadata,
            files=self.files,
        )


class PairedDataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        path: str,
        n_initial_conditions: int,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        timestep: datetime.timedelta,
        enable_prediction_netcdfs: bool,
        enable_monthly_netcdfs: bool,
        save_names: Sequence[str] | None,
        dataset_metadata: DatasetMetadata,
        time_coarsen: TimeCoarsenConfig | None = None,
        files: list[FileWriterConfig] | None = None,
    ):
        """
        Args:
            path: Path to write netCDF file(s).
            n_initial_conditions: Number of ICs/ensemble members to write to the file.
            n_timesteps: Number of timesteps to write to the file.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            timestep: Timestep of the model.
            enable_prediction_netcdfs: Whether to enable writing of netCDF files
                containing the predictions and target values.
            enable_monthly_netcdfs: Whether to enable writing of netCDF files
                containing the monthly predictions and target values.
            save_names: Names of variables to save in the prediction
                and monthly netCDF files.
            dataset_metadata: Metadata for the dataset.
            time_coarsen: Configuration for time coarsening of written outputs.
            files: Configurations for individual data writers.

        """
        self._writers: list[PairedSubwriter] = []
        self.path = path
        self.coords = coords
        self.variable_metadata = variable_metadata
        self.dataset_metadata = dataset_metadata

        def _time_coarsen_builder(data_writer: PairedSubwriter) -> PairedSubwriter:
            if time_coarsen is not None:
                return time_coarsen.build_paired(data_writer)
            return data_writer

        if enable_prediction_netcdfs:
            self._writers.append(
                _time_coarsen_builder(
                    PairedRawDataWriter(
                        path=path,
                        n_initial_conditions=n_initial_conditions,
                        save_names=save_names,
                        variable_metadata=variable_metadata,
                        coords=coords,
                        dataset_metadata=dataset_metadata,
                    )
                )
            )
        if enable_monthly_netcdfs:
            self._writers.append(
                PairedMonthlyDataWriter(
                    path=path,
                    n_samples=n_initial_conditions,
                    n_timesteps=n_timesteps,
                    timestep=timestep,
                    save_names=save_names,
                    variable_metadata=variable_metadata,
                    coords=coords,
                    dataset_metadata=dataset_metadata,
                )
            )
        if files is not None:
            for writer_config in files:
                self._writers.append(
                    writer_config.build_paired(
                        experiment_dir=path,
                        n_initial_conditions=n_initial_conditions,
                        n_timesteps=n_timesteps,
                        variable_metadata=variable_metadata,
                        coords=coords,
                        dataset_metadata=dataset_metadata,
                    )
                )
        self._n_timesteps_seen = 0

    def write(self, data: PrognosticState, filename: str):
        """Eagerly write data to a single netCDF file.

        Args:
            data: the data to be written.
            filename: the filename to use for the netCDF file.
        """
        _write(
            data=data.as_batch_data(),
            path=self.path,
            filename=filename,
            variable_metadata=self.variable_metadata,
            coords=self.coords,
            dataset_metadata=self.dataset_metadata,
        )

    def append_batch(
        self,
        batch: PairedData,
    ):
        """
        Append a batch of data to the file.

        Args:
            batch: Prediction and target data.
        """
        for writer in self._writers:
            writer.append_batch(
                target=dict(batch.reference),
                prediction=dict(batch.prediction),
                start_timestep=self._n_timesteps_seen,
                batch_time=batch.time,
            )
        self._n_timesteps_seen += batch.time.shape[1]

    def flush(self):
        """
        Flush the data to disk.
        """
        for writer in self._writers:
            writer.flush()

    def finalize(self):
        for writer in self._writers:
            writer.finalize()


def _write(
    data: BatchData,
    path: str,
    filename: str,
    variable_metadata: Mapping[str, VariableMetadata],
    coords: Mapping[str, np.ndarray],
    dataset_metadata: DatasetMetadata,
):
    """Write provided data to a single netCDF at specified path/filename.

    If the data has only one timestep, the data is squeezed to remove
    the time dimension.

    Args:
        data: Batch data to written.
        path: Directory to write the netCDF file in.
        filename: filename to use for netCDF.
        variable_metadata: Metadata for each variable to be written to the file.
        coords: Coordinate data to be written to the file.
        dataset_metadata: Metadata for the dataset.
    """
    if data.time.sizes["time"] == 1:
        time_dim = data.dims.index("time")
        dims_to_write = data.dims[:time_dim] + data.dims[time_dim + 1 :]

        def maybe_squeeze(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(dim=time_dim)

        time_array = data.time.isel(time=0)
    else:
        dims_to_write = data.dims

        def maybe_squeeze(x):
            return x

        time_array = data.time

    data_arrays = {}
    for name in data.data:
        array = maybe_squeeze(data.data[name]).detach().cpu().numpy()
        data_arrays[name] = xr.DataArray(array, dims=dims_to_write)
        if name in variable_metadata:
            data_arrays[name].attrs = {
                "long_name": variable_metadata[name].long_name,
                "units": variable_metadata[name].units,
            }
    data_arrays["time"] = time_array
    ds = xr.Dataset(data_arrays, coords=coords)
    ds.attrs.update(dataset_metadata.as_flat_str_dict())
    ds.to_netcdf(str(Path(path) / filename))


class DataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        path: str,
        n_initial_conditions: int,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        timestep: datetime.timedelta,
        enable_prediction_netcdfs: bool,
        enable_monthly_netcdfs: bool,
        save_names: Sequence[str] | None,
        dataset_metadata: DatasetMetadata,
        time_coarsen: TimeCoarsenConfig | None = None,
        files: list[FileWriterConfig] | None = None,
    ):
        """
        Args:
            path: Directory within which to write netCDF file(s).
            n_initial_conditions: Number of initial conditions / timeseries
                to write to the file.
            n_timesteps: Number of timesteps to write to the file.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            timestep: Timestep of the model.
            enable_prediction_netcdfs: Whether to enable writing of netCDF files
                containing the predictions and target values.
            enable_monthly_netcdfs: Whether to enable writing of netCDF files
            save_names: Names of variables to save in the prediction
                and monthly netCDF files.
            dataset_metadata: Metadata for the dataset.
            time_coarsen: Configuration for time coarsening of raw outputs.
            files: Configurations for individual data writers.
        """
        self._writers: list[Subwriter] = []
        if "face" in coords:
            # TODO: handle writing HEALPix data
            # https://github.com/ai2cm/full-model/issues/1089
            return

        def _time_coarsen_builder(data_writer: Subwriter) -> Subwriter:
            if time_coarsen is not None:
                return time_coarsen.build(data_writer)
            return data_writer

        if enable_prediction_netcdfs:
            self._writers.append(
                _time_coarsen_builder(
                    RawDataWriter(
                        path=path,
                        label="autoregressive_predictions.nc",
                        n_initial_conditions=n_initial_conditions,
                        save_names=save_names,
                        variable_metadata=variable_metadata,
                        coords=coords,
                        dataset_metadata=dataset_metadata,
                    )
                )
            )

        if enable_monthly_netcdfs:
            self._writers.append(
                MonthlyDataWriter(
                    path=path,
                    label="predictions",
                    n_samples=n_initial_conditions,
                    n_months=months_for_timesteps(n_timesteps, timestep),
                    save_names=save_names,
                    variable_metadata=variable_metadata,
                    coords=coords,
                    dataset_metadata=dataset_metadata,
                )
            )

        if files is not None:
            for writer_config in files:
                self._writers.append(
                    writer_config.build(
                        experiment_dir=path,
                        n_initial_conditions=n_initial_conditions,
                        n_timesteps=n_timesteps,
                        variable_metadata=variable_metadata,
                        coords=coords,
                        dataset_metadata=dataset_metadata,
                    )
                )

        self.path = path
        self.variable_metadata = variable_metadata
        self.dataset_metadata = dataset_metadata
        self.coords = coords
        self._n_timesteps_seen = 0

    def append_batch(self, batch: PairedData):
        """
        Append prediction data to the file. The prognostic data
        and forcing data are merged before writing.

        Args:
            batch: Paired data to be written.
        """
        merged = {**batch.prediction, **batch.forcing}
        unpaired_batch = BatchData.new_on_device(
            data=merged,
            time=batch.time,
            labels=batch.labels,
        )
        self._append_batch(unpaired_batch)

    def _append_batch(self, batch: BatchData):
        for writer in self._writers:
            writer.append_batch(
                data=dict(batch.data),
                start_timestep=self._n_timesteps_seen,
                batch_time=batch.time,
            )
        self._n_timesteps_seen += batch.time.shape[1]

    def flush(self):
        """
        Flush the data to disk.
        """
        for writer in self._writers:
            writer.flush()

    def finalize(self):
        for writer in self._writers:
            writer.finalize()

    def write(self, data: PrognosticState, filename: str):
        _write(
            data=data.as_batch_data(),
            path=self.path,
            filename=filename,
            variable_metadata=self.variable_metadata,
            coords=self.coords,
            dataset_metadata=self.dataset_metadata,
        )
