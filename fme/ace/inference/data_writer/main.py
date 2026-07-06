import dataclasses
import datetime
import os
import warnings
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import cftime
import fsspec
import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.core.cloud import to_netcdf_via_inter_filesystem_copy
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.generics.writer import WriterABC
from fme.core.stepper_state import StepperState

from .dataset_metadata import DatasetMetadata
from .file_writer import FileWriter, FileWriterConfig, PairedFileWriter
from .monthly import MonthlyDataWriter, PairedMonthlyDataWriter
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
        files: Configuration for a sequence of individual data writers. Each data
            writer must have a unique label to avoid filename collisions.
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
        all_filenames = self._get_all_filenames()
        if len(set(all_filenames)) != len(all_filenames):
            raise ValueError(
                "Duplicate filenames found in file writer configurations. "
                f"Filenames: {all_filenames}"
            )

    def _get_all_filenames(self) -> list[str]:
        filenames = []
        for file in self.files or []:
            filenames.extend(file.filenames)
        return filenames

    def validate_time_coarsen(self, forward_steps_in_memory: int, n_forward_steps: int):
        """Validate time coarsening (top-level and per-file) against the schedule."""
        if self.time_coarsen is not None:
            self.time_coarsen.validate(forward_steps_in_memory, n_forward_steps)
        for file_config in self.files or []:
            file_config.validate_time_coarsen(forward_steps_in_memory, n_forward_steps)

    def build_paired(
        self,
        experiment_dir: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        n_timesteps: int,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> "PairedDataWriter":
        writers: list[PairedSubwriter] = []
        if self.save_prediction_files:
            raw_writer: PairedSubwriter = PairedRawDataWriter(
                path=experiment_dir,
                initial_condition_times=initial_condition_times,
                save_names=self.names,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
            if self.time_coarsen is not None:
                raw_writer = self.time_coarsen.build_paired(raw_writer)
            writers.append(raw_writer)
        if self.save_monthly_files:
            writers.append(
                PairedMonthlyDataWriter(
                    path=experiment_dir,
                    initial_condition_times=initial_condition_times,
                    n_timesteps=n_timesteps,
                    timestep=timestep,
                    save_names=self.names,
                    variable_metadata=variable_metadata,
                    coords=coords,
                    dataset_metadata=dataset_metadata,
                )
            )
        for writer_config in self.files or []:
            writers.append(
                writer_config.build_paired(
                    experiment_dir=experiment_dir,
                    initial_condition_times=initial_condition_times,
                    n_timesteps=n_timesteps,
                    timestep=timestep,
                    variable_metadata=variable_metadata,
                    coords=coords,
                    dataset_metadata=dataset_metadata,
                )
            )
        return PairedDataWriter(
            writers=writers,
            path=experiment_dir,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )

    def build(
        self,
        experiment_dir: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        n_timesteps: int,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> "DataWriter":
        writers: list[Subwriter] = []
        # TODO: handle writing HEALPix data
        # https://github.com/ai2cm/full-model/issues/1089
        if "face" not in coords:
            if self.save_prediction_files:
                raw_writer: Subwriter = RawDataWriter(
                    path=experiment_dir,
                    label="autoregressive_predictions",
                    initial_condition_times=initial_condition_times,
                    save_names=self.names,
                    variable_metadata=variable_metadata,
                    coords=coords,
                    dataset_metadata=dataset_metadata,
                )
                if self.time_coarsen is not None:
                    raw_writer = self.time_coarsen.build(raw_writer)
                writers.append(raw_writer)
            if self.save_monthly_files:
                writers.append(
                    MonthlyDataWriter(
                        path=experiment_dir,
                        label="monthly_mean_predictions",
                        initial_condition_times=initial_condition_times,
                        save_names=self.names,
                        variable_metadata=variable_metadata,
                        coords=coords,
                        dataset_metadata=dataset_metadata,
                    )
                )
            for writer_config in self.files or []:
                writers.append(
                    writer_config.build(
                        experiment_dir=experiment_dir,
                        initial_condition_times=initial_condition_times,
                        n_timesteps=n_timesteps,
                        timestep=timestep,
                        variable_metadata=variable_metadata,
                        coords=coords,
                        dataset_metadata=dataset_metadata,
                    )
                )
        return DataWriter(
            writers=writers,
            path=experiment_dir,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )


class PairedDataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        writers: list[PairedSubwriter],
        path: str,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ):
        """
        Args:
            writers: The sub-writers to dispatch each batch to, already built by
                `DataWriterConfig.build_paired`.
            path: Path to write single-snapshot netCDF file(s) via `write`.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
        """
        self._writers = writers
        self.path = path
        self.coords = coords
        self.variable_metadata = variable_metadata
        self.dataset_metadata = dataset_metadata

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

    def write_stepper_state(self, data: PrognosticState, filename: str):
        _write_stepper_state(data, path=self.path, filename=filename)

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
                batch_time=batch.time,
            )

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
            data_arrays[name].attrs = variable_metadata[name].as_attrs()
    data_arrays["time"] = time_array
    ds = xr.Dataset(data_arrays, coords=coords)
    ds.attrs.update(dataset_metadata.as_flat_str_dict())
    to_netcdf_via_inter_filesystem_copy(ds, os.path.join(path, filename))


def _write_stepper_state(data: PrognosticState, path: str, filename: str):
    """Serialize the prognostic state's ``StepperState`` to a restart file.

    A no-op when there is no ``StepperState`` to save. Uses ``fsspec`` so the
    ``.pt`` file lands beside a local or remote ``restart.nc``.
    """
    stepper_state: StepperState | None = data.as_batch_data().stepper_state
    if stepper_state is None:
        return
    # Serialize on CPU so the state restores independently of the device the
    # rollout ran on (the random generator is already CPU by construction).
    with fsspec.open(os.path.join(path, filename), "wb") as f:
        torch.save(stepper_state.to_cpu().to_state_dict(), f)


class DataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        writers: list[Subwriter],
        path: str,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ):
        """
        Args:
            writers: The sub-writers to dispatch each batch to, already built by
                `DataWriterConfig.build`. Empty for HEALPix (`face`)
                coordinates, which are not yet supported.
            path: Directory within which to write single-snapshot netCDF file(s)
                via `write`.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
        """
        self._writers = writers
        self.path = path
        self.variable_metadata = variable_metadata
        self.dataset_metadata = dataset_metadata
        self.coords = coords

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
                batch_time=batch.time,
            )

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

    def write_stepper_state(self, data: PrognosticState, filename: str):
        _write_stepper_state(data, path=self.path, filename=filename)
