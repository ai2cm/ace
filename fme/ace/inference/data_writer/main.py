import dataclasses
import datetime
import os
import warnings
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import cftime
import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import (
    _RESERVED_PREFIX,
    BatchData,
    GatheredBatchData,
    PairedData,
    PrognosticState,
)
from fme.core.cloud import to_netcdf_via_inter_filesystem_copy
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.writer import WriterABC
from fme.core.step.step_diagnostics import StepDiagnostics

from .dataset_metadata import DatasetMetadata
from .file_writer import FileWriter, FileWriterConfig, PairedFileWriter
from .monthly import MonthlyDataWriter, PairedMonthlyDataWriter
from .raw import PairedRawDataWriter, RawDataWriter
from .step_diagnostics import STEP_DIAGNOSTICS_DIR, StepDiagnosticsWriter
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
        save_step_diagnostics: Whether to enable writing of netCDF files
            containing per-step diagnostics of the prediction, one file per
            named diagnostics dataset under a ``step_diagnostics/`` directory
            (currently the corrector's per-step correction delta, written to
            ``step_diagnostics/correction_deltas.nc``). Nothing is written
            when no corrector modified anything.
        names: Names of variables to save in the prediction and monthly
            netCDF files.
        time_coarsen: Configuration for time coarsening of written outputs to the
            raw data writer.
        files: Configuration for a sequence of individual data writers. Each data
            writer must have a unique label to avoid filename collisions.
    """

    save_prediction_files: bool = True
    save_monthly_files: bool = True
    save_step_diagnostics: bool = False
    names: Sequence[str] | None = None
    time_coarsen: TimeCoarsenConfig | None = None
    files: list[FileWriterConfig] | None = None

    def __post_init__(self):
        if (
            not any(
                [
                    self.save_prediction_files,
                    self.save_monthly_files,
                    self.save_step_diagnostics,
                ]
            )
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

    @property
    def has_subwriters_enabled(self) -> bool:
        """True when any per-timestep output writer is enabled."""
        return (
            self.save_prediction_files
            or self.save_monthly_files
            or self.save_step_diagnostics
            or bool(self.files)
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
        dist = Distributed.get_instance()
        is_multi_gpu = dist.total_data_parallel_ranks > 1
        is_root = dist.is_data_parallel_root()
        if is_multi_gpu:
            all_ic_times = dist.data_parallel_gather_object(initial_condition_times)
            if is_root:
                if all_ic_times is None:
                    raise RuntimeError(
                        "data_parallel_gather_object returned None on "
                        "data-parallel root"
                    )
                initial_condition_times = np.concatenate(all_ic_times)
        writers: list[PairedSubwriter] = []
        if self.save_prediction_files and (not is_multi_gpu or is_root):
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
        if self.save_monthly_files and (not is_multi_gpu or is_root):
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
        if not is_multi_gpu or is_root:
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
        step_diagnostics_writer: StepDiagnosticsWriter | None = None
        if not is_multi_gpu or is_root:
            step_diagnostics_writer = self._build_step_diagnostics_writer(
                experiment_dir=experiment_dir,
                initial_condition_times=initial_condition_times,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
        return PairedDataWriter(
            writers=writers,
            path=experiment_dir,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
            step_diagnostics_writer=step_diagnostics_writer,
        )

    def _build_step_diagnostics_writer(
        self,
        experiment_dir: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> StepDiagnosticsWriter | None:
        if not self.save_step_diagnostics:
            return None

        def writer_factory(name: str) -> RawDataWriter | TimeCoarsen:
            path = os.path.join(experiment_dir, STEP_DIAGNOSTICS_DIR)
            os.makedirs(path, exist_ok=True)
            writer: RawDataWriter | TimeCoarsen = RawDataWriter(
                path=path,
                label=name,
                initial_condition_times=initial_condition_times,
                save_names=self.names,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
            if self.time_coarsen is not None:
                writer = self.time_coarsen.build(writer)
            return writer

        return StepDiagnosticsWriter(writer_factory)

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
        dist = Distributed.get_instance()
        is_multi_gpu = dist.total_data_parallel_ranks > 1
        is_root = dist.is_data_parallel_root()
        if is_multi_gpu:
            all_ic_times = dist.data_parallel_gather_object(initial_condition_times)
            if is_root:
                if all_ic_times is None:
                    raise RuntimeError(
                        "data_parallel_gather_object returned None on "
                        "data-parallel root"
                    )
                initial_condition_times = np.concatenate(all_ic_times)
        build_subwriters = not is_multi_gpu or is_root
        writers: list[Subwriter] = []
        # TODO: handle writing HEALPix data
        # https://github.com/ai2cm/full-model/issues/1089
        if "face" not in coords and build_subwriters:
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
                        timestep=timestep,
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
        step_diagnostics_writer: StepDiagnosticsWriter | None = None
        if build_subwriters:
            step_diagnostics_writer = self._build_step_diagnostics_writer(
                experiment_dir=experiment_dir,
                initial_condition_times=initial_condition_times,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
        return DataWriter(
            writers=writers,
            path=experiment_dir,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
            step_diagnostics_writer=step_diagnostics_writer,
        )


def _gather_tensor_dict(
    data: dict[str, torch.Tensor], dist: Distributed
) -> dict[str, torch.Tensor] | None:
    """Gather each tensor across data-parallel ranks, concatenating along dim 0.

    Returns the concatenated dict on the data-parallel root, None on non-root.
    Keys are sorted before calling collectives so every rank uses the same order.
    """
    device = get_device()
    gathered: dict[str, torch.Tensor] = {}
    for name in sorted(data):
        rank_tensors = dist.data_parallel_gather(data[name].to(device).contiguous())
        if dist.is_data_parallel_root():
            if rank_tensors is None:
                raise RuntimeError(
                    "_gather_tensor_dict: gather returned None on data-parallel root"
                )
            gathered[name] = torch.cat(rank_tensors, dim=0).cpu()
    if not dist.is_data_parallel_root():
        return None
    return gathered


def _gather_time(time: xr.DataArray, dist: Distributed) -> xr.DataArray | None:
    """Gather an xr.DataArray time coordinate across data-parallel ranks.

    Returns the concatenated time on the data-parallel root, None on non-root.
    """
    parts = dist.data_parallel_gather_object(time)
    if not dist.is_data_parallel_root():
        return None
    if parts is None:
        raise RuntimeError(
            "_gather_time: gather_object returned None on data-parallel root"
        )
    return xr.concat(parts, dim="sample")


class PairedDataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        writers: list[PairedSubwriter],
        path: str,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
        step_diagnostics_writer: StepDiagnosticsWriter | None = None,
    ):
        """
        Args:
            writers: The sub-writers to dispatch each batch to, already built by
                `DataWriterConfig.build_paired`.
            path: Path to write single-snapshot netCDF file(s) via `write`.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
            step_diagnostics_writer: Writer for per-step diagnostics of the
                prediction, or None to not write them. Held separately from
                the paired sub-writers: it consumes only the prediction's
                diagnostics, not the target/prediction pair.
        """
        self._writers = writers
        self._step_diagnostics_writer = step_diagnostics_writer
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
        _gather_and_write(
            batch=data.as_batch_data(),
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

        Under multi-GPU inference, gathers tensor data and time from all
        data-parallel ranks to root before dispatching to subwriters.
        Non-root ranks return early (their subwriter lists are empty, but the
        gather collectives must still be called by every rank).

        Args:
            batch: Prediction and target data.
        """
        dist = Distributed.get_instance()
        if dist.total_data_parallel_ranks > 1:
            prediction = _gather_tensor_dict(dict(batch.prediction), dist)
            target = _gather_tensor_dict(dict(batch.reference), dist)
            time = _gather_time(batch.time, dist)
            if batch.step_diagnostics is not None:
                gathered_diag = _gather_tensor_dict(
                    dict(batch.step_diagnostics.delta), dist
                )
            else:
                gathered_diag = None
            if not dist.is_data_parallel_root():
                return
            # On root, gathered values are guaranteed non-None.
            if prediction is None or target is None or time is None:
                raise RuntimeError("gather returned None on data-parallel root")
        else:
            prediction = dict(batch.prediction)
            target = dict(batch.reference)
            time = batch.time
            gathered_diag = (
                dict(batch.step_diagnostics.delta)
                if batch.step_diagnostics is not None
                else None
            )

        for writer in self._writers:
            writer.append_batch(
                target=target,
                prediction=prediction,
                batch_time=time,
            )
        if self._step_diagnostics_writer is not None and gathered_diag is not None:
            diag = StepDiagnostics(delta=gathered_diag)
            self._step_diagnostics_writer.append_batch(diag.to_datasets(time))

    def flush(self):
        """
        Flush the data to disk.
        """
        for writer in self._writers:
            writer.flush()
        if self._step_diagnostics_writer is not None:
            self._step_diagnostics_writer.flush()

    def finalize(self):
        for writer in self._writers:
            writer.finalize()
        if self._step_diagnostics_writer is not None:
            self._step_diagnostics_writer.finalize()


def _write(
    data: BatchData | GatheredBatchData,
    path: str,
    filename: str,
    variable_metadata: Mapping[str, VariableMetadata],
    coords: Mapping[str, np.ndarray],
    dataset_metadata: DatasetMetadata,
):
    """Write provided data to a single netCDF at specified path/filename.

    ``BatchData.to_xarray_dataset`` produces the complete dataset: prognostic
    variables under plain names + time (with the single-timestep time dimension
    squeezed), plus the round-trippable extras (stepper_state, labels, data_mask)
    embedded under reserved ``_fme_state__`` variables when present. This adds the
    writer's presentation on top - per-variable metadata attrs, coordinates, and
    dataset metadata. When ``data`` carries no extras the file is byte-identical
    to the data+time-only file written before this feature.

    Args:
        data: Batch data to written.
        path: Directory to write the netCDF file in.
        filename: filename to use for netCDF.
        variable_metadata: Metadata for each variable to be written to the file.
        coords: Coordinate data to be written to the file.
        dataset_metadata: Metadata for the dataset.
    """
    ds = data.to_xarray_dataset()
    for name in ds.data_vars:
        # Metadata applies only to physical prognostic variables, never to
        # ``time`` or the reserved embedded-state variables.
        if str(name) == "time" or str(name).startswith(_RESERVED_PREFIX):
            continue
        if str(name) in variable_metadata:
            ds[name].attrs.update(variable_metadata[str(name)].as_attrs())
    ds = ds.assign_coords(coords)
    ds.attrs.update(dataset_metadata.as_flat_str_dict())
    to_netcdf_via_inter_filesystem_copy(ds, os.path.join(path, filename))


def _gather_and_write(
    batch: BatchData,
    path: str,
    filename: str,
    variable_metadata: Mapping[str, VariableMetadata],
    coords: Mapping[str, np.ndarray],
    dataset_metadata: DatasetMetadata,
    dist: Distributed | None = None,
) -> None:
    """Gather data-parallel BatchData shards to root and write once.

    Each data-parallel rank holds a contiguous block of samples.  The gather
    concatenates them along the sample dimension in rank order so the written
    file has the same layout as a serial run.  Non-root ranks skip the write.
    A barrier at the end ensures all ranks see the file before continuing
    (required for segmented inference, where the next segment reads the
    restart).  With a single data-parallel rank the gather is a no-op.
    """
    if dist is None:
        dist = Distributed.get_instance()

    gathered_batch = batch.data_parallel_gather(dist)

    try:
        if dist.is_root():
            if gathered_batch is None:
                raise RuntimeError("batch.gather returned None on root")
            _write(
                data=gathered_batch,
                path=path,
                filename=filename,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
    finally:
        dist.barrier()


class DataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        writers: list[Subwriter],
        path: str,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
        step_diagnostics_writer: StepDiagnosticsWriter | None = None,
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
            step_diagnostics_writer: Writer for per-step diagnostics of the
                prediction, or None to not write them. Held separately from
                the sub-writers: it consumes only the prediction's
                diagnostics.
        """
        self._writers = writers
        self._step_diagnostics_writer = step_diagnostics_writer
        self.path = path
        self.variable_metadata = variable_metadata
        self.dataset_metadata = dataset_metadata
        self.coords = coords

    def append_batch(self, batch: PairedData):
        """
        Append prediction data to the file. The prognostic data
        and forcing data are merged before writing.

        Under multi-GPU inference, gathers tensor data and time from all
        data-parallel ranks to root before dispatching to subwriters.
        Non-root ranks return early after participating in the gather
        collectives.

        Args:
            batch: Paired data to be written.
        """
        dist = Distributed.get_instance()
        merged = {**batch.prediction, **batch.forcing}

        if dist.total_data_parallel_ranks > 1:
            gathered_data = _gather_tensor_dict(merged, dist)
            time = _gather_time(batch.time, dist)
            if batch.step_diagnostics is not None:
                gathered_diag = _gather_tensor_dict(
                    dict(batch.step_diagnostics.delta), dist
                )
            else:
                gathered_diag = None
            if not dist.is_data_parallel_root():
                return
            if gathered_data is None or time is None:
                raise RuntimeError("gather returned None on data-parallel root")
        else:
            unpaired_batch = BatchData.new_on_device(
                data=merged,
                time=batch.time,
                labels=batch.labels,
            )
            gathered_data = dict(unpaired_batch.data)
            time = unpaired_batch.time
            gathered_diag = (
                dict(batch.step_diagnostics.delta)
                if batch.step_diagnostics is not None
                else None
            )

        for writer in self._writers:
            writer.append_batch(
                data=gathered_data,
                batch_time=time,
            )
        if self._step_diagnostics_writer is not None and gathered_diag is not None:
            diag = StepDiagnostics(delta=gathered_diag)
            self._step_diagnostics_writer.append_batch(diag.to_datasets(time))

    def flush(self):
        """
        Flush the data to disk.
        """
        for writer in self._writers:
            writer.flush()
        if self._step_diagnostics_writer is not None:
            self._step_diagnostics_writer.flush()

    def finalize(self):
        for writer in self._writers:
            writer.finalize()
        if self._step_diagnostics_writer is not None:
            self._step_diagnostics_writer.finalize()

    def write(self, data: PrognosticState, filename: str):
        """Eagerly write data to a single netCDF file.

        Args:
            data: the data to be written.
            filename: the filename to use for the netCDF file.
        """
        _gather_and_write(
            batch=data.as_batch_data(),
            path=self.path,
            filename=filename,
            variable_metadata=self.variable_metadata,
            coords=self.coords,
            dataset_metadata=self.dataset_metadata,
        )
