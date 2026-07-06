from collections.abc import Mapping, Sequence

import cftime
import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.raw import RawDataWriter
from fme.ace.inference.data_writer.time_coarsen import coarsen_batch
from fme.core.dataset.data_typing import VariableMetadata

VALID_TIME = "valid_time"


class StepDiagnosticsWriter:
    """
    Write per-step diagnostics (prediction-only, no target series) to
    ``autoregressive_step_diagnostics.nc``.

    Consumes datasets exported by the prediction's step diagnostics container;
    it has no knowledge of the container's internals. Values are written as-is,
    already in physical units.
    """

    def __init__(
        self,
        path: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        save_names: Sequence[str] | None,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
        coarsen_factor: int = 1,
    ):
        """
        Args:
            path: Directory within which to write the file.
            initial_condition_times: 1D array of initial condition times
                (start time for each inference run).
            save_names: Names of variables to save in the output file.
                If None, all provided variables will be saved.
            variable_metadata: Metadata for each variable to be written to the
                file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
            coarsen_factor: Factor by which to block-mean the data in time
                before writing.
        """
        self._writer = RawDataWriter(
            path=path,
            label="autoregressive_step_diagnostics",
            initial_condition_times=initial_condition_times,
            save_names=save_names,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
        )
        self._coarsen_factor = coarsen_factor

    def append_batch(self, dataset: xr.Dataset):
        """
        Append a batch of step diagnostics to the file.

        Args:
            dataset: Diagnostics variables with dims ``(sample, time, ...)``
                and a ``valid_time`` coordinate of dims ``(sample, time)``.
        """
        if len(dataset.data_vars) == 0:
            return
        data = {
            str(name): torch.as_tensor(variable.values)
            for name, variable in dataset.data_vars.items()
        }
        batch_time = xr.DataArray(dataset[VALID_TIME].values, dims=("sample", "time"))
        if self._coarsen_factor > 1:
            data, batch_time = coarsen_batch(data, batch_time, self._coarsen_factor)
        self._writer.append_batch(data=data, batch_time=batch_time)

    def flush(self):
        self._writer.flush()

    def finalize(self):
        self._writer.finalize()
