from collections.abc import Callable, Mapping

import torch
import xarray as xr

from fme.ace.inference.data_writer.raw import RawDataWriter
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsen

VALID_TIME = "valid_time"

STEP_DIAGNOSTICS_DIR = "step_diagnostics"


class StepDiagnosticsWriter:
    """
    Write per-step diagnostics (prediction-only, no target series) to one
    netCDF file per named dataset, ``step_diagnostics/{name}.nc``.

    Consumes named datasets exported by the prediction's step diagnostics
    container; it has no knowledge of the container's internals. Values are
    written as-is, already in physical units.
    """

    def __init__(self, writer_factory: Callable[[str], RawDataWriter | TimeCoarsen]):
        """
        Args:
            writer_factory: Factory building the sub-writer for a named dataset
                (optionally wrapped for time coarsening), called once per name
                on its first non-empty append.
        """
        self._writer_factory = writer_factory
        self._writers: dict[str, RawDataWriter | TimeCoarsen] = {}

    def append_batch(self, datasets: Mapping[str, xr.Dataset]):
        """
        Append a batch of step diagnostics to the per-name files.

        Args:
            datasets: Mapping of dataset name to diagnostics variables with
                dims ``(sample, time, ...)`` and a ``valid_time`` coordinate
                of dims ``(sample, time)``. Empty datasets are skipped.
        """
        for name, dataset in datasets.items():
            if len(dataset.data_vars) == 0:
                continue
            if name not in self._writers:
                self._writers[name] = self._writer_factory(name)
            data = {
                str(var_name): torch.as_tensor(variable.values)
                for var_name, variable in dataset.data_vars.items()
            }
            batch_time = xr.DataArray(
                dataset[VALID_TIME].values, dims=("sample", "time")
            )
            self._writers[name].append_batch(data=data, batch_time=batch_time)

    def flush(self):
        for writer in self._writers.values():
            writer.flush()

    def finalize(self):
        for writer in self._writers.values():
            writer.finalize()
