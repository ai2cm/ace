import torch
import xarray as xr

from fme.ace.inference.data_writer.raw import RawDataWriter
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsen

VALID_TIME = "valid_time"

STEP_DIAGNOSTICS_LABEL = "autoregressive_step_diagnostics"


class StepDiagnosticsWriter:
    """
    Write per-step diagnostics (prediction-only, no target series) to
    ``autoregressive_step_diagnostics.nc``.

    Consumes datasets exported by the prediction's step diagnostics container;
    it has no knowledge of the container's internals. Values are written as-is,
    already in physical units.
    """

    def __init__(self, writer: RawDataWriter | TimeCoarsen):
        """
        Args:
            writer: The writer to append diagnostics to, labeled with
                ``STEP_DIAGNOSTICS_LABEL`` and optionally wrapped for time
                coarsening.
        """
        self._writer = writer

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
        self._writer.append_batch(data=data, batch_time=batch_time)

    def flush(self):
        self._writer.flush()

    def finalize(self):
        self._writer.finalize()
