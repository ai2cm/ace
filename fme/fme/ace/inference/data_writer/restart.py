import tempfile
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import xarray as xr

from fme.core.data_loading.data_typing import VariableMetadata


class RestartWriter:
    """
    Write raw prediction data to a netCDF file.
    """

    def __init__(
        self,
        path: str,
        is_restart_step: Callable[[int], bool],
        prognostic_names: Sequence[str],
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
    ):
        """
        Args:
            path: Path to the directory in which to save the file.
            is_restart_step: Function that returns True if the given timestep is a
                restart step.
            prognostic_names: Names of prognostic variables to write to the file.
            metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
        """
        self.restart_filename = str(Path(path) / "restart.nc")
        self._save_names = prognostic_names
        self.metadata = metadata
        self.coords = coords
        self.is_restart_step = is_restart_step

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
            target: Target data. Ignored for this data writer.
            prediction: Prediction data.
            start_timestep: Timestep (lead time dim) at which to start writing.
            start_sample: Sample (initialization time dim) at which to start writing.
            batch_times: Time coordinates for each sample in the batch.
        """
        n_times = prediction[list(prediction.keys())[0]].shape[1]
        restart_step = get_last_restart_step(
            start_timestep, n_times, self.is_restart_step
        )
        if restart_step is not None:
            window_step = restart_step - start_timestep
            self._write_restart_data(
                prediction, window_step, restart_step, batch_times[:, window_step]
            )

    def _write_restart_data(
        self,
        prediction: Dict[str, torch.Tensor],
        i_window: int,
        i_restart: int,
        restart_time: xr.DataArray,
    ):
        data_vars = {}
        for name in self._save_names:
            prediction_data = prediction[name][:, i_window, :, :].cpu().numpy()
            data_vars[name] = xr.DataArray(
                prediction_data, dims=["sample", "lat", "lon"]
            )
            if name in self.metadata:
                data_vars[name].attrs = {
                    "long_name": self.metadata[name].long_name,
                    "units": self.metadata[name].units,
                }
        data_vars["time"] = restart_time
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=self.coords,
            attrs={"timestep": i_restart},
        )
        # first write to a temporary file so we have the old restart even if
        # the job ends or errors while writing the new one
        try:
            tmpfile = tempfile.NamedTemporaryFile()
            ds.to_netcdf(tmpfile.name)
            # delete the old restart file if it exists, and replace it
            Path(self.restart_filename).unlink(missing_ok=True)
            Path(tmpfile.name).rename(self.restart_filename)
        finally:
            if Path(tmpfile.name).exists():
                tmpfile.close()

    def flush(self):
        pass


def get_last_restart_step(
    start_step: int, n_steps: int, is_restart_step: Callable[[int], bool]
) -> Optional[int]:
    """
    Get the last restart step in a sequence of steps.

    Args:
        start_step: The first step in the sequence.
        n_steps: The number of steps in the sequence.
        is_restart_step: A function that returns True if a given step
            is a restart step.

    Returns:
        The last restart step in the sequence, or None if there is no restart step.
    """
    timesteps = list(range(n_steps))
    timesteps.reverse()  # we want the last step, so start at the end
    for i_time in timesteps:
        if is_restart_step(start_step + i_time):
            return start_step + i_time
    return None
