from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import xarray as xr

from fme.core.aggregator.inference.video import VideoAggregator
from fme.core.data_loading.data_typing import VariableMetadata


class VideoDataWriter:
    """
    Write [time, lat, lon] metric data to a netCDF file.
    """

    def __init__(
        self,
        path: str,
        n_timesteps: int,
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
    ):
        """
        Args:
            filename: Path to write netCDF file(s).
            n_samples: Number of samples to write to the file.
            n_timesteps: Number of timesteps to write to the file.
            metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
        """
        self.path = path
        self._metrics_filename = str(
            Path(path) / "reduced_autoregressive_predictions.nc"
        )
        self.metadata = metadata
        self.coords = coords
        self._video = VideoAggregator(
            n_timesteps=n_timesteps, enable_extended_videos=True, metadata=metadata
        )

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        batch_times: xr.DataArray,
    ):
        """
        Append a batch of data to the file.

        Args:
            target: Target data.
            prediction: Prediction data.
            start_timestep: Timestep at which to start writing.
            batch_times: Time coordinates for each sample in the batch. Unused.
        """
        self._video.record_batch(
            loss=np.nan,
            target_data=target,
            gen_data=prediction,
            i_time_start=start_timestep,
        )

    def flush(self):
        """
        Flush the data to disk.
        """
        metric_dataset = self._video.get_dataset()
        coords = {}
        if "lat" in self.coords:
            coords["lat"] = self.coords["lat"]
        if "lon" in self.coords:
            coords["lon"] = self.coords["lon"]
        metric_dataset = metric_dataset.assign_coords(coords)
        metric_dataset.to_netcdf(self._metrics_filename)
