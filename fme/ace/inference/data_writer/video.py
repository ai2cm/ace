import copy
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from fme.ace.aggregator.inference.video import VideoAggregator
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.core.dataset.data_typing import VariableMetadata


class PairedVideoDataWriter:
    """
    Write [time, lat, lon] metric data to a netCDF file.
    """

    def __init__(
        self,
        path: str,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
        label: str = "reduced_autoregressive_predictions",
    ):
        """
        Args:
            path: Directory within which to write the file.
            n_samples: Number of samples to write to the file.
            n_timesteps: Number of timesteps to write to the file.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
            label: Label for the filename.
        """
        self.path = path
        self._metrics_filename = str(Path(path) / f"{label}.nc")
        self.variable_metadata = variable_metadata
        self.coords = coords
        self._video = VideoAggregator(
            n_timesteps=n_timesteps,
            enable_extended_videos=True,
            variable_metadata=variable_metadata,
        )
        self._dataset_metadata = copy.copy(dataset_metadata)
        self._dataset_metadata.title = f"ACE {label.replace('_', ' ')} data file"

    def append_batch(
        self,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        """
        Append a batch of data to the file.

        Args:
            target: Target data.
            prediction: Prediction data.
            start_timestep: Timestep at which to start writing.
            batch_time: Time coordinate for each sample in the batch. Unused.
        """
        self._video.record_batch(
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
        metric_dataset.attrs.update(self._dataset_metadata.as_flat_str_dict())
        metric_dataset.to_netcdf(self._metrics_filename)
