from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
import torch
from netCDF4 import Dataset

from fme.core.aggregator.inference.video import VideoAggregator
from fme.core.data_loading.typing import VariableMetadata


class DataWriter:
    def __init__(
        self,
        path: str,
        n_samples: int,
        n_timesteps: int,
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        enable_prediction_netcdfs: bool,
        enable_video_netcdfs: bool,
    ):
        """
        Args:
            path: Path to write netCDF file(s).
            n_samples: Number of samples to write to the file.
            n_timesteps: Number of timesteps to write to the file.
            metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            enable_prediction_netcdfs: Whether to enable writing of netCDF files
                containing the predictions.
            enable_video_netcdfs: Whether to enable writing of netCDF files
                containing video metrics.
        """
        self._writers: List[Union[_PredictionDataWriter, _VideoDataWriter]] = []
        if enable_prediction_netcdfs:
            self._writers.append(
                _PredictionDataWriter(
                    path=path,
                    n_samples=n_samples,
                    metadata=metadata,
                    coords=coords,
                )
            )
        if enable_video_netcdfs:
            self._writers.append(
                _VideoDataWriter(
                    path=path,
                    n_timesteps=n_timesteps,
                    metadata=metadata,
                    coords=coords,
                )
            )

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        start_sample: int,
    ):
        """
        Append a batch of data to the file.

        Args:
            target: Target data.
            prediction: Prediction data.
            start_timestep: Timestep at which to start writing.
            start_sample: Sample at which to start writing.
        """
        for writer in self._writers:
            writer.append_batch(
                target=target,
                prediction=prediction,
                start_timestep=start_timestep,
                start_sample=start_sample,
            )

    def flush(self):
        """
        Flush the data to disk.
        """
        for writer in self._writers:
            writer.flush()


class _PredictionDataWriter:
    """
    Write raw prediction data to a netCDF file.
    """

    def __init__(
        self,
        path: str,
        n_samples: int,
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
        filename = str(Path(path) / "autoregressive_predictions.nc")
        self.metadata = metadata
        self.coords = coords
        self.dataset = Dataset(filename, "w", format="NETCDF4")
        self.dataset.createDimension("source", 2)
        self.dataset.createDimension("timestep", None)  # unlimited dimension
        self.dataset.createDimension("sample", n_samples)
        self.dataset.createVariable("source", "str", ("source",))
        self.dataset.variables["source"][:] = np.array(["target", "prediction"])
        self._n_lat: Optional[int] = None
        self._n_lon: Optional[int] = None

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        start_sample: int,
    ):
        """
        Append a batch of data to the file.

        Args:
            target: Target data.
            prediction: Prediction data.
            start_timestep: Timestep at which to start writing.
            start_sample: Sample at which to start writing.
        """
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

        dims = ("source", "sample", "timestep", "lat", "lon")
        for variable_name in set(target.keys()).union(prediction.keys()):
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

        self.dataset.sync()  # Flush the data to disk

    def flush(self):
        """
        Flush the data to disk.
        """
        self.dataset.sync()


class _VideoDataWriter:
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
            n_timesteps=n_timesteps, enable_extended_videos=True
        )

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        start_sample: int,
    ):
        """
        Append a batch of data to the file.

        Args:
            target: Target data.
            prediction: Prediction data.
            start_timestep: Timestep at which to start writing.
            start_sample: Sample at which to start writing.
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


class NullDataWriter:
    """
    Null pattern for DataWriter, which does nothing.
    """

    def __init__(self):
        pass

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        start_sample: int,
    ):
        pass
