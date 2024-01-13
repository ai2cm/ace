import dataclasses
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import xarray as xr

from fme.core.data_loading.data_typing import VariableMetadata

from .histograms import HistogramDataWriter
from .prediction import PredictionDataWriter
from .time_coarsen import TimeCoarsen, TimeCoarsenConfig
from .video import VideoDataWriter

Subwriter = Union[
    PredictionDataWriter, VideoDataWriter, HistogramDataWriter, TimeCoarsen
]


@dataclasses.dataclass
class DataWriterConfig:
    """
    Configuration for inference data writers.

    Args:
        log_extended_video_netcdfs: Whether to enable writing of netCDF files
            containing video metrics.
        save_prediction_files: Whether to enable writing of netCDF files
            containing the predictions.
        save_raw_prediction_names: Names of variables to save in the predictions
            netcdf file.
        time_coarsen: Configuration for time coarsening of written outputs.
    """

    log_extended_video_netcdfs: bool = False
    save_prediction_files: bool = True
    save_raw_prediction_names: Optional[Sequence[str]] = None
    time_coarsen: Optional[TimeCoarsenConfig] = None

    def __post_init__(self):
        if (
            not self.save_prediction_files
            and self.save_raw_prediction_names is not None
        ):
            raise ValueError(
                "save_raw_prediction_names provided but save_prediction_files is False"
            )

    def build(
        self,
        experiment_dir: str,
        n_samples: int,
        n_timesteps: int,
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
    ) -> "DataWriter":
        return DataWriter(
            path=experiment_dir,
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            metadata=metadata,
            coords=coords,
            enable_prediction_netcdfs=self.save_prediction_files,
            save_names=self.save_raw_prediction_names,
            enable_video_netcdfs=self.log_extended_video_netcdfs,
            time_coarsen=self.time_coarsen,
        )


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
        save_names: Optional[Sequence[str]],
        time_coarsen: Optional[TimeCoarsenConfig] = None,
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
            save_names: Names of variables to save in the predictions netcdf file.
            time_coarsen: Configuration for time coarsening of written outputs.
        """
        self._writers: List[Subwriter] = []

        if time_coarsen is not None:
            n_timesteps = time_coarsen.n_coarsened_timesteps(n_timesteps)

        def _time_coarsen_builder(data_writer: Subwriter) -> Subwriter:
            if time_coarsen is not None:
                return time_coarsen.build(data_writer)
            else:
                return data_writer

        if enable_prediction_netcdfs:
            self._writers.append(
                _time_coarsen_builder(
                    PredictionDataWriter(
                        path=path,
                        n_samples=n_samples,
                        save_names=save_names,
                        metadata=metadata,
                        coords=coords,
                    )
                )
            )
        if enable_video_netcdfs:
            self._writers.append(
                _time_coarsen_builder(
                    VideoDataWriter(
                        path=path,
                        n_timesteps=n_timesteps,
                        metadata=metadata,
                        coords=coords,
                    )
                )
            )
        self._writers.append(
            _time_coarsen_builder(
                HistogramDataWriter(
                    path=path,
                    n_timesteps=n_timesteps,
                    metadata=metadata,
                )
            )
        )

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
            target: Target data.
            prediction: Prediction data.
            start_timestep: Timestep at which to start writing.
            start_sample: Sample at which to start writing.
            batch_times: Time coordinates for each sample in the batch.
        """
        for writer in self._writers:
            writer.append_batch(
                target=target,
                prediction=prediction,
                start_timestep=start_timestep,
                start_sample=start_sample,
                batch_times=batch_times,
            )

    def flush(self):
        """
        Flush the data to disk.
        """
        for writer in self._writers:
            writer.flush()


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
        batch_times: xr.DataArray,
    ):
        pass
