import dataclasses
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import xarray as xr

from fme.core.data_loading.data_typing import VariableMetadata

from .histograms import HistogramDataWriter
from .monthly import MonthlyDataWriter, PairedMonthlyDataWriter, months_for_timesteps
from .raw import PairedRawDataWriter, RawDataWriter
from .restart import PairedRestartWriter, RestartWriter
from .time_coarsen import PairedTimeCoarsen, TimeCoarsen, TimeCoarsenConfig
from .video import VideoDataWriter

Subwriter = Union[
    PairedRawDataWriter,
    VideoDataWriter,
    HistogramDataWriter,
    PairedTimeCoarsen,
    PairedMonthlyDataWriter,
    PairedRestartWriter,
]

PredictionOnlySubwriter = Union[
    MonthlyDataWriter, RawDataWriter, RestartWriter, TimeCoarsen
]


@dataclasses.dataclass
class DataWriterConfig:
    """
    Configuration for inference data writers.

    Args:
        log_extended_video_netcdfs: Whether to enable writing of netCDF files
            containing video metrics.
        save_prediction_files: Whether to enable writing of netCDF files
            containing the predictions and target values.
        save_monthly_files: Whether to enable writing of netCDF files
            containing the monthly predictions and target values.
        save_raw_prediction_names: Names of variables to save in the prediction,
            histogram, and monthly netCDF files.
        save_histogram_files: Enable writing of netCDF files containing histograms.
        time_coarsen: Configuration for time coarsening of written outputs.
    """

    log_extended_video_netcdfs: bool = False
    save_prediction_files: bool = True
    save_monthly_files: bool = True
    save_raw_prediction_names: Optional[Sequence[str]] = None
    save_histogram_files: bool = False
    time_coarsen: Optional[TimeCoarsenConfig] = None

    def __post_init__(self):
        if (
            not any(
                [
                    self.save_prediction_files,
                    self.save_monthly_files,
                    self.save_histogram_files,
                ]
            )
            and self.save_raw_prediction_names is not None
        ):
            warnings.warn(
                "save_raw_prediction_names provided but all options to "
                "save subsettable output files are False."
            )

    def build(
        self,
        experiment_dir: str,
        n_samples: int,
        n_timesteps: int,
        prognostic_names: Sequence[str],
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
            enable_monthly_netcdfs=self.save_monthly_files,
            enable_video_netcdfs=self.log_extended_video_netcdfs,
            save_names=self.save_raw_prediction_names,
            prognostic_names=prognostic_names,
            enable_histogram_netcdfs=self.save_histogram_files,
            time_coarsen=self.time_coarsen,
        )

    def build_prediction_only(
        self,
        experiment_dir: str,
        n_samples: int,
        n_timesteps: int,
        prognostic_names: Sequence[str],
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
    ) -> "PredictionOnlyDataWriter":
        if self.save_histogram_files:
            raise NotImplementedError(
                "Saving histograms is not supported for prediction-only data writers. "
                "Make sure to set `save_histogram_files=False`."
            )
        if self.log_extended_video_netcdfs:
            raise NotImplementedError(
                "Saving 'extended video' netCDFs is not supported for prediction-only "
                "data writers. Make sure to set `log_extended_video_netcdfs=False`."
            )
        return PredictionOnlyDataWriter(
            path=experiment_dir,
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            metadata=metadata,
            coords=coords,
            enable_prediction_netcdfs=self.save_prediction_files,
            enable_monthly_netcdfs=self.save_monthly_files,
            save_names=self.save_raw_prediction_names,
            prognostic_names=prognostic_names,
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
        enable_monthly_netcdfs: bool,
        enable_video_netcdfs: bool,
        save_names: Optional[Sequence[str]],
        prognostic_names: Sequence[str],
        enable_histogram_netcdfs: bool,
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
                containing the predictions and target values.
            enable_monthly_netcdfs: Whether to enable writing of netCDF files
                containing the monthly predictions and target values.
            enable_video_netcdfs: Whether to enable writing of netCDF files
                containing video metrics.
            save_names: Names of variables to save in the prediction, histogram,
                and monthly netCDF files.
            enable_histogram_netcdfs: Whether to write netCDFs with histogram data.
            time_coarsen: Configuration for time coarsening of written outputs.
        """
        self._writers: List[Subwriter] = []
        self.path = path
        self.coords = coords
        self.metadata = metadata
        self.prognostic_names = prognostic_names

        if time_coarsen is not None:
            n_coarsened_timesteps = time_coarsen.n_coarsened_timesteps(n_timesteps)
        else:
            n_coarsened_timesteps = n_timesteps

        def _time_coarsen_builder(data_writer: Subwriter) -> Subwriter:
            if time_coarsen is not None:
                return time_coarsen.build(data_writer)
            return data_writer

        if enable_prediction_netcdfs:
            self._writers.append(
                _time_coarsen_builder(
                    PairedRawDataWriter(
                        path=path,
                        n_samples=n_samples,
                        save_names=save_names,
                        metadata=metadata,
                        coords=coords,
                    )
                )
            )
        if enable_monthly_netcdfs:
            self._writers.append(
                PairedMonthlyDataWriter(
                    path=path,
                    n_samples=n_samples,
                    n_timesteps=n_timesteps,
                    save_names=save_names,
                    metadata=metadata,
                    coords=coords,
                )
            )
        if enable_video_netcdfs:
            self._writers.append(
                _time_coarsen_builder(
                    VideoDataWriter(
                        path=path,
                        n_timesteps=n_coarsened_timesteps,
                        metadata=metadata,
                        coords=coords,
                    )
                )
            )
        if enable_histogram_netcdfs:
            self._writers.append(
                _time_coarsen_builder(
                    HistogramDataWriter(
                        path=path,
                        n_timesteps=n_coarsened_timesteps,
                        metadata=metadata,
                        save_names=save_names,
                    )
                )
            )
        self._writers.append(
            PairedRestartWriter(
                path=path,
                is_restart_step=lambda i: i == n_timesteps - 1,
                prognostic_names=prognostic_names,
                metadata=metadata,
                coords=coords,
            )
        )

    def save_initial_condition(
        self,
        ic_data: Dict[str, torch.Tensor],
        ic_time: xr.DataArray,
    ):
        data_arrays = {}
        for name in self.prognostic_names:
            if name not in ic_data:
                raise KeyError(
                    f"Initial condition data missing for prognostic variable {name}."
                )
            data = ic_data[name].cpu().numpy()
            data_arrays[name] = xr.DataArray(data, dims=["sample", "lat", "lon"])
            if name in self.metadata:
                data_arrays[name].attrs = {
                    "long_name": self.metadata[name].long_name,
                    "units": self.metadata[name].units,
                }
        data_arrays["time"] = ic_time
        ds = xr.Dataset(data_arrays, coords=self.coords)
        ds.to_netcdf(str(Path(self.path) / "initial_condition.nc"))

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
            batch_times: Time coordinates for each sample in the batch.
        """
        for writer in self._writers:
            writer.append_batch(
                target=target,
                prediction=prediction,
                start_timestep=start_timestep,
                batch_times=batch_times,
            )

    def flush(self):
        """
        Flush the data to disk.
        """
        for writer in self._writers:
            writer.flush()


class PredictionOnlyDataWriter:
    def __init__(
        self,
        path: str,
        n_samples: int,
        n_timesteps: int,
        metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        enable_prediction_netcdfs: bool,
        enable_monthly_netcdfs: bool,
        save_names: Optional[Sequence[str]],
        prognostic_names: Sequence[str],
        time_coarsen: Optional[TimeCoarsenConfig] = None,
    ):
        """
        Args:
            path: Directory within which to write netCDF file(s).
            n_samples: Number of samples to write to the file.
            n_timesteps: Number of timesteps to write to the file.
            metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            enable_prediction_netcdfs: Whether to enable writing of netCDF files
                containing the predictions and target values.
            enable_monthly_netcdfs: Whether to enable writing of netCDF files
            save_names: Names of variables to save in the prediction, histogram,
                and monthly netCDF files.
            time_coarsen: Configuration for time coarsening of raw outputs.
        """
        self._writers: List[PredictionOnlySubwriter] = []

        def _time_coarsen_builder(
            data_writer: PredictionOnlySubwriter,
        ) -> PredictionOnlySubwriter:
            if time_coarsen is not None:
                return time_coarsen.build_prediction_only(data_writer)
            return data_writer

        if enable_prediction_netcdfs:
            self._writers.append(
                _time_coarsen_builder(
                    RawDataWriter(
                        path=path,
                        label="autoregressive_predictions.nc",
                        n_samples=n_samples,
                        save_names=save_names,
                        metadata=metadata,
                        coords=coords,
                    )
                )
            )

        if enable_monthly_netcdfs:
            self._writers.append(
                MonthlyDataWriter(
                    path=path,
                    label="predictions",
                    n_samples=n_samples,
                    n_months=months_for_timesteps(n_timesteps),
                    save_names=save_names,
                    metadata=metadata,
                    coords=coords,
                )
            )

        self._writers.append(
            RestartWriter(
                path=path,
                is_restart_step=lambda i: i == n_timesteps - 1,
                prognostic_names=prognostic_names,
                metadata=metadata,
                coords=coords,
            )
        )

    def append_batch(
        self,
        data: Dict[str, torch.Tensor],
        start_timestep: int,
        batch_times: xr.DataArray,
    ):
        """
        Append a batch of data to the file.
        Args:
            data: Data to write.
            start_timestep: Timestep at which to start writing.
            start_sample: Sample at which to start writing.
            batch_times: Time coordinates for each sample in the batch.
        """
        for writer in self._writers:
            writer.append_batch(data, start_timestep, batch_times)

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
        batch_times: xr.DataArray,
    ):
        pass

    def save_initial_condition(
        self,
        ic_data: Dict[str, torch.Tensor],
        ic_time: xr.DataArray,
    ):
        pass
