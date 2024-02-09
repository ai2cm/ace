from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import xarray as xr

from fme.core.data_loading.data_typing import VariableMetadata
from fme.core.histogram import DynamicHistogram


class _HistogramAggregator:
    def __init__(self, n_times: int, names: Optional[Sequence[str]] = None):
        self._prediction_histograms: Optional[Mapping[str, DynamicHistogram]] = None
        self._target_histograms: Optional[Mapping[str, DynamicHistogram]] = None
        self._n_times = n_times
        self._names = names

    def record_batch(
        self,
        target_data: Dict[str, torch.Tensor],
        prediction_data: Dict[str, torch.Tensor],
        i_time_start: int,
    ):
        if self._target_histograms is None:
            if self._names is not None:
                target_names = [k for k in target_data if k in self._names]
            else:
                target_names = list(target_data)
            self._target_histograms = {
                var_name: DynamicHistogram(n_times=self._n_times)
                for var_name in target_names
            }
        if self._prediction_histograms is None:
            if self._names is not None:
                prediction_names = [k for k in prediction_data if k in self._names]
            else:
                prediction_names = list(prediction_data)
            self._prediction_histograms = {
                var_name: DynamicHistogram(n_times=self._n_times)
                for var_name in prediction_names
            }
        for var_name, histogram in self._prediction_histograms.items():
            # go from [n_samples, n_timesteps, n_lat, n_lon] to
            #     [n_timesteps, n_samples, n_lat, n_lon]
            # and then reshape to [n_timesteps, n_hist_samples]
            n_times = prediction_data[var_name].shape[1]
            data = (
                prediction_data[var_name]
                .cpu()
                .numpy()
                .transpose(1, 0, 2, 3)
                .reshape(n_times, -1)
            )
            histogram.add(data, i_time_start=i_time_start)
        for var_name, histogram in self._target_histograms.items():
            # go from [n_samples, n_timesteps, n_lat, n_lon] to
            #     [n_timesteps, n_samples, n_height, n_width]
            # and then reshape to [n_timesteps, n_hist_samples]
            n_times = target_data[var_name].shape[1]
            data = (
                target_data[var_name]
                .cpu()
                .numpy()
                .transpose(1, 0, 2, 3)
                .reshape(n_times, -1)
            )
            histogram.add(data, i_time_start=i_time_start)

    def get_dataset(self) -> xr.Dataset:
        if self._target_histograms is None or self._prediction_histograms is None:
            raise RuntimeError("No data has been recorded.")
        target_dataset = self._get_single_dataset(self._target_histograms)
        prediction_dataset = self._get_single_dataset(self._prediction_histograms)
        for missing_target_name in set(prediction_dataset.data_vars) - set(
            target_dataset.data_vars
        ):
            if not missing_target_name.endswith("_bin_edges"):
                target_dataset[missing_target_name] = xr.DataArray(
                    np.zeros_like(prediction_dataset[missing_target_name]),
                    dims=("time", "bin"),
                )
                target_dataset[f"{missing_target_name}_bin_edges"] = prediction_dataset[
                    f"{missing_target_name}_bin_edges"
                ]
        for missing_prediction_name in set(target_dataset.data_vars) - set(
            prediction_dataset.data_vars
        ):
            if not missing_prediction_name.endswith("_bin_edges"):
                prediction_dataset[missing_prediction_name] = xr.DataArray(
                    np.zeros_like(target_dataset[missing_prediction_name]),
                    dims=("time", "bin"),
                )
                prediction_dataset[
                    f"{missing_prediction_name}_bin_edges"
                ] = target_dataset[f"{missing_prediction_name}_bin_edges"]
        ds = xr.concat([target_dataset, prediction_dataset], dim="source")
        ds["source"] = ["target", "prediction"]
        return ds

    @staticmethod
    def _get_single_dataset(histograms: Mapping[str, DynamicHistogram]) -> xr.Dataset:
        data = {}
        for var_name, histogram in histograms.items():
            data[var_name] = xr.DataArray(
                histogram.counts,
                dims=("time", "bin"),
            )
            data[f"{var_name}_bin_edges"] = xr.DataArray(
                histogram.bin_edges,
                dims=("bin_edges",),
            )
        return xr.Dataset(data)


class HistogramDataWriter:
    """
    Write [time, bin] histogram data for each variable to a netCDF file.
    """

    def __init__(
        self,
        path: str,
        n_timesteps: int,
        metadata: Mapping[str, VariableMetadata],
        save_names: Optional[Sequence[str]],
    ):
        """
        Args:
            path: Path to write netCDF file(s).
            n_timesteps: Number of timesteps to write to the file.
            metadata: Metadata for each variable to be written to the file.
        """
        self.path = path
        self._metrics_filename = str(Path(path) / "histograms.nc")
        self.metadata = metadata
        self._histogram = _HistogramAggregator(n_times=n_timesteps, names=save_names)

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
        del start_sample, batch_times
        self._histogram.record_batch(
            target_data=target,
            prediction_data=prediction,
            i_time_start=start_timestep,
        )

    def flush(self):
        """
        Flush the data to disk.
        """
        metric_dataset = self._histogram.get_dataset()
        for name in self.metadata:
            metric_dataset[f"{name}_bin_edges"].attrs["units"] = self.metadata[
                name
            ].units
        for name in metric_dataset.data_vars:
            if not name.endswith("_bin_edges"):
                metric_dataset[f"{name}_bin_edges"].attrs[
                    "long_name"
                ] = f"{name} bin edges"
                metric_dataset[name].attrs["units"] = "count"
                metric_dataset[name].attrs["long_name"] = f"{name} histogram"
        metric_dataset.to_netcdf(self._metrics_filename)
