import copy
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import xarray as xr

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.histogram import DynamicHistogram


class _HistogramAggregator:
    def __init__(self, n_times: int, names: Sequence[str] | None = None):
        self._histograms: Mapping[str, DynamicHistogram] | None = None
        self._n_times = n_times
        self._names = names

    def record_batch(
        self,
        data: dict[str, torch.Tensor],
        i_time_start: int,
    ):
        if self._histograms is None:
            if self._names is not None:
                save_names_not_in_data = set(self._names) - set(data.keys())
                if len(save_names_not_in_data) > 0:
                    logging.warning(
                        f"Variables {save_names_not_in_data} in save_names are not "
                        "in histogram data."
                    )
                _names = [k for k in data if k in self._names]
            else:
                _names = list(data)
            self._histograms = {
                var_name: DynamicHistogram(n_times=self._n_times) for var_name in _names
            }

        for var_name, histogram in self._histograms.items():
            # go from [n_samples, n_timesteps, n_lat, n_lon] to
            #     [n_timesteps, n_samples, n_lat, n_lon]
            # and then reshape to [n_timesteps, n_hist_samples]
            n_times = data[var_name].shape[1]
            reshaped_data = data[var_name].transpose(1, 0).reshape(n_times, -1)
            histogram.add(reshaped_data, i_time_start=i_time_start)

    def get_dataset(self) -> xr.Dataset:
        if self._histograms is None:
            raise RuntimeError("No data has been recorded.")
        return self._get_single_dataset(self._histograms)

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


class PairedHistogramDataWriter:
    """
    Wrapper over HistogramDataWriter to write both target and prediction data.
    Gives the same interface as HistogramDataWriter.
    """

    def __init__(
        self,
        path: str,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata],
        save_names: Sequence[str] | None,
        dataset_metadata: DatasetMetadata,
    ):
        self._target_writer = HistogramDataWriter(
            path=path,
            n_timesteps=n_timesteps,
            filename="histograms_target.nc",
            variable_metadata=variable_metadata,
            save_names=save_names,
            dataset_metadata=dataset_metadata,
        )
        self._prediction_writer = HistogramDataWriter(
            path=path,
            n_timesteps=n_timesteps,
            filename="histograms_prediction.nc",
            variable_metadata=variable_metadata,
            save_names=save_names,
            dataset_metadata=dataset_metadata,
        )

    def append_batch(
        self,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        self._target_writer.append_batch(
            data=target,
            start_timestep=start_timestep,
            batch_time=batch_time,
        )
        self._prediction_writer.append_batch(
            data=prediction,
            start_timestep=start_timestep,
            batch_time=batch_time,
        )

    def flush(self):
        self._target_writer.flush()
        self._prediction_writer.flush()


class HistogramDataWriter:
    """
    Write [time, bin] histogram data for each variable to a netCDF file.
    """

    def __init__(
        self,
        path: str,
        n_timesteps: int,
        filename: str,
        variable_metadata: Mapping[str, VariableMetadata],
        save_names: Sequence[str] | None,
        dataset_metadata: DatasetMetadata,
    ):
        """
        Args:
            path: The directory within which to write the file.
            n_timesteps: Number of timesteps to write to the file.
            filename: Name of the file to write.
            variable_metadata: Metadata for each variable to be written to the file.
            save_names: Names of variables to save. If None, all variables are saved.
            dataset_metadata: Metadata for the dataset.
        """
        self.path = path
        self._metrics_filename = str(Path(path) / filename)
        self.variable_metadata = variable_metadata
        self._histogram = _HistogramAggregator(n_times=n_timesteps, names=save_names)
        self._dataset_metadata = copy.copy(dataset_metadata)
        self._dataset_metadata.title = (
            f"ACE {filename.removesuffix('.nc').replace('_', ' ')} data file"
        )

    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        """
        Append a batch of data to the file.

        Args:
            data: The data to write.
            start_timestep: Timestep at which to start writing.
            batch_time: Time coordinate for each sample in the batch.
        """
        del batch_time
        self._histogram.record_batch(
            data=data,
            i_time_start=start_timestep,
        )

    def flush(self):
        """
        Flush the data to disk.
        """
        metric_dataset = self._histogram.get_dataset()
        for name in self.variable_metadata:
            try:
                metric_dataset[f"{name}_bin_edges"].attrs["units"] = (
                    self.variable_metadata[name].units
                )
            except KeyError:
                logging.info(
                    f"{name} in metadata but not in data written to "
                    f"{self._metrics_filename}."
                )
        for name in metric_dataset.data_vars:
            if not name.endswith("_bin_edges"):
                metric_dataset[f"{name}_bin_edges"].attrs["long_name"] = (
                    f"{name} bin edges"
                )
                metric_dataset[name].attrs["units"] = "count"
                metric_dataset[name].attrs["long_name"] = f"{name} histogram"
        metric_dataset.attrs.update(self._dataset_metadata.as_flat_str_dict())
        metric_dataset.to_netcdf(self._metrics_filename)
