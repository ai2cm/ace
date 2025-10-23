import abc
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

import cftime
import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict, TensorMapping

from ...plotting import plot_mean_and_samples

SAMPLE_DIM, TIME_DIM, LAT_DIM, LON_DIM = 0, 1, -2, -1

SEA_SURFACE_TEMPERATURE_NAMES = ["sst", "surface_temperature", "TS"]


class Region(abc.ABC):
    @property
    @abc.abstractmethod
    def regional_weights(self) -> torch.Tensor: ...


@dataclass
class LatLonRegion(Region):
    lat: torch.Tensor
    lon: torch.Tensor
    lat_bounds: tuple[float, float]
    lon_bounds: tuple[float, float]
    horizontal_dims: ClassVar[tuple[int, int]] = (LAT_DIM, LON_DIM)

    def __post_init__(self):
        lat_mask = (
            (self.lat >= self.lat_bounds[0]) & (self.lat <= self.lat_bounds[1])
        ).unsqueeze(self.horizontal_dims[1])
        lon_mask = (
            (self.lon >= self.lon_bounds[0]) & (self.lon <= self.lon_bounds[1])
        ).unsqueeze(self.horizontal_dims[0])
        self._regional_weights = torch.where(
            torch.logical_and(lat_mask, lon_mask), 1.0, 0.0
        )

        dist = Distributed.get_instance()
        if dist.is_spatial_distributed():
          # CHECK:
          crop_shape = self._regional_weights.shape
          local_shape_h, local_offset_h, local_shape_w, local_offset_w = dist.get_local_shape_and_offset(crop_shape)
          self._regional_weights = self._regional_weights[local_offset_h : local_offset_h + local_shape_h, local_offset_w : local_offset_w + local_shape_w]

    @property
    def regional_weights(self) -> torch.Tensor:
        return self._regional_weights


def compute_power_spectrum(
    data: xr.DataArray,
    fs=1,
    time_dim: int = TIME_DIM,
    sample_dim: int = SAMPLE_DIM,
):
    """
    Compute the power spectrum of the input data.

    Args:
        data: A tensor of size n_time_steps containing the data.
        fs: The sampling frequency, defaults to 1 sample per month.
        time_dim: Time dimension index
        sample_dim: Sample dimension index
    """
    if len(data.shape) == 2:
        uhat = np.fft.rfft(data, axis=time_dim)
        power = np.abs(uhat) ** 2
        power = power.mean(axis=sample_dim)
    else:
        raise ValueError(
            "Expected indicies to be of shape (sample, time) when calculating "
            f"power spectrum, got {data.shape}"
        )
    n_samples = data.shape[time_dim]
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    freqs_per_year = freqs / fs * 12.0
    return freqs_per_year, power


def _compute_sample_mean_std(
    data: xr.DataArray,
    target_data: xr.DataArray | None = None,
    time_dim: int = TIME_DIM,
) -> float:
    """Compute the standard deviation of each sample in the data and return the
    average standard deviation across all samples.

    Args:
        data: The generated nino34 index data with sample and a time dims.
        target_data: (Optional) The target nino34 index data of the same shape. If
            provided, the generated standard deviations are normalized by the target
            standard deviation prior to computing the mean across samples.
        time_dim: Time dimension index
    """
    std_by_sample = np.nanstd(data, axis=time_dim)
    if target_data is not None:
        std_by_sample = std_by_sample / np.nanstd(target_data, axis=time_dim)
    return std_by_sample.mean().item()


class RegionalIndexAggregator:
    """Aggregator for computing a regional index, in this case a monthly- and area-
    weighted average of a variable over a region.

    Args:
        regional_weights: A tensor of weights for the region.
        regional_mean: A function that computes the regional mean of a variable,
            given weights.
        running_mean_n_months: The number of months to use for the running mean.
    """

    _sample_dim = SAMPLE_DIM
    _time_dim = TIME_DIM
    sea_surface_temperature_names = SEA_SURFACE_TEMPERATURE_NAMES

    def __init__(
        self,
        regional_weights: torch.Tensor,
        regional_mean: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        running_mean_n_months: int = 5,
    ):
        self._regional_weights = regional_weights
        self._regional_mean = regional_mean
        self._running_mean_n_months = running_mean_n_months
        self._raw_indices: TensorDict = {}
        self._raw_index_times: xr.DataArray | None = None
        self._calendar: str | None = None
        self._already_logged: list[str] = []

    def record_batch(self, time: xr.DataArray, data: TensorMapping) -> None:
        for sst_name in self.sea_surface_temperature_names:
            if sst_name not in data:
                if sst_name not in self._already_logged:
                    logging.info(
                        f"Variable {sst_name} not found in data. "
                        "Skipping Nino3.4 index computation for this variable."
                    )
                    self._already_logged.append(sst_name)
                continue
            regional_average = self._regional_mean(
                data[sst_name], self._regional_weights
            )
            if sst_name not in self._raw_indices:
                self._raw_indices[sst_name] = regional_average
            else:
                self._raw_indices[sst_name] = torch.cat(
                    [self._raw_indices[sst_name], regional_average],
                    dim=self._time_dim,
                )
        if self._raw_index_times is None:
            self._raw_index_times = time
        else:
            self._raw_index_times = xr.concat([self._raw_index_times, time], dim="time")
        if self._calendar is None and time.dt.calendar is not None:
            self._calendar = time.dt.calendar

    def get_indices(self) -> xr.Dataset:
        indices = {}
        for sst_name in self.sea_surface_temperature_names:
            if sst_name not in self._raw_indices:
                continue
            anomaly_index_values = anomalies_from_monthly_climo(
                self._raw_indices[sst_name], self._raw_index_times
            )
            time_averaged_index, unique_months = running_monthly_mean(
                anomaly_index_values,
                self._raw_index_times,
                n_months=self._running_mean_n_months,
            )
            gathered_index = self._get_gathered_index(
                time_averaged_index, unique_months.years, unique_months.months
            )
            if gathered_index is None:
                continue
            indices[sst_name] = gathered_index
        return xr.Dataset(indices)

    def _get_gathered_index(
        self, index: torch.Tensor, years: torch.Tensor, months: torch.Tensor
    ) -> xr.DataArray | None:
        dist = Distributed.get_instance()
        if dist.world_size > 1:
            gathered_index, gathered_years, gathered_months = (
                dist.gather_irregular(index),
                dist.gather_irregular(years),
                dist.gather_irregular(months),
            )
            if (
                gathered_index is None
                or gathered_years is None
                or gathered_months is None
            ):
                return None
        else:
            gathered_index, gathered_years, gathered_months = [index], [years], [months]
        return self._to_index_data_array(
            gathered_index, gathered_years, gathered_months
        )

    def _to_index_data_array(
        self,
        indices: list[torch.Tensor],
        years: list[torch.Tensor],
        months: list[torch.Tensor],
    ) -> xr.DataArray:
        index_data_arrays = []
        for index, year, month in zip(indices, years, months):
            time_coord = [  # approximate the middle of the month
                cftime.datetime(
                    single_year.item(),
                    single_month.item(),
                    15,
                    calendar=(
                        self._calendar if self._calendar is not None else "standard"
                    ),
                )
                for single_year, single_month in zip(year.cpu(), month.cpu())
            ]
            index_data_arrays.append(
                xr.DataArray(
                    data=index.cpu().numpy(),
                    dims=["sample", "time"],
                    coords={"time": time_coord},
                )
            )
        return xr.concat(index_data_arrays, dim="sample")

    def get_logs(self, label: str) -> dict[str, Any]:
        indices = self.get_indices()
        logs = {}
        for sst_name in self.sea_surface_temperature_names:
            if sst_name in indices and indices[sst_name].sizes["time"] > 1:
                fig, ax = plt.subplots(1, 1)
                index_plottable = indices[sst_name].assign_coords(
                    {"time": convert_cftime_to_datetime_coord(indices[sst_name].time)}
                )
                plot_mean_and_samples(
                    ax, index_plottable, "ensemble_mean", time_series_dim="time"
                )
                ax.set_title("Nino3.4 Index")
                ax.set_ylabel("K")
                ax.legend()
                fig.tight_layout()
                logs[f"{sst_name}_nino34_index"] = fig
                logs[f"{sst_name}_nino34_index_std"] = _compute_sample_mean_std(
                    indices[sst_name]
                )
        for sst_name in self.sea_surface_temperature_names:
            if (
                sst_name in indices
                and indices[sst_name].dropna("time").sizes["time"] > 1
            ):
                freq, power_spectrum = _calculate_sample_average_power_spectrum(
                    indices[sst_name]
                )
                fig, ax = plt.subplots(1, 1)
                ax.plot(freq, power_spectrum, label="predicted ensemble mean")
                ax.set_title("Power Spectrum of Nino3.4 Index")
                ax.set_xlabel("Frequency [cycles/year]")
                ax.set_ylabel("Power [K**2]")
                ax.set(yscale="log")
                ax.legend()
                fig.tight_layout()
                logs[f"{sst_name}_nino34_index_power_spectrum"] = fig

        if len(label) > 0:
            label = label + "/"
        return {f"{label}{k}": v for k, v in logs.items()}

    def get_dataset(self) -> xr.Dataset:
        indices = self.get_indices()
        if indices is None:
            return xr.Dataset()
        else:
            return indices


class PairedRegionalIndexAggregator:
    """A paired (target, prediction) RegionalIndexAggregator."""

    def __init__(
        self,
        target_aggregator: RegionalIndexAggregator,
        prediction_aggregator: RegionalIndexAggregator,
    ):
        self._prediction_aggregator = prediction_aggregator
        self._target_aggregator = target_aggregator

    def record_batch(
        self,
        time: xr.DataArray,
        target_data: TensorMapping,
        gen_data: TensorMapping,
    ) -> None:
        self._target_aggregator.record_batch(time=time, data=target_data)
        self._prediction_aggregator.record_batch(time=time, data=gen_data)

    def get_logs(self, label: str) -> dict[str, Any]:
        target_indices = self._target_aggregator.get_indices()
        prediction_indices = self._prediction_aggregator.get_indices()
        logs = {}
        for sst_name in self._prediction_aggregator.sea_surface_temperature_names:
            if (
                sst_name in prediction_indices
                and prediction_indices[sst_name].sizes["time"] > 1
            ):
                target_indices_plottable = target_indices.assign_coords(
                    {"time": convert_cftime_to_datetime_coord(target_indices.time)}
                )
                prediction_indices_plottable = prediction_indices.assign_coords(
                    {"time": convert_cftime_to_datetime_coord(prediction_indices.time)}
                )

                fig, ax = plt.subplots(1, 1)
                plot_mean_and_samples(
                    ax,
                    prediction_indices_plottable[sst_name],
                    "predicted ensemble mean",
                    time_series_dim="time",
                )
                plot_mean_and_samples(
                    ax,
                    target_indices_plottable[sst_name],
                    "target",
                    time_series_dim="time",
                    color="orange",
                    plot_samples=False,
                )
                ax.set_title("Nino3.4 Index")
                ax.set_ylabel("K")
                ax.legend()
                fig.tight_layout()
                logs[f"{sst_name}_nino34_index"] = fig
                logs[f"{sst_name}_nino34_index_std"] = _compute_sample_mean_std(
                    prediction_indices[sst_name]
                )
                logs[f"{sst_name}_nino34_index_std_norm"] = _compute_sample_mean_std(
                    prediction_indices[sst_name],
                    target_indices[sst_name],
                )
        for sst_name in self._prediction_aggregator.sea_surface_temperature_names:
            if (
                sst_name in prediction_indices
                and prediction_indices[sst_name].notnull().any().item()
            ):
                pred_freq, prediction_power_spectrum = (
                    _calculate_sample_average_power_spectrum(
                        prediction_indices[sst_name]
                    )
                )
                target_freq, target_power_spectrum = (
                    _calculate_sample_average_power_spectrum(target_indices[sst_name])
                )
                fig, ax = plt.subplots(1, 1)
                ax.plot(
                    pred_freq,
                    prediction_power_spectrum,
                    label="predicted ensemble mean",
                )
                ax.plot(
                    target_freq, target_power_spectrum, label="target", color="orange"
                )
                ax.set_title("Power Spectrum of Nino3.4 Index")
                ax.set_xlabel("Frequency [cycles/year]")
                ax.set(yscale="log")
                ax.legend()
                fig.tight_layout()
                logs[f"{sst_name}_nino34_index_power_spectrum"] = fig
        if len(label) > 0:
            label = label + "/"

        return {f"{label}{k}": v for k, v in logs.items()}

    def get_dataset(self) -> xr.Dataset:
        prediction = self._prediction_aggregator.get_dataset()
        target = self._target_aggregator.get_dataset()
        if len(prediction) == 0 or len(target) == 0:
            return xr.Dataset()
        else:
            return xr.concat(
                [
                    target.expand_dims({"source": ["target"]}),
                    prediction.expand_dims({"source": ["prediction"]}),
                ],
                dim="source",
            )


def _calculate_sample_average_power_spectrum(
    timeseries: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """This function handles the case where samples have different lengths
    by truncating to the shortest length.
    """
    data_arrays = []
    data_lengths = []
    for sample in range(timeseries.sizes["sample"]):
        data_without_nan = timeseries.isel(sample=sample).dropna("time")
        data_arrays.append(data_without_nan)
        data_lengths.append(data_without_nan.sizes["time"])
    min_data_length = min(data_lengths)
    if max(data_lengths) != min_data_length:
        warnings.warn(
            "Samples have different lengths, truncating to shortest length "
            f"of {min_data_length} steps for power spectrum calculation. The maximum "
            f"input sample length is {max(data_lengths)} steps."
        )
    all_data = np.array(
        [
            data_array.isel(time=slice(0, min_data_length)).values
            for data_array in data_arrays
        ]
    )
    return compute_power_spectrum(all_data)


def anomalies_from_monthly_climo(
    data: torch.Tensor,
    time: xr.DataArray,
    time_dim: int = TIME_DIM,
) -> torch.Tensor:
    anomalies = torch.empty_like(data, device=data.device)
    for month in range(1, 13):
        month_mask = torch.tensor(
            time.dt.month.values == month, dtype=torch.bool, device=data.device
        )
        masked_data = data.where(
            month_mask, torch.tensor(float("nan"), device=data.device)
        )
        monthly_climo = (
            masked_data.nansum(dim=time_dim) / month_mask.sum(dim=time_dim)
        ).unsqueeze(dim=time_dim)
        anomalies = torch.where(month_mask, data - monthly_climo, anomalies)
    return anomalies


@dataclass
class UniqueMonths:
    years: torch.Tensor
    months: torch.Tensor

    @classmethod
    def from_arrays(cls, years: np.ndarray, months: np.ndarray) -> "UniqueMonths":
        if years.shape != months.shape:
            raise ValueError("Years and months arrays must have the same shape.")
        if years.ndim != 1:
            raise ValueError("Years and months arrays must be 1D.")
        unique_months_arr = np.unique(
            np.array(
                [(year, month) for year, month in zip(years, months)],
                dtype=("i,i"),
            ).astype("object")
        )
        return cls(
            years=torch.tensor(
                [unique_month[0] for unique_month in unique_months_arr],
                dtype=torch.int64,
                device=get_device(),
            ),
            months=torch.tensor(
                [unique_month[1] for unique_month in unique_months_arr],
                dtype=torch.int64,
                device=get_device(),
            ),
        )

    def __iter__(self):
        return zip(self.years, self.months)

    def __len__(self):
        return len(self.years)


def running_monthly_mean(
    data: torch.Tensor,
    time: xr.DataArray,
    n_months: int,
    time_dim: int = TIME_DIM,
    sample_dim: int = SAMPLE_DIM,
) -> tuple[torch.Tensor, UniqueMonths]:
    """Compute an n-month running mean of the input data.

    First compute the monthly mean of the data, then compute the running mean.
    """
    unique_months = UniqueMonths.from_arrays(
        time.dt.year.values.flatten(),
        time.dt.month.values.flatten(),
    )
    monthly_data = torch.full(
        (data.shape[sample_dim], len(unique_months)),
        fill_value=float("nan"),
        device=data.device,
    )
    running_monthly_data = torch.full_like(monthly_data, fill_value=float("nan"))
    for i_month, (year, month) in enumerate(unique_months):
        month_mask = torch.tensor(
            (time.dt.year.values == year.item())
            & (time.dt.month.values == month.item()),
            dtype=torch.bool,
            device=data.device,
        )
        masked_data = data.where(
            month_mask, torch.tensor(float("nan"), device=data.device)
        )
        monthly_data[:, i_month] = masked_data.nanmean(dim=TIME_DIM)
        if i_month >= n_months - 1:
            running_monthly_data[:, i_month] = monthly_data[
                :, i_month - n_months + 1 : i_month + 1
            ].nanmean(dim=time_dim)
    return running_monthly_data, unique_months


def convert_cftime_to_datetime_coord(time_coord: xr.DataArray) -> xr.DataArray:
    """Convert cftime array to datetime array that is approximately the same.
    We do this because it's necessary to plot data via plotly with datetime coordinates
    in order to get ticklabels that are intelligible, but we want to maintain a cftime
    coordinate for accuracy and saving raw data.
    """
    return xr.DataArray(
        data=[
            datetime(time.item().year, time.item().month, time.item().day)
            for time in time_coord
        ],
        dims=time_coord.dims,
    )
