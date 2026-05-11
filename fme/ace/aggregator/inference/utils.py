import abc
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import numpy as np
import torch
import xarray as xr

from fme.core.device import get_device

SAMPLE_DIM, TIME_DIM, LAT_DIM, LON_DIM = 0, 1, -2, -1


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
        region_mask = torch.logical_and(lat_mask, lon_mask).float()
        cos_lat = torch.cos(torch.deg2rad(self.lat)).unsqueeze(self.horizontal_dims[1])
        self._regional_weights = region_mask * cos_lat

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
        data: The generated index data with sample and a time dims.
        target_data: (Optional) The target index data of the same shape. If
            provided, the generated standard deviations are normalized by the target
            standard deviation prior to computing the mean across samples.
        time_dim: Time dimension index
    """
    std_by_sample = np.nanstd(data, axis=time_dim)
    if target_data is not None:
        std_by_sample = std_by_sample / np.nanstd(target_data, axis=time_dim)
    return std_by_sample.mean().item()


def _calculate_sample_average_power_spectrum(
    timeseries: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum averaged across samples.

    Handles the case where samples have different lengths by truncating to
    the shortest length.
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
