import logging
import warnings
from typing import Any

import cftime
import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt
from scipy import signal

from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict

from ...plotting import plot_mean_and_samples
from ..data import InferenceBatchData
from ..enso.dynamic_index import (
    LatLonRegion,
    UniqueMonths,
    anomalies_from_monthly_climo,
    compute_power_spectrum,
    convert_cftime_to_datetime_coord,
    running_monthly_mean,
)

SAMPLE_DIM, TIME_DIM = 0, 1

DEFAULT_SST_NAMES = ["sst"]

TPI_REGIONS = {
    "T1": {"lat_bounds": (25.0, 45.0), "lon_bounds": (140.0, 215.0)},
    "T2": {"lat_bounds": (-10.0, 10.0), "lon_bounds": (170.0, 270.0)},
    "T3": {"lat_bounds": (-50.0, -15.0), "lon_bounds": (150.0, 200.0)},
}


def _nan_aware_regional_mean(data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute area-weighted regional mean, excluding NaN points.

    Args:
        data: Tensor of shape (sample, time, lat, lon), may contain NaN.
        weights: Tensor of shape (lat, lon) with regional area weights.

    Returns:
        Tensor of shape (sample, time) with the NaN-excluded weighted mean.
    """
    valid_mask = ~torch.isnan(data)
    data_filled = torch.where(valid_mask, data, torch.zeros_like(data))
    # weights broadcast: (lat, lon) -> (1, 1, lat, lon)
    w = weights.unsqueeze(0).unsqueeze(0)
    weighted_sum = (data_filled * w * valid_mask).sum(dim=(-2, -1))
    weight_total = (w * valid_mask).sum(dim=(-2, -1))
    return weighted_sum / weight_total


def low_pass_filter(
    data: np.ndarray,
    cutoff_period_yrs: float = 13.0,
    sampling_freq_per_yr: int = 12,
    filter_order: int = 5,
    passband_ripple_db: float = 0.5,
) -> np.ndarray:
    """Apply a Chebyshev Type I low-pass filter (zero-phase) to a 1-d array.

    Args:
        data: Input time series (monthly values).
        cutoff_period_yrs: Cutoff period in years.
        sampling_freq_per_yr: Number of samples per year.
        filter_order: Filter order.
        passband_ripple_db: Passband ripple in dB.

    Returns:
        Filtered array of same shape as input.
    """
    nyquist_freq = 0.5 * sampling_freq_per_yr
    cutoff_freq = 1.0 / cutoff_period_yrs
    wn = cutoff_freq / nyquist_freq

    b, a = signal.cheby1(
        N=filter_order, rp=passband_ripple_db, Wn=wn, btype="low", analog=False
    )
    return signal.filtfilt(b, a, data)


def _compute_sample_mean_std_ratio(
    prediction_data: np.ndarray,
    target_data: np.ndarray,
    time_dim: int = TIME_DIM,
) -> float:
    """Compute std ratio (prediction / target) averaged over samples.

    Args:
        prediction_data: Array of shape (sample, time).
        target_data: Array of shape (sample, time).
        time_dim: Time dimension index.

    Returns:
        Mean of per-sample std(prediction) / std(target).
    """
    pred_std = np.nanstd(prediction_data, axis=time_dim)
    target_std = np.nanstd(target_data, axis=time_dim)
    ratios = pred_std / target_std
    return float(np.nanmean(ratios))


def _smooth_spectrum(
    freqs: np.ndarray, power: np.ndarray, n_bins: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a power spectrum using logarithmically-spaced frequency bins.

    Preserves resolution at low frequencies (where IPO signal lives)
    while averaging out noise at high frequencies.
    """
    positive_mask = freqs > 0
    if not positive_mask.any():
        return freqs, power
    pos_freqs = freqs[positive_mask]
    pos_power = power[positive_mask]

    log_bins = np.logspace(np.log10(pos_freqs[0]), np.log10(pos_freqs[-1]), n_bins + 1)
    bin_centers = []
    bin_powers = []
    for i in range(len(log_bins) - 1):
        mask = (pos_freqs >= log_bins[i]) & (pos_freqs < log_bins[i + 1])
        if mask.any():
            bin_centers.append(np.sqrt(log_bins[i] * log_bins[i + 1]))
            bin_powers.append(pos_power[mask].mean())
    return np.array(bin_centers), np.array(bin_powers)


def _calculate_sample_average_power_spectrum(
    timeseries: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum averaged across samples, handling NaN truncation."""
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
            f"of {min_data_length} steps for IPO TPI power spectrum. The maximum "
            f"input sample length is {max(data_lengths)} steps."
        )
    all_data = np.array(
        [
            data_array.isel(time=slice(0, min_data_length)).values
            for data_array in data_arrays
        ]
    )
    return compute_power_spectrum(all_data)


class _IPORegionalAccumulator:
    """Accumulates area-weighted regional means for the three TPI regions.

    Uses NaN-aware weighted mean so that ocean-only SST fields (with NaN
    over land) are handled correctly.
    """

    def __init__(
        self,
        regions: dict[str, LatLonRegion],
        sst_names: list[str] | None = None,
    ):
        self._regions = regions
        self._sst_names = sst_names if sst_names is not None else DEFAULT_SST_NAMES
        self._raw_means: dict[str, TensorDict] = {region: {} for region in regions}
        self._raw_times: xr.DataArray | None = None
        self._calendar: str | None = None
        self._already_logged: list[str] = []

    def record_batch(self, data: InferenceBatchData) -> None:
        time = data.time
        prediction = data.prediction
        for sst_name in self._sst_names:
            if sst_name not in prediction:
                if sst_name not in self._already_logged:
                    logging.info(
                        f"Variable {sst_name} not found in data. "
                        "Skipping IPO TPI computation for this variable."
                    )
                    self._already_logged.append(sst_name)
                continue
            for region_name, region in self._regions.items():
                regional_avg = _nan_aware_regional_mean(
                    prediction[sst_name], region.regional_weights
                )
                if sst_name not in self._raw_means[region_name]:
                    self._raw_means[region_name][sst_name] = regional_avg
                else:
                    self._raw_means[region_name][sst_name] = torch.cat(
                        [self._raw_means[region_name][sst_name], regional_avg],
                        dim=TIME_DIM,
                    )
        if self._raw_times is None:
            self._raw_times = time
        else:
            self._raw_times = xr.concat([self._raw_times, time], dim="time")
        if self._calendar is None and time.dt.calendar is not None:
            self._calendar = time.dt.calendar

    def get_tpi_indices(self) -> xr.Dataset:
        """Compute TPI = T2_anom - 0.5 * (T1_anom + T3_anom) for each SST var.

        Returns monthly TPI (not low-pass filtered).
        """
        indices = {}
        for sst_name in self._sst_names:
            if not all(sst_name in self._raw_means[r] for r in self._regions):
                continue

            regional_anomalies = {}
            for region_name in self._regions:
                raw = self._raw_means[region_name][sst_name]
                anom = anomalies_from_monthly_climo(raw, self._raw_times)
                monthly_mean, unique_months = running_monthly_mean(
                    anom, self._raw_times, n_months=1
                )
                regional_anomalies[region_name] = monthly_mean

            tpi = regional_anomalies["T2"] - 0.5 * (
                regional_anomalies["T1"] + regional_anomalies["T3"]
            )

            gathered_tpi = self._gather_index(tpi, unique_months)
            if gathered_tpi is not None:
                indices[sst_name] = gathered_tpi
        return xr.Dataset(indices)

    def _gather_index(
        self, index: torch.Tensor, unique_months: UniqueMonths
    ) -> xr.DataArray | None:
        dist = Distributed.get_instance()
        if dist.world_size > 1:
            gathered_index = dist.gather_irregular(index)
            gathered_years = dist.gather_irregular(unique_months.years)
            gathered_months = dist.gather_irregular(unique_months.months)
            if (
                gathered_index is None
                or gathered_years is None
                or gathered_months is None
            ):
                return None
        else:
            gathered_index = [index]
            gathered_years = [unique_months.years]
            gathered_months = [unique_months.months]
        return self._to_data_array(gathered_index, gathered_years, gathered_months)

    def _to_data_array(
        self,
        indices: list[torch.Tensor],
        years: list[torch.Tensor],
        months: list[torch.Tensor],
    ) -> xr.DataArray:
        calendar = self._calendar if self._calendar is not None else "standard"
        index_data_arrays = []
        for index, year, month in zip(indices, years, months):
            time_coord = [
                cftime.datetime(
                    single_year.item(), single_month.item(), 15, calendar=calendar
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


class PairedIPOIndexAggregator:
    """Paired (target, prediction) aggregator for IPO Tripolar Index.

    Computes the TPI (Henley et al. 2017) from model SST output,
    applies a 13-year Chebyshev low-pass filter, and reports:
    - Filtered TPI time series plot (prediction vs target)
    - Power spectrum of unfiltered monthly TPI
    - Scalar std ratio of filtered prediction vs target TPI
    """

    def __init__(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        cutoff_period_yrs: float = 13.0,
        sst_names: list[str] | None = None,
    ):
        regions = {
            name: LatLonRegion(
                lat=lat,
                lon=lon,
                lat_bounds=spec["lat_bounds"],
                lon_bounds=spec["lon_bounds"],
            )
            for name, spec in TPI_REGIONS.items()
        }
        self._sst_names = sst_names if sst_names is not None else DEFAULT_SST_NAMES
        self._target_accumulator = _IPORegionalAccumulator(
            regions, sst_names=self._sst_names
        )
        self._prediction_accumulator = _IPORegionalAccumulator(
            regions, sst_names=self._sst_names
        )
        self._cutoff_period_yrs = cutoff_period_yrs

    def record_batch(self, data: InferenceBatchData) -> None:
        target_data = data.replace(prediction=data.target)
        prediction_data = data.replace(prediction=data.prediction)
        self._target_accumulator.record_batch(target_data)
        self._prediction_accumulator.record_batch(prediction_data)

    def get_logs(self, label: str) -> dict[str, Any]:
        target_tpi = self._target_accumulator.get_tpi_indices()
        prediction_tpi = self._prediction_accumulator.get_tpi_indices()
        logs: dict[str, Any] = {}

        for sst_name in self._sst_names:
            if sst_name not in prediction_tpi or sst_name not in target_tpi:
                continue
            pred_da = prediction_tpi[sst_name]
            tgt_da = target_tpi[sst_name]
            if pred_da.sizes["time"] < 2:
                continue

            pred_filtered = self._apply_filter_to_samples(pred_da)
            tgt_filtered = self._apply_filter_to_samples(tgt_da)

            if pred_filtered is not None and tgt_filtered is not None:
                fig = self._plot_filtered_tpi(pred_filtered, tgt_filtered)
                logs[f"{sst_name}_ipo_tpi_filtered"] = fig

                std_ratio = _compute_sample_mean_std_ratio(
                    pred_filtered.values, tgt_filtered.values
                )
                logs[f"{sst_name}_ipo_tpi_std_ratio"] = std_ratio

            logs[f"{sst_name}_ipo_tpi_std_norm"] = _compute_sample_mean_std_ratio(
                pred_da.values, tgt_da.values
            )

            pred_nonan = pred_da.dropna("time")
            tgt_nonan = tgt_da.dropna("time")
            if pred_nonan.sizes["time"] > 1 and tgt_nonan.sizes["time"] > 1:
                fig = self._plot_power_spectrum(pred_da, tgt_da, sst_name)
                logs[f"{sst_name}_ipo_tpi_power_spectrum"] = fig

        if len(label) > 0:
            label = label + "/"
        return {f"{label}{k}": v for k, v in logs.items()}

    def get_dataset(self) -> xr.Dataset:
        prediction_tpi = self._prediction_accumulator.get_tpi_indices()
        target_tpi = self._target_accumulator.get_tpi_indices()
        if len(prediction_tpi) == 0 or len(target_tpi) == 0:
            return xr.Dataset()
        return xr.concat(
            [
                target_tpi.expand_dims({"source": ["target"]}),
                prediction_tpi.expand_dims({"source": ["prediction"]}),
            ],
            dim="source",
        )

    def _apply_filter_to_samples(self, tpi_da: xr.DataArray) -> xr.DataArray | None:
        """Apply low-pass filter to each sample, trimming edge transients.

        Trims one cutoff period from each end to remove filtfilt edge artifacts.
        Returns None if the series is too short after trimming.
        """
        trim = int(self._cutoff_period_yrs * 12)
        min_length = 3 * trim  # need at least one cutoff period after trimming
        filtered_samples = []
        for sample in range(tpi_da.sizes["sample"]):
            sample_data = tpi_da.isel(sample=sample).dropna("time")
            if sample_data.sizes["time"] < min_length:
                return None
            filtered_values = low_pass_filter(
                sample_data.values, cutoff_period_yrs=self._cutoff_period_yrs
            )
            trimmed = xr.DataArray(
                filtered_values[trim:-trim],
                coords={"time": sample_data.time[trim:-trim]},
                dims=sample_data.dims,
            )
            filtered_samples.append(trimmed)
        return xr.concat(filtered_samples, dim="sample")

    def _plot_filtered_tpi(
        self, prediction: xr.DataArray, target: xr.DataArray
    ) -> plt.Figure:
        fig, ax = plt.subplots(1, 1)
        pred_plottable = prediction.assign_coords(
            {"time": convert_cftime_to_datetime_coord(prediction.time)}
        )
        tgt_plottable = target.assign_coords(
            {"time": convert_cftime_to_datetime_coord(target.time)}
        )
        plot_mean_and_samples(
            ax,
            pred_plottable,
            "predicted ensemble mean",
            time_series_dim="time",
        )
        plot_mean_and_samples(
            ax,
            tgt_plottable,
            "target",
            time_series_dim="time",
            color="orange",
            plot_samples=False,
        )
        ax.set_title("IPO TPI (13-yr low-pass filtered)")
        ax.set_ylabel("K")
        ax.legend()
        fig.tight_layout()
        return fig

    def _plot_power_spectrum(
        self,
        prediction_tpi: xr.DataArray,
        target_tpi: xr.DataArray,
        sst_name: str,
    ) -> plt.Figure:
        pred_freq, pred_power = _calculate_sample_average_power_spectrum(prediction_tpi)
        tgt_freq, tgt_power = _calculate_sample_average_power_spectrum(target_tpi)
        pred_freq_s, pred_power_s = _smooth_spectrum(pred_freq, pred_power)
        tgt_freq_s, tgt_power_s = _smooth_spectrum(tgt_freq, tgt_power)
        fig, ax = plt.subplots(1, 1)
        ax.plot(pred_freq_s, pred_power_s, label="predicted ensemble mean")
        ax.plot(tgt_freq_s, tgt_power_s, label="target", color="orange")
        ax.set_title("Power Spectrum of IPO TPI (unfiltered)")
        ax.set_xlabel("Frequency [cycles/year]")
        ax.set_ylabel("Power [K**2]")
        ax.set(yscale="log")
        ax.legend()
        fig.tight_layout()
        return fig
