import dataclasses
import logging
from collections.abc import Callable
from typing import Any, Literal

import cftime
import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt

from fme.core.coordinates import LatLonCoordinates
from fme.core.distributed import Distributed
from fme.core.gridded_ops import LatLonOperations
from fme.core.typing_ import TensorDict

from ...plotting import plot_mean_and_samples
from ..build_context import MetricBuildContext, MetricNotSupportedError
from ..data import InferenceBatchData, MetricBuildResult
from ..utils import (
    SAMPLE_DIM,
    TIME_DIM,
    LatLonRegion,
    _calculate_sample_average_power_spectrum,
    _compute_sample_mean_std,
    anomalies_from_monthly_climo,
    compute_psd_band_power,
    convert_cftime_to_datetime_coord,
    running_monthly_mean,
)

SEA_SURFACE_TEMPERATURE_NAMES = ["sst", "surface_temperature", "TS"]


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

    def record_batch(self, data: InferenceBatchData) -> None:
        time = data.time
        prediction = data.prediction
        for sst_name in self.sea_surface_temperature_names:
            if sst_name not in prediction:
                if sst_name not in self._already_logged:
                    logging.info(
                        f"Variable {sst_name} not found in data. "
                        "Skipping Nino3.4 index computation for this variable."
                    )
                    self._already_logged.append(sst_name)
                continue
            regional_average = self._regional_mean(
                prediction[sst_name], self._regional_weights
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
                logs[f"{sst_name}_nino34_index_power_2_5yr"] = compute_psd_band_power(
                    freq, power_spectrum
                )
                logs[f"{sst_name}_nino34_index_power_1_16yr"] = compute_psd_band_power(
                    freq, power_spectrum, period_bounds=(1.0, 16.0)
                )

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
        data: InferenceBatchData,
    ) -> None:
        target_data = data.replace(prediction=data.target)
        prediction_data = data.replace(prediction=data.prediction)
        self._target_aggregator.record_batch(target_data)
        self._prediction_aggregator.record_batch(prediction_data)

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
                pred_power_2_5 = compute_psd_band_power(
                    pred_freq, prediction_power_spectrum
                )
                target_power_2_5 = compute_psd_band_power(
                    target_freq, target_power_spectrum
                )
                logs[f"{sst_name}_nino34_index_power_2_5yr"] = pred_power_2_5
                if target_power_2_5 != 0 and not np.isnan(target_power_2_5):
                    logs[f"{sst_name}_nino34_index_power_2_5yr_norm"] = (
                        pred_power_2_5 / target_power_2_5
                    )
                pred_power_1_16 = compute_psd_band_power(
                    pred_freq, prediction_power_spectrum, period_bounds=(1.0, 16.0)
                )
                target_power_1_16 = compute_psd_band_power(
                    target_freq, target_power_spectrum, period_bounds=(1.0, 16.0)
                )
                logs[f"{sst_name}_nino34_index_power_1_16yr"] = pred_power_1_16
                if target_power_1_16 != 0 and not np.isnan(target_power_1_16):
                    logs[f"{sst_name}_nino34_index_power_1_16yr_norm"] = (
                        pred_power_1_16 / target_power_1_16
                    )
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


NINO34_LAT = (-5, 5)
NINO34_LON = (190, 240)


@dataclasses.dataclass
class EnsoIndexMetricConfig:
    type: Literal["enso_index"] = "enso_index"
    name: str = "enso_index"

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        if not isinstance(ctx.horizontal_coordinates, LatLonCoordinates):
            raise MetricNotSupportedError(
                "enso_index metric requires LatLonCoordinates."
            )
        if not isinstance(ctx.ops, LatLonOperations):
            raise MetricNotSupportedError(
                "enso_index metric requires LatLonOperations."
            )
        nino34_region = LatLonRegion(
            lat_bounds=NINO34_LAT,
            lon_bounds=NINO34_LON,
            lat=ctx.horizontal_coordinates.lat,
            lon=ctx.horizontal_coordinates.lon,
        )
        return MetricBuildResult(
            aggregator=PairedRegionalIndexAggregator(
                target_aggregator=RegionalIndexAggregator(
                    regional_weights=nino34_region.regional_weights,
                    regional_mean=ctx.ops.regional_area_weighted_mean,
                ),
                prediction_aggregator=RegionalIndexAggregator(
                    regional_weights=nino34_region.regional_weights,
                    regional_mean=ctx.ops.regional_area_weighted_mean,
                ),
            )
        )
