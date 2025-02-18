import abc
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import cftime
import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping

from ...plotting import plot_mean_and_samples

SAMPLE_DIM, TIME_DIM, LAT_DIM, LON_DIM = 0, 1, -2, -1


class Region(abc.ABC):
    @property
    @abc.abstractmethod
    def regional_weights(self) -> torch.Tensor: ...


@dataclass
class LatLonRegion(Region):
    lat: torch.Tensor
    lon: torch.Tensor
    lat_bounds: Tuple[float, float]
    lon_bounds: Tuple[float, float]
    horizontal_dims: ClassVar[Tuple[int, int]] = (LAT_DIM, LON_DIM)

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

    @property
    def regional_weights(self) -> torch.Tensor:
        return self._regional_weights


class RegionalIndexAggregator:
    """Aggregator for computing a regional index, in this case a monthly- and area-
    weighted average of a variable over a region.

    Args:
        regional_weights: A tensor of weights for the region.
        regional_mean: A function that computes the regional mean of a variable,
            given weights.
        variable_name: The name of the variable for which to compute the index.
        running_mean_n_months: The number of months to use for the running mean.
    """

    _sample_dim = SAMPLE_DIM
    _time_dim = TIME_DIM

    def __init__(
        self,
        regional_weights: torch.Tensor,
        regional_mean: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        variable_name: str,
        running_mean_n_months: int = 5,
    ):
        self._regional_weights = regional_weights
        self._regional_mean = regional_mean
        self._variable_name: str = variable_name
        self._running_mean_n_months = running_mean_n_months
        self._raw_index_values: Optional[torch.Tensor] = None
        self._raw_index_times: Optional[xr.DataArray] = None
        self._calendar: Optional[str] = None

    def record_batch(self, data: TensorMapping, time: xr.DataArray) -> None:
        if self._variable_name not in data:
            logging.info(
                f"Variable {self._variable_name} not found in data. "
                "Skipping RegionalIndexAggregator.",
            )
            return None
        regional_average = self._regional_mean(
            data[self._variable_name], self._regional_weights
        )
        if self._raw_index_values is None:
            self._raw_index_values = regional_average
        else:
            self._raw_index_values = torch.cat(
                [self._raw_index_values, regional_average], dim=self._time_dim
            )
        if self._raw_index_times is None:
            self._raw_index_times = time
        else:
            self._raw_index_times = xr.concat([self._raw_index_times, time], dim="time")
        if self._calendar is None and time.dt.calendar is not None:
            self._calendar = time.dt.calendar

    def _get_index(self) -> Optional[xr.DataArray]:
        if self._raw_index_values is None:
            return None
        anomaly_index_values = anomalies_from_monthly_climo(
            self._raw_index_values, self._raw_index_times
        )
        time_averaged_index, unique_months = running_monthly_mean(
            anomaly_index_values,
            self._raw_index_times,
            n_months=self._running_mean_n_months,
        )
        return self._get_gathered_index(
            time_averaged_index, unique_months.years, unique_months.months
        )

    def _get_gathered_index(
        self, index: torch.Tensor, years: torch.Tensor, months: torch.Tensor
    ) -> Optional[xr.DataArray]:
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
        indices: List[torch.Tensor],
        years: List[torch.Tensor],
        months: List[torch.Tensor],
    ):
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

    def get_logs(self, label: str) -> Dict[str, Any]:
        index = self._get_index()
        if index is None:  # not the root rank
            return {}
        else:
            fig, ax = plt.subplots(1, 1)
            if index.sizes["time"] > 1:
                index_plottable = index.assign_coords(
                    {"time": convert_cftime_to_datetime_coord(index.time)}
                )
                plot_mean_and_samples(
                    ax, index_plottable, "ensemble_mean", time_series_dim="time"
                )
            ax.set_title("Nino3.4 Index")
            ax.set_ylabel("K")
            ax.legend()
            fig.tight_layout()

        if len(label) > 0:
            label = label + "/"

        return {f"{label}nino34_index": fig}

    def get_dataset(self) -> xr.Dataset:
        index = self._get_index()
        if index is None:
            return xr.Dataset()
        else:
            return xr.Dataset({"nino34_index": index})


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
        self._target_aggregator.record_batch(target_data, time)
        self._prediction_aggregator.record_batch(gen_data, time)

    def get_logs(self, label: str) -> Dict[str, Any]:
        target_index = self._target_aggregator._get_index()
        prediction_index = self._prediction_aggregator._get_index()
        if prediction_index is None or target_index is None:  # not the root rank
            return {}
        else:
            target_index_plottable = target_index.assign_coords(
                {"time": convert_cftime_to_datetime_coord(target_index.time)}
            )
            prediction_index_plottable = prediction_index.assign_coords(
                {"time": convert_cftime_to_datetime_coord(prediction_index.time)}
            )
            fig, ax = plt.subplots(1, 1)
            if prediction_index_plottable.sizes["time"] > 1:
                plot_mean_and_samples(
                    ax,
                    prediction_index_plottable,
                    "predicted ensemble mean",
                    time_series_dim="time",
                )
                plot_mean_and_samples(
                    ax,
                    target_index_plottable,
                    "target",
                    time_series_dim="time",
                    color="orange",
                    plot_samples=False,
                )
            ax.set_title("Nino3.4 Index")
            ax.set_ylabel("K")
            ax.legend()
            fig.tight_layout()

        if len(label) > 0:
            label = label + "/"

        return {f"{label}nino34_index": fig}

    def get_dataset(self) -> xr.Dataset:
        prediction = self._prediction_aggregator.get_dataset()
        target = self._target_aggregator.get_dataset()
        return xr.concat(
            [
                target.expand_dims({"source": ["target"]}),
                prediction.expand_dims({"source": ["prediction"]}),
            ],
            dim="source",
        )


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
) -> Tuple[torch.Tensor, UniqueMonths]:
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
