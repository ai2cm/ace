from collections.abc import Mapping
from functools import partial
from typing import Any, Protocol

import matplotlib.pyplot as plt
import torch
import xarray as xr

from fme.ace.aggregator.plotting import get_cmap_limits, plot_imshow
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.histogram import DynamicHistogramAggregator
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import WandB
from fme.downscaling.aggregators.adapters import DynamicHistogramsAdapter
from fme.downscaling.aggregators.main import (
    Mean,
    ZonalPowerSpectrumAggregator,
    batch_mean,
    ensure_trailing_slash,
)
from fme.downscaling.aggregators.shape_helpers import (
    get_data_dim,
    subselect_and_squeeze,
    upsample_tensor,
)

from ..metrics_and_maths import filter_tensor_mapping, map_tensor_mapping


class _AggregatorInterface(Protocol):
    def record_batch(self, *args, **kwargs) -> None: ...
    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]: ...
    def get_dataset(self) -> xr.Dataset: ...


class NoTargetAggregator:
    """
    Aggregator used when generating downscaled outputs without any corresponding
    fine-res target data.
    Mean map outputs assume that all data passed to record_batch has the same
    lat/lon coordinates.
    """

    def __init__(
        self,
        downscale_factor: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        ensemble_dim: int = 1,
        latlon_coordinates: LatLonCoordinates | None = None,
    ):
        self.downscale_factor = downscale_factor
        self.latlon_coordinates = latlon_coordinates
        self.ensemble_dim = ensemble_dim
        self.variable_metadata = variable_metadata
        self.single_sample_aggregators: list[_AggregatorInterface] = [
            ZonalPowerSpectrumAggregator(
                downscale_factor=downscale_factor,
                name="single_sample_time_mean_power_spectrum",
            ),
            _MapAggregator(
                name="single_sample_time_mean",
                variable_metadata=variable_metadata,
            ),
        ]
        self.aggregators: list[_AggregatorInterface] = [
            TimeSeriesAggregator(
                name="time_series", variable_metadata=variable_metadata
            ),
            DynamicHistogramsAdapter(
                name="histogram",
                histograms=DynamicHistogramAggregator(
                    n_bins=300, percentiles=[99.9999, 99.99999]
                ),
            ),
        ]

    @torch.no_grad()
    def record_batch(
        self,
        prediction: TensorDict,
        coarse: TensorDict,
        time: xr.DataArray,
    ) -> None:
        for single_sample_agg in self.single_sample_aggregators:
            single_sample_agg.record_batch(
                prediction=subselect_and_squeeze(prediction, self.ensemble_dim),
                coarse=subselect_and_squeeze(coarse, self.ensemble_dim),
            )
        for agg in self.aggregators:
            agg.record_batch(prediction=prediction, coarse=coarse, time=time)

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        ret: dict[str, Any] = {}
        for single_sample_agg in self.single_sample_aggregators:
            ret.update(single_sample_agg.get_wandb(prefix))
        for agg in self.aggregators:
            ret.update(agg.get_wandb(prefix))
        return ret

    def get_dataset(self) -> xr.Dataset:
        """
        Get the dataset from all sub aggregators.
        """
        ds = xr.Dataset()
        for single_sample_agg in self.single_sample_aggregators:
            ds = ds.merge(single_sample_agg.get_dataset())
        for agg in self.aggregators:
            ds = ds.merge(agg.get_dataset())
        if self.latlon_coordinates is not None:
            ds = ds.assign_coords(self.latlon_coordinates.coords)
        return ds


def _concat_mapping_value(map0: TensorMapping, map1: TensorMapping) -> TensorMapping:
    """
    Append the values of two TensorMappings with the same keys.
    """
    if len(map0) == 0:
        return map1
    if map0.keys() != map1.keys():
        raise ValueError("Keys of both mappings must match.")
    return {k: torch.cat((map0[k], map1[k]), dim=0) for k in map0.keys()}


def _plot_timeseries(
    times: xr.DataArray,
    fine_data: xr.DataArray,
    coarse_data: xr.DataArray | None = None,
    variable_metadata: VariableMetadata | None = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for sample in fine_data.generated_sample:
        plt.plot(
            times,
            fine_data.sel(generated_sample=sample),
            linewidth=0.70,
            label=(
                f"fine samples 0-{fine_data.generated_sample.size-1}"
                if sample == 0
                else None
            ),
        )
    if coarse_data is not None:
        plt.plot(times, coarse_data, color="black", label="coarse", linestyle="-.")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.xlabel("time")
    if variable_metadata is not None:
        if fine_data.name in variable_metadata:
            caption_name = variable_metadata[fine_data.name].long_name
            units = variable_metadata[fine_data.name].units
    else:
        caption_name, units = fine_data.name, "unknown_units"
    plt.ylabel(f"{caption_name} [{units}]")
    plt.legend()

    return fig


class TimeSeriesAggregator:
    def __init__(
        self,
        name: str = "",
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._name = ensure_trailing_slash(name)
        self._variable_metadata = variable_metadata or {}
        self._max = map_tensor_mapping(partial(torch.amax, dim=(-2, -1)))
        self._mean = map_tensor_mapping(partial(torch.mean, dim=(-2, -1)))
        self.times: list[xr.DataArray] = []
        self.prediction_max_series: TensorMapping = {}
        self.prediction_mean_series: TensorMapping = {}
        self.coarse_mean_series: TensorMapping = {}

    def record_batch(
        self,
        prediction: TensorMapping,
        coarse: TensorMapping,
        time: xr.DataArray,
    ) -> None:
        """
        Tensor dimensions are [time_sample, generated_sample, lat, lon].
        Batching and concatenation of results is over the time dimension.
        """
        coarse = filter_tensor_mapping(coarse, prediction.keys())
        self.times.append(time)
        self.prediction_max_series = _concat_mapping_value(
            self.prediction_max_series, self._max(prediction)
        )
        self.prediction_mean_series = _concat_mapping_value(
            self.prediction_mean_series, self._mean(prediction)
        )
        self.coarse_mean_series = _concat_mapping_value(
            self.coarse_mean_series, self._mean(coarse)
        )

    def _get_dims(self, shape: tuple[int, ...]):
        if len(shape) == 1:
            dims = [
                "time",
            ]
        elif len(shape) == 2:
            dims = ["time", "generated_sample"]
        else:
            raise ValueError(f"Reduced tensor must be 1D or 2D, got shape: {shape}")
        return tuple(dims)

    @property
    def _all_times(self) -> xr.DataArray:
        concat_dim = self.times[0].dims[0]
        times = xr.concat(self.times, dim=concat_dim)
        return times.rename({concat_dim: "time"})

    def get_dataset(self):
        times = self._all_times
        ds = xr.Dataset()
        for k, v in self.prediction_max_series.items():
            ds[f"{k}_prediction_max"] = xr.DataArray(
                v.cpu().numpy(), dims=self._get_dims(v.shape)
            )
        for k, v in self.prediction_mean_series.items():
            ds[f"{k}_prediction_mean"] = xr.DataArray(
                v.cpu().numpy(), dims=self._get_dims(v.shape)
            )
        for k, v in self.coarse_mean_series.items():
            squeezed = v.cpu().numpy().squeeze()
            ds[f"{k}_coarse_mean"] = xr.DataArray(
                squeezed, dims=self._get_dims(squeezed.shape)
            )
        ds = ds.assign_coords({"time": times})
        return ds

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the wandb data for this aggregator.
        """
        prefix = ensure_trailing_slash(prefix)
        ds = self.get_dataset()
        ret = {}
        try:
            time_coords = ds.indexes["time"].to_datetimeindex()
        except AttributeError:
            time_coords = ds.indexes["time"]
        for key in self.prediction_max_series:
            ret[f"{prefix}{self._name}{key}_max"] = _plot_timeseries(
                time_coords,
                ds[f"{key}_prediction_max"],
                variable_metadata=self._variable_metadata.get(key),
            )
        for key in self.prediction_mean_series:
            ret[f"{prefix}{self._name}{key}_mean"] = _plot_timeseries(
                time_coords,
                ds[f"{key}_prediction_mean"],
                ds[f"{key}_coarse_mean"],
                variable_metadata=self._variable_metadata.get(key),
            )
        return ret


class _MapAggregator:
    def __init__(
        self,
        name: str = "",
        gap_width: int = 4,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._mean_prediction = Mean(batch_mean, name="prediction")
        self._mean_coarse = Mean(batch_mean, name="coarse")
        self.gap_width = gap_width
        self._name = ensure_trailing_slash(name)
        if variable_metadata is None:
            self._variable_metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._variable_metadata = variable_metadata
        self._expected_ndims = 3

    def _get_downscale_factor(self, prediction: TensorMapping, coarse: TensorMapping):
        k = list(prediction.keys())[0]
        return prediction[k].shape[-1] // coarse[k].shape[-1]

    @torch.no_grad()
    def record_batch(
        self,
        prediction: TensorMapping,
        coarse: TensorMapping,
    ) -> None:
        coarse = filter_tensor_mapping(coarse, prediction.keys())
        downscale_factor = self._get_downscale_factor(prediction, coarse)
        coarse = {k: upsample_tensor(v, downscale_factor) for k, v in coarse.items()}
        for data in [prediction, coarse]:
            ndim = get_data_dim(data)
            if ndim != self._expected_ndims:
                raise ValueError(
                    "Data passed to _MapAggregator must be 3D, i.e. any sample dim "
                    "is already folded into batch dim, or subselected and squeezed."
                )

        self._mean_prediction.record_batch(prediction)
        self._mean_coarse.record_batch(coarse)

    def _get_maps(self) -> Mapping[str, Any]:
        coarse = self._mean_coarse.get()
        prediction = self._mean_prediction.get()

        maps = {}
        for var_name in prediction.keys():
            gap = torch.full(
                (prediction[var_name].shape[-2], self.gap_width),
                float(prediction[var_name].min()),
                device=prediction[var_name].device,
            )
            maps[f"maps/{self._name}full-field/{var_name}"] = torch.cat(
                (prediction[var_name], gap, coarse[var_name]), dim=1
            )
        return maps

    def _get_caption(self, key: str, name: str, vmin: float, vmax: float) -> str:
        _caption = (
            "{name}  mean full field; (left) generated and " "(right) coarse [{units}]"
        )

        if name in self._variable_metadata:
            caption_name = self._variable_metadata[name].long_name
            units = self._variable_metadata[name].units
        else:
            caption_name, units = name, "unknown_units"
        caption = _caption.format(name=caption_name, units=units)
        caption += f" vmin={vmin:.4g}, vmax={vmax:.4g}."
        return caption

    def get_wandb(self, prefix: str = ""):
        prefix = ensure_trailing_slash(prefix)
        ret = {}
        wandb = WandB.get_instance()
        maps = self._get_maps()
        for key, data in maps.items():
            if "error" in key:
                diverging, cmap = True, "RdBu_r"
            else:
                diverging, cmap = False, None
            data = data.cpu().numpy()
            vmin, vmax = get_cmap_limits(data, diverging=diverging)
            map_name, var_name = key.split("/")[-2:]
            caption = self._get_caption(map_name, var_name, vmin, vmax)
            fig = plot_imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
            ret[f"{prefix}{key}"] = wandb.Image(fig, caption=caption)
            plt.close(fig)

        return ret

    def get_dataset(self) -> xr.Dataset:
        """
        Get the time mean maps dataset.
        """
        coarse = self._mean_coarse.get()
        prediction = self._mean_prediction.get()
        data = {}
        for key in prediction:
            data[f"{self._name}coarse.{key}"] = coarse[key].cpu().numpy()
            data[f"{self._name}prediction.{key}"] = prediction[key].cpu().numpy()
        ds = xr.Dataset({k: (("lat", "lon"), v) for k, v in data.items()})
        return ds
