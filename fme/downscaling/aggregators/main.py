"""Contains classes for aggregating evaluation metrics and various statistics."""

from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from fme.ace.aggregator.one_step.snapshot import (
    SnapshotAggregator as CoreSnapshotAggregator,
)
from fme.ace.aggregator.plotting import get_cmap_limits, plot_imshow
from fme.core import metrics
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.histogram import ComparedDynamicHistograms
from fme.core.typing_ import TensorMapping
from fme.core.wandb import WandB

from ..datasets import PairedBatchData
from ..metrics_and_maths import (
    compute_zonal_power_spectrum,
    filter_tensor_mapping,
    interpolate,
    map_tensor_mapping,
)
from ..models import ModelOutputs
from .shape_helpers import _check_batch_dims_for_recording
from .typing import (
    ComparisonInputFunc,
    SingleInputFunc,
    _CoarseComparisonAggregator,
    _ComparisonAggregator,
    _DynamicMetricComparisonAggregator,
)


def _ensure_trailing_slash(key: str) -> str:
    if key and not key.endswith("/"):
        key += "/"
    return key


def _tensor_mapping_to_numpy(data: TensorMapping) -> TensorMapping:
    return {k: v.cpu().numpy() for k, v in data.items()}


class Mean:
    """
    Tracks a running average of a metric over multiple batches.

    Args:
        metric: The metric function to be calculated and added to the running average.
        name: The name to use for the metric in the wandb output.
    """

    def __init__(
        self,
        metric: SingleInputFunc,
        name: str = "",
    ) -> None:
        self._mapped_metric = map_tensor_mapping(metric)
        self._sum: Optional[TensorMapping] = None
        self._count: int = 0
        self._add = map_tensor_mapping(torch.add)
        self._dist = Distributed.get_instance()
        self._name = _ensure_trailing_slash(name)

    @torch.no_grad()
    def record_batch(self, data: TensorMapping) -> None:
        """
        Record the metric for the current batch.
        """
        metric = self._mapped_metric(data)

        if self._sum is None:
            self._sum = {k: torch.zeros_like(v) for k, v in metric.items()}

        self._sum = self._add(self._sum, metric)
        self._count += 1

    def get(self) -> TensorMapping:
        """
        Calculates and return the current mean of the metric values.
        """
        if self._sum is None:
            raise ValueError("No values have been added to the running average")
        return {
            k: self._dist.reduce_mean(self._sum[k] / self._count) for k in self._sum
        }

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the running average metric logs for wandb.
        """
        prefix = _ensure_trailing_slash(prefix)
        return {
            f"{prefix}{self._name}{k}": v.detach().cpu().numpy()
            for k, v in self.get().items()
        }


class SumComparison:
    """
    Tracks a running sum of a comparison metric over multiple batches.

    Args:
        metric: The metric function to be summed which compares two tensors
        (e.g. target and prediction)
        name: The name to use for the metric in the wandb output.
    """

    def __init__(
        self,
        metric: ComparisonInputFunc,
        name: str = "",
    ) -> None:
        self._mapped_metric = map_tensor_mapping(metric)
        self._sum: Optional[TensorMapping] = None
        self._add = map_tensor_mapping(torch.add)
        self._dist = Distributed.get_instance()
        self._count = 0
        self._name = _ensure_trailing_slash(name)

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping) -> None:
        """
        Record the metric for the current target and prediction batch.
        """
        metric = self._mapped_metric(target, prediction)

        if self._sum is None:
            self._sum = {k: torch.zeros_like(v) for k, v in metric.items()}

        self._sum = self._add(self._sum, metric)
        self._count += 1

    def get(self) -> TensorMapping:
        """
        Calculates and return the current sum of the metric values.
        """
        if self._sum is None:
            raise ValueError("No values have been added to the running sum")
        return {k: self._dist.reduce_sum(self._sum[k]) for k in self._sum}

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the running sum comparison metric logs for wandb.
        """
        prefix = _ensure_trailing_slash(prefix)
        return {
            f"{prefix}{self._name}{k}": v.cpu().numpy() for k, v in self.get().items()
        }


class MeanComparison:
    """
    Tracks a running average of a comparison metric over multiple batches.

    Args:
        metric: The metric function to be averaged which compares two tensors
        (e.g. target and prediction) If not provided, you must provide a
        dynamic metric to the record_batch method.

    Raises:
        ValueError: If no values have been added to the running average or if
            no default metric or dynamic metric is provided.
    """

    def __init__(
        self,
        metric: Optional[ComparisonInputFunc] = None,
        name: str = "",
    ) -> None:
        self._mapped_metric = map_tensor_mapping(metric) if metric is not None else None
        self._sum: Optional[TensorMapping] = None
        self._count: int = 0
        self._add = map_tensor_mapping(torch.add)
        self._dist = Distributed.get_instance()
        self._name = _ensure_trailing_slash(name)

    @torch.no_grad()
    def record_batch(
        self,
        target: TensorMapping,
        prediction: TensorMapping,
        dynamic_metric: Optional[ComparisonInputFunc] = None,
    ) -> None:
        """
        Record the metric for the current target and prediction batch.

        Args:
            target: the 'truth' target values.
            prediction: the predicted values.
            dynamic_metric: a metric that changes with the batch. overrides the
                default metric if both are provided.
        """
        if dynamic_metric is not None:
            metric = map_tensor_mapping(dynamic_metric)(target, prediction)
        elif self._mapped_metric is not None:
            metric = self._mapped_metric(target, prediction)
        else:
            raise ValueError("No metric function provided to MeanComparisonAggregator")

        if self._sum is None:
            self._sum = {k: torch.zeros_like(v) for k, v in metric.items()}

        self._sum = self._add(self._sum, metric)
        self._count += 1

    def get(self) -> TensorMapping:
        """
        Calculates and returns the current mean of the comparison metric values.
        """
        if self._sum is None:
            raise ValueError("No values have been added to the running average")
        return {
            k: self._dist.reduce_mean(self._sum[k] / self._count) for k in self._sum
        }

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the running average comparison metric logs for wandb.
        """
        prefix = _ensure_trailing_slash(prefix)
        return {
            f"{prefix}{self._name}{k}": v.cpu().numpy() for k, v in self.get().items()
        }


class ComparedDynamicHistogramsAdapter:
    """
    Adapter to use ComparedDynamicHistograms with the naming and prefix
    scheme used by downscaling aggregators.

    Args:
        histograms: The ComparedDynamicHistograms object to adapt.
        name: The name to use for the histograms in the wandb output.
    """

    def __init__(self, histograms: ComparedDynamicHistograms, name: str = "") -> None:
        self._histograms = histograms
        self._name = _ensure_trailing_slash(name)

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping) -> None:
        """
        Record the histograms for the current batch comparison.
        """
        self._histograms.record_batch(target, prediction)

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the histogram logs for wandb.
        """
        prefix = _ensure_trailing_slash(prefix)
        histograms = self._histograms.get_wandb()
        return {f"{prefix}{self._name}{k}": v for k, v in histograms.items()}


def _compute_zonal_mean_power_spectrum(x):
    if not len(x.shape) == 3:
        raise ValueError(
            f"Expected input (batch, height, width) but received {x.shape}"
        )

    # (batch, wavenumber) -> (wavenumber,)
    return compute_zonal_power_spectrum(x).mean(axis=0)


class ZonalPowerSpectrumAggregator:
    """
    Record the mean zonal power spectrum for the input coarse data
    and the predicted fine data.  Coarse data is interpolated to
    the fine grid before calculating the spectrum.

    Args:
        downscale_factor: The factor by which the coarse data has been downscaled.
        name: The name to use for the spectrum in the wandb output.
    """

    def __init__(self, downscale_factor: int, name: str = "") -> None:
        self.wandb = WandB.get_instance()
        self._name = _ensure_trailing_slash(name)
        self._coarse_spectrum = Mean(_compute_zonal_mean_power_spectrum)
        self._fine_spectrum = Mean(_compute_zonal_mean_power_spectrum)
        self.downscale_factor = downscale_factor

    def _interpolate(
        self, data: TensorMapping, keys: Optional[Collection[str]] = None
    ) -> TensorMapping:
        # interpolate expects 4D input (batch, channel, height, width)
        # for 2D - bicubic interpolation, so we need to expand below
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        channel_dim = -3
        keys = keys if keys is not None else data.keys()
        return {
            k: interpolate(v.unsqueeze(channel_dim), self.downscale_factor).squeeze(
                channel_dim
            )
            for k, v in data.items()
            if k in keys
        }

    @torch.no_grad()
    def record_batch(self, prediction: TensorMapping, coarse: TensorMapping) -> None:
        """
        Record the zonal power spectrum for the current batch.
        """
        self._fine_spectrum.record_batch(prediction)
        self._coarse_spectrum.record_batch(
            self._interpolate(coarse, keys=list(prediction.keys()))
        )

    def get(self) -> Tuple[TensorMapping, TensorMapping]:
        """
        Get the mean zonal power spectrum tensors for the fine and coarse data.
        """
        fine = self._fine_spectrum.get()
        coarse = self._coarse_spectrum.get()
        return fine, coarse

    def _plot_spectrum(self, prediction: np.ndarray, coarse: np.ndarray) -> Any:
        fig = plt.figure()
        plt.loglog(prediction, linestyle="--", label="prediction")
        plt.loglog(coarse, linestyle="-.", label="coarse")
        plt.legend()
        plt.grid()
        plt.close(fig)
        return fig

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the zonal power spectrum logs for wandb.
        """
        prefix = _ensure_trailing_slash(prefix)
        fine, coarse = self.get()
        fine = _tensor_mapping_to_numpy(fine)
        coarse = _tensor_mapping_to_numpy(coarse)
        ret = {}
        for name, values in fine.items():
            ret[f"{prefix}{self._name}{name}"] = self._plot_spectrum(
                values, coarse[name]
            )
        return ret


class ZonalPowerSpectrumComparison:
    """
    Record the zonal power spectrum for the interpolated input coarse data, and
    the predicted and target fine data.  Coarse data is interpolated to the fine
    grid before calculating the spectrum.

    Args:
        downscale_factor: The factor by which the data has been downscaled.
        name: The name to use for the spectrum in the wandb output.
    """

    def __init__(self, downscale_factor: int, name: str = "") -> None:
        self._name = _ensure_trailing_slash(name)
        self._mean_target_aggregator = Mean(_compute_zonal_mean_power_spectrum)
        # power spectrum for prediction and coarse data
        self._mean_prediction_aggregator = ZonalPowerSpectrumAggregator(
            downscale_factor, name=name
        )

    @torch.no_grad()
    def record_batch(
        self, target: TensorMapping, prediction: TensorMapping, coarse: TensorMapping
    ) -> None:
        """
        Record the zonal power spectrum for the current batch.

        Args:
            target: The target fine data for the current batch.
            prediction: The predicted fine data for the current batch.
            coarse: The coarse data for the current batch.
        """
        self._mean_prediction_aggregator.record_batch(prediction, coarse)
        self._mean_target_aggregator.record_batch(target)

    def _plot_spectrum_all(
        self, target: np.ndarray, prediction: np.ndarray, coarse: np.ndarray
    ) -> Any:
        fig = plt.figure()
        plt.loglog(target, label="target")
        plt.loglog(prediction, linestyle="--", label="prediction")
        plt.loglog(coarse, linestyle="-.", label="coarse")
        plt.legend()
        plt.grid()
        plt.close(fig)
        return fig

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the zonal power spectrum logs for wandb.
        """
        prefix = _ensure_trailing_slash(prefix)
        prediction, coarse = self._mean_prediction_aggregator.get()
        target = self._mean_target_aggregator.get()
        prediction = _tensor_mapping_to_numpy(prediction)
        coarse = _tensor_mapping_to_numpy(coarse)
        target = _tensor_mapping_to_numpy(target)
        ret = {}
        for name, values in target.items():
            ret[f"{prefix}{self._name}{name}"] = self._plot_spectrum_all(
                values, prediction[name], coarse[name]
            )
        return ret


class SnapshotAggregator:
    """
    Records a snapshot of the target and prediction tensors for visualization.

    Args:
        variable_metadata: metadata for the variables to use for snapshot caption
    """

    def __init__(
        self,
        dims: List[str],
        variable_metadata: Optional[Mapping[str, VariableMetadata]],
        name: str = "",
    ) -> None:
        self._snapshot_aggregator = CoreSnapshotAggregator(dims, variable_metadata)
        self._name = _ensure_trailing_slash(name)

    def _tile_time_dim(self, x: torch.Tensor) -> torch.Tensor:
        time_dim = -3
        return x.unsqueeze(time_dim).repeat_interleave(2, dim=time_dim)

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping) -> None:
        """
        Record values for a snapshot of the target and prediction tensors.
        """
        # core SnapshotAggregator expects a time dimension of length two, so we
        # provide it by tiling the data.
        target = map_tensor_mapping(self._tile_time_dim)(target)
        prediction = map_tensor_mapping(self._tile_time_dim)(prediction)
        self._snapshot_aggregator.record_batch(-1.0, target, prediction, {}, {})

    def _get(self, label: str = "") -> Mapping[str, Any]:
        label = _ensure_trailing_slash(label)
        logs = self._snapshot_aggregator.get_logs(label)
        ret = {}
        for k, v in logs.items():
            k = k.removeprefix("/")
            # residual is meaningless for single steps
            if "residual" not in k:
                # The core SnapshotAggregator returns {label}/{key} even when
                # label == "". In this case, removing the leading slash.
                ret[f"{label}maps/{self._name}{k}"] = v
        return ret

    def get_wandb(self, label: str = "") -> Mapping[str, Any]:
        """
        Retrieve a snapshot from the currently stored batch for wandb.
        """
        label = _ensure_trailing_slash(label)
        return self._get(label)


def batch_mean(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    assert (
        len(x.shape) == 3
    ), f"Expected input (batch, height, width) but got {x.shape}."
    return x.mean(dim=0)


class MeanMapAggregator:
    """
    Aggregates time mean maps of target and prediction tensors, time mean
    bias, and a log10-scaled relative mean bias to the target data.

    Args:
        variable_metadata: metadata for the variables.
        gap_width: Width between the prediction and target images.
        name: The name to use for the maps in the wandb output.
    """

    def __init__(
        self,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        gap_width: int = 4,
        name: str = "",
    ):
        self.gap_width = gap_width
        if variable_metadata is None:
            self._variable_metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._variable_metadata = variable_metadata
        self._name = _ensure_trailing_slash(name)

        self._mean_target = Mean(batch_mean)
        self._mean_prediction = Mean(batch_mean)

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping) -> None:
        """
        Record the running average maps for the current batch.

        Args:
            target: Tensor of shape (n_examples, n_lat, n_lon)
            prediction: Tensor of shape (n_examples, n_lat, n_lon)
        """
        assert set(target.keys()) == set(prediction.keys()), "Keys do not match"
        self._mean_target.record_batch(target)
        self._mean_prediction.record_batch(prediction)

    def _get(self) -> Tuple[TensorMapping, TensorMapping]:
        target = self._mean_target.get()
        prediction = self._mean_prediction.get()

        def get_relative_mean(target, prediction):
            return torch.log10(torch.abs(prediction / target))

        relative = map_tensor_mapping(get_relative_mean)(target, prediction)

        maps = {}
        metrics = {}
        for var_name in target.keys():
            gap = torch.full(
                (target[var_name].shape[-2], self.gap_width),
                float(target[var_name].min()),
                device=target[var_name].device,
            )
            maps[f"maps/{self._name}full-field/{var_name}"] = torch.cat(
                (prediction[var_name], gap, target[var_name]), dim=1
            )
            maps[f"maps/{self._name}log10_relative_mean/{var_name}"] = relative[
                var_name
            ]
            error = prediction[var_name] - target[var_name]
            maps[f"maps/{self._name}error/{var_name}"] = error
            metrics[f"metrics/{self._name}bias/{var_name}"] = error.mean()
        return metrics, maps

    _captions = {
        "full-field": (
            "{name}  mean full field; (left) generated and (right) target [{units}]"
        ),
        "error": "{name} mean full field error (generated - target) [{units}]",
        "log10_relative_mean": "{name} log10(prediction/target) of mean full field",
    }

    def _get_caption(self, key: str, name: str, vmin: float, vmax: float) -> str:
        if name in self._variable_metadata:
            caption_name = self._variable_metadata[name].long_name
            units = self._variable_metadata[name].units
        else:
            caption_name, units = name, "unknown_units"
        caption = self._captions[key].format(name=caption_name, units=units)
        if "error" in key:
            caption += f" vmin={vmin:.4g} (blue), vmax={vmax:.4g} (red)."
        else:
            caption += f" vmin={vmin:.4g}, vmax={vmax:.4g}."
        return caption

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get all time average map logs for wandb.
        """
        prefix = _ensure_trailing_slash(prefix)
        ret = {}
        wandb = WandB.get_instance()
        metrics, maps = self._get()
        ret.update({f"{prefix}{k}": v.cpu().numpy() for k, v in metrics.items()})
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


class RelativeMSEInterpAggregator:
    """
    Records the running relative mean squared error (MSE) of the prediction and the
    interpolated coarse field.

    Args:
        downscale_factor: The factor by which the data has been downscaled.
        name: The name to use for the metric in the wandb output.
    """

    def __init__(self, downscale_factor: int, name: str = "") -> None:
        self.downscale_factor = downscale_factor
        self._prediction_mse = MeanComparison(torch.nn.MSELoss())
        self._interpolated_mse = MeanComparison(torch.nn.MSELoss())
        self._name = _ensure_trailing_slash(name)

    def _interpolate(
        self, data: TensorMapping, keys: Optional[Collection[str]] = None
    ) -> TensorMapping:
        channel_dim = -3
        keys = keys if keys is not None else data.keys()
        return {
            k: interpolate(v.unsqueeze(channel_dim), self.downscale_factor).squeeze(
                channel_dim
            )
            for k, v in data.items()
            if k in keys
        }

    @torch.no_grad()
    def record_batch(
        self, target: TensorMapping, prediction: TensorMapping, coarse: TensorMapping
    ) -> None:
        """
        Record the relative MSE for the current batch.
        """
        self._prediction_mse.record_batch(target, prediction)
        self._interpolated_mse.record_batch(
            target, self._interpolate(coarse, keys=target.keys())
        )

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the relative MSE logs for wandb.
        """
        prefix = _ensure_trailing_slash(prefix)
        prediction_mse = self._prediction_mse.get()
        interpolated_mse = self._interpolated_mse.get()
        return {
            f"{prefix}{self._name}{k}": prediction_mse[k] / interpolated_mse[k]
            for k in prediction_mse
        }


class Aggregator:
    """
    Class for aggregating evaluation for training/validation batch predictions
    for 3D data. E.g., the case in training/validation where there is no sample
    dimension, or for folded generation data where the batch and samples are
    folded into the first dimension.

    For generation metrics, where the leading dimensions are [batch, sample],
    use GenerationAggregator instead.

    Args:
        dims: Spatial dimensions of the data (e.g. [lat, lon]). Used by the
            CoreSnapshotAggregator to output a dataset of map metrics.
        downscale_factor: the scale factor for going from coarse to fine data.
        n_histogram_bins: Number of bins for the histogram.
        percentiles: List of percentiles to compute for the histogram.
        ssim_kwargs: Keyword arguments for SSIM computation.
        variable_metadata: Metadata for each variable.
        include_positional_comparisons: Include comparison metrics that output
            positional maps (e.g. mean maps).  This should be disabled when
            using random subsetting because averaged maps won't make sense.
    """

    def __init__(
        self,
        dims: List[str],
        downscale_factor: int,
        n_histogram_bins: int = 300,
        percentiles: Optional[List[float]] = None,
        ssim_kwargs: Optional[Mapping[str, Any]] = None,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        include_positional_comparisons: bool = True,
    ) -> None:
        self.downscale_factor = downscale_factor

        if ssim_kwargs is None:
            ssim_kwargs = {}

        # Folded samples into batch dimension
        self._comparisons: List[_ComparisonAggregator] = [
            MeanComparison(metrics.root_mean_squared_error, name="metrics/rmse"),
            SnapshotAggregator(dims, variable_metadata, name="snapshot"),
            ComparedDynamicHistogramsAdapter(
                histograms=ComparedDynamicHistograms(
                    n_bins=n_histogram_bins, percentiles=percentiles
                ),
                name="histogram",
            ),
        ]
        self._area_weighted: List[_DynamicMetricComparisonAggregator] = [
            MeanComparison(name="metrics/area_weighted_rmse"),
        ]

        # Includes coarse input ontop of target/prediction
        self._coarse_comparisons: List[_CoarseComparisonAggregator] = [
            RelativeMSEInterpAggregator(
                downscale_factor, name="metrics/relative_mse_bicubic"
            ),
            ZonalPowerSpectrumComparison(downscale_factor, name="power_spectrum"),
        ]

        # Turned off for random subsetting where average maps don't make sense
        if include_positional_comparisons:
            self._comparisons += [
                MeanMapAggregator(variable_metadata, name="time_mean"),
            ]

        self.loss = Mean(torch.mean)

    @torch.no_grad()
    def record_batch(
        self,
        outputs: ModelOutputs,
        coarse: TensorMapping,
        batch: PairedBatchData,
    ) -> None:
        """
        Records a batch of target and prediction tensors for metric computation.
        Expects (batch, ...) shaped tensors.

        Args:
            outputs: The model outputs from a downscaling prediction step.
            coarse: The coarse data for the current batch.
            batch: Paired batch data with spatial information.
        """
        _check_batch_dims_for_recording(outputs, coarse, 3)
        target, prediction = outputs.target, outputs.prediction
        target = filter_tensor_mapping(target, prediction.keys())

        for comparison_aggregator in self._comparisons:
            comparison_aggregator.record_batch(target, prediction)

        for weighted_aggregator in self._area_weighted:
            area_weights = batch.fine.latlon_coordinates.area_weights.to(get_device())

            def weighted_rmse(truth, pred):
                return metrics.root_mean_squared_error(truth, pred, area_weights)

            weighted_aggregator.record_batch(
                target, prediction, dynamic_metric=weighted_rmse
            )

        for coarse_comparison_aggregator in self._coarse_comparisons:
            coarse_comparison_aggregator.record_batch(target, prediction, coarse)

        self.loss.record_batch({"loss": outputs.loss})

    def get_wandb(
        self,
        prefix: str = "",
    ) -> Mapping[str, Any]:
        """
        Get the wandb output to log from all sub aggregators.
        """
        prefix = _ensure_trailing_slash(prefix)

        ret: Dict[str, Any] = {}
        ret.update(self.loss.get_wandb(prefix))
        for comparison in self._comparisons:
            ret.update(comparison.get_wandb(prefix))
        for coarse_comparison in self._coarse_comparisons:
            ret.update(coarse_comparison.get_wandb(prefix))

        return ret
