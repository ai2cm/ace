"""Contains classes for aggregating inference metrics and various statistics."""

from typing import Any, Callable, Dict, Literal, Mapping, Optional, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

import fme.core.histogram
from fme.core import metrics
from fme.core.aggregator.plotting import plot_imshow
from fme.core.typing_ import TensorMapping
from fme.core.wandb import WandB
from fme.downscaling.metrics_and_maths import (
    compute_psnr,
    compute_ssim,
    compute_zonal_power_spectrum,
    map_tensor_mapping,
)


def _detach_and_to_cpu(x: TensorMapping) -> TensorMapping:
    return {k: v.detach().cpu() for k, v in x.items()}


class Mean:
    """
    Tracks a running average of a metric over multiple batches.

    Args:
        metric: The metric function to be calculated.

    Raises:
        ValueError: If no values have been added to the running average.
    """

    def __init__(
        self,
        metric: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._mapped_metric = map_tensor_mapping(metric)
        self._sum: Optional[TensorMapping] = None
        self._count: int = 0
        self._add = map_tensor_mapping(torch.add)

    def record_batch(self, data: TensorMapping) -> None:
        """
        Records the metric values of a batch.

        Args:
        """
        data = _detach_and_to_cpu(data)
        metric = self._mapped_metric(data)

        if self._sum is None:
            self._sum = {k: torch.zeros_like(v) for k, v in metric.items()}

        self._sum = self._add(self._sum, metric)
        self._count += 1

    def get(self) -> TensorMapping:
        """
        Calculates and returns the current mean of the metric values.

        Returns:
            TensorMapping corresponding to the current value of the metric on
            each variable.

        Raises:
            ValueError: If no values have been added to the running average.
        """
        if self._sum is None:
            raise ValueError("No values have been added to the running average")
        return {k: self._sum[k] / self._count for k in self._sum}

    def get_wandb(self) -> Mapping[str, Any]:
        return {k: v.numpy() for k, v in self.get().items()}


class MeanComparison:
    """
    Tracks a running average of a metric over multiple batches.

    Args:
        metric: The metric function to be calculated which compares two tensors
        (e.g. target and prediction)

    Raises:
        ValueError: If no values have been added to the running average.
    """

    def __init__(
        self,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self._mapped_metric = map_tensor_mapping(metric)
        self._sum: Optional[TensorMapping] = None
        self._count: int = 0
        self._add = map_tensor_mapping(torch.add)

    def record_batch(self, target: TensorMapping, prediction: TensorMapping) -> None:
        """
        Records the metric values of a batch.

        Args:
            *values: The metric values of the batch. Arguments could be for
                example, truth and prediction data, and should correspond to the
                `torch.Tensor` arguments of the `metric` used to initialize this
                object.
        """
        target = _detach_and_to_cpu(target)
        prediction = _detach_and_to_cpu(prediction)
        metric = self._mapped_metric(target, prediction)

        if self._sum is None:
            self._sum = {k: torch.zeros_like(v) for k, v in metric.items()}

        self._sum = self._add(self._sum, metric)
        self._count += 1

    def get(self) -> TensorMapping:
        """
        Calculates and returns the current mean of the metric values.

        Returns:
            TensorMapping corresponding to the current value of the metric on
            each variable.

        Raises:
            ValueError: If no values have been added to the running average.
        """
        if self._sum is None:
            raise ValueError("No values have been added to the running average")
        return {k: self._sum[k] / self._count for k in self._sum}

    def get_wandb(self) -> Mapping[str, Any]:
        return {k: v.numpy() for k, v in self.get().items()}


class ZonalPowerSpectrum:
    def __init__(self, latitudes: torch.Tensor) -> None:
        def _compute_zonal_power_spectrum(x):
            assert (
                len(x.shape) == 3
            ), f"Expected input (batch, height, width) but received {x.shape}"
            return compute_zonal_power_spectrum(x, latitudes).mean(  # type: ignore
                axis=-2
            )  # (batch, wavenumber) -> (wavenumber,)

        self._mean_aggregator = Mean(_compute_zonal_power_spectrum)

    @torch.no_grad()
    def record_batch(self, data: TensorMapping) -> None:
        self._mean_aggregator.record_batch(data)

    def get(self) -> TensorMapping:
        return self._mean_aggregator.get()

    def _plot_spectrum(self, spectrum: np.ndarray) -> matplotlib.figure.Figure:
        fig = plt.figure()
        plt.loglog(spectrum)
        plt.grid()
        return fig

    def get_wandb(self) -> Mapping[str, Any]:
        aggregated = self._mean_aggregator.get_wandb()
        ret = {}
        for name, values in aggregated.items():
            ret[name] = self._plot_spectrum(values)
        return ret


class Snapshot:
    """
    A class for creating and retrieving snapshots of values. Stores the first
    snapshot of values recorded

    Attributes:
        snapshot: The snapshot of values.
    """

    def __init__(self) -> None:
        self.snapshot: Optional[TensorMapping] = None

    @torch.no_grad()
    def record_batch(self, data: TensorMapping) -> None:
        """Creates a snapshot if one has not yet been set."""
        if self.snapshot is None:
            self.snapshot = _detach_and_to_cpu(data)

    def get(self) -> TensorMapping:
        """
        Returns the snapshot.

        Raises:
            ValueError: If no values have been added to the snapshot.
        """
        if self.snapshot is None:
            raise ValueError("No values have been added to the snapshot")
        return self.snapshot

    def get_wandb(self) -> Mapping[str, Any]:
        return {
            k: plot_imshow(v.squeeze(dim=-3)[0].numpy()) for k, v in self.get().items()
        }


class DynamicHistogram:
    """Wrapper of DynamicHistogram for multiple histograms, one per variable."""

    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins
        self.histograms: Optional[
            Mapping[str, fme.core.histogram.DynamicHistogram]
        ] = None
        self._time_dim = -2

    @torch.no_grad()
    def record_batch(self, data: TensorMapping):
        data = _detach_and_to_cpu(data)
        if self.histograms is None:
            self.histograms = {
                k: fme.core.histogram.DynamicHistogram(1, n_bins=self.n_bins)
                for k in data
            }

        for k, v in data.items():
            assert (
                len(v.shape) == 3
            ), f"Expected input (batch, height, width) but got {v.shape}"
            self.histograms[k].add(v.flatten().unsqueeze(0).numpy())

    def get(self):
        if self.histograms is None:
            raise ValueError("No data has been added to the histogram")
        return {
            k: (v.counts.squeeze(self._time_dim), v.bin_edges)
            for k, v in self.histograms.items()
        }

    def get_wandb(self) -> Mapping[str, Any]:
        wandb = WandB.get_instance()
        return {k: wandb.Histogram(np_histogram=v) for k, v in self.get().items()}


class Aggregator:
    """
    Class for aggregating inference metrics and intrinsic statistics.

    Args:
        area_weights: Tensor of area weights.
        latitudes: Tensor of latitudes.
        ssim_kwargs: Optional keyword arguments for SSIM computation.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        latitudes: torch.Tensor,
        n_histogram_bins: int = 300,
        ssim_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        def _area_weighted_rmse(truth, pred):
            return metrics.root_mean_squared_error(truth, pred, area_weights)

        if ssim_kwargs is None:
            ssim_kwargs = {}

        def _compute_ssim(x, y):
            return compute_ssim(x, y, add_channel_dim=True, **ssim_kwargs)

        def _compute_psnr(x, y):
            return compute_psnr(x, y, add_channel_dim=True)

        self._comparisons = {
            "rmse": MeanComparison(metrics.root_mean_squared_error),
            "weighted_rmse": MeanComparison(_area_weighted_rmse),
            "ssim": MeanComparison(_compute_ssim),
            "psnr": MeanComparison(_compute_psnr),
        }
        self._intrinsics: Mapping[
            str, Mapping[str, Union[DynamicHistogram, Snapshot, ZonalPowerSpectrum]]
        ] = {
            input_type: {
                "histogram": DynamicHistogram(n_bins=n_histogram_bins),
                "snapshot": Snapshot(),
                "spectrum": ZonalPowerSpectrum(latitudes),
            }
            for input_type in ("target", "prediction")
        }
        self.loss = Mean(torch.mean)

    @torch.no_grad()
    def record_batch(
        self, loss: torch.Tensor, target: TensorMapping, prediction: TensorMapping
    ) -> None:
        """
        Records a batch of target and prediction tensors for metric computation.

        Args:
            target: Ground truth
            pred: Model outputs
        """
        for _, comparison_aggregator in self._comparisons.items():
            comparison_aggregator.record_batch(target, prediction)

        for input, input_type in zip((target, prediction), ("target", "prediction")):
            for _, intrinsic_aggregator in self._intrinsics[input_type].items():
                intrinsic_aggregator.record_batch(input)

        self.loss.record_batch({"loss": loss})

    def _get(
        self, getter: Literal["get", "get_wandb"], prefix: str = ""
    ) -> Mapping[str, Any]:
        """Helper methods for accumulating all metrics into a single mapping
        with consistent keys. Values depend on the specified `getter`
        function."""

        if prefix != "":
            prefix += "/"

        ret: Dict[str, Any] = {f"{prefix}loss": self.loss.get()["loss"]}
        for metric_name, agg in self._comparisons.items():
            ret.update(
                {
                    f"{prefix}{metric_name}/{k}": v
                    for (k, v) in getattr(agg, getter)().items()
                }
            )

        for data_role in ("target", "prediction"):
            _aggregators = self._intrinsics[data_role]
            for metric_name, agg in _aggregators.items():  # type: ignore
                _metric_values = {
                    f"{prefix}{metric_name}/{var_name}_{data_role}": v
                    for (var_name, v) in getattr(agg, getter)().items()
                }
                ret.update(_metric_values)

        return ret

    def get(self, prefix: str = "") -> Mapping[str, torch.Tensor]:
        """
        Returns a mapping of aggregated metrics and statistics.

        Returns:
            Mapping of aggregated metrics and statistics with keys corresponding
            to the metric and variable name (e.g. for passing to wandb).
        """
        return self._get(getter="get", prefix=prefix)

    def get_wandb(
        self,
        prefix: str = "",
    ) -> Mapping[str, Any]:
        return self._get(getter="get_wandb", prefix=prefix)


class NullInferenceAggregator:
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        del args, kwargs

    def record_batch(
        self, loss: torch.Tensor, target: TensorMapping, prediction: TensorMapping
    ) -> None:
        del loss, target, prediction
