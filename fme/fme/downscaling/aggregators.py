"""Contains classes for aggregating inference metrics and various statistics."""

from typing import Any, Callable, Literal, Mapping, Optional, Union

import numpy as np
import torch
import wandb

import fme.core.histogram
from fme.core import metrics
from fme.core.aggregator.plotting import plot_imshow
from fme.core.typing_ import TensorMapping
from fme.downscaling.metrics_and_maths import (
    compute_psnr,
    compute_ssim,
    compute_zonal_power_spectrum,
    map_tensor_mapping,
)


class Mean:
    """
    Tracks a running average of a metric over multiple batches.

    Args:
        metric: The metric function to be calculated.

    Raises:
        ValueError: If no values have been added to the running average.
    """

    def __init__(self, metric: Callable[..., torch.Tensor]) -> None:
        self._mapped_metric = map_tensor_mapping(metric)
        self._sum: Optional[TensorMapping] = None
        self._count: int = 0
        self._add = map_tensor_mapping(torch.add)

    def record_batch(self, *values: TensorMapping) -> None:
        """
        Records the metric values of a batch.

        Args:
            *values: The metric values of the batch. Arguments could be for
                example, truth and prediction data, and should correspond to the
                `torch.Tensor` arguments of the `metric` used to initialize this
                object.
        """
        metric = self._mapped_metric(*values)

        if self._sum is None:
            self._sum = {k: torch.zeros_like(v) for k, v in metric.items()}

        self._sum = self._add(self._sum, self._mapped_metric(*values))
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

    def get_wandb(self) -> Mapping[str, np.ndarray]:
        return {k: v.numpy() for k, v in self.get().items()}


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
    def record_batch(self, values: TensorMapping) -> None:
        """Creates a snapshot if one has not yet been set."""
        if self.snapshot is None:
            self.snapshot = values

    def get(self) -> TensorMapping:
        """
        Returns the snapshot.

        Raises:
            ValueError: If no values have been added to the snapshot.
        """
        if self.snapshot is None:
            raise ValueError("No values have been added to the snapshot")
        return self.snapshot

    def get_wandb(self) -> Mapping[str, wandb.Image]:
        return {
            k: plot_imshow(v.squeeze(dim=-3)[0].numpy()) for k, v in self.get().items()
        }


class DynamicHistogram:
    """Wrapper of DynamicHistogram for multiple histograms, one per variable.

    Assumes that the time dimension is of shape (1,) since this is the case for
    downscaling.
    """

    def __init__(self, n_bins) -> None:
        self.n_bins = n_bins
        self.histograms: Optional[
            Mapping[str, fme.core.histogram.DynamicHistogram]
        ] = None

    @torch.no_grad()
    def record_batch(self, data: TensorMapping):
        if self.histograms is None:
            self.histograms = {
                k: fme.core.histogram.DynamicHistogram(1, n_bins=self.n_bins)
                for k in data
            }

        for k, v in data.items():
            self.histograms[k].add(v.flatten().unsqueeze(0).numpy())

    def get(self):
        if self.histograms is None:
            raise ValueError("No data has been added to the histogram")
        return self.histograms

    def get_wandb(self) -> Mapping[str, wandb.Histogram]:
        return {
            k: wandb.Histogram(np_histogram=(v.counts.squeeze(axis=0), v.bin_edges))
            for k, v in self.get().items()
        }


class InferenceAggregator:
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

        def _compute_zonal_power_spectrum(x):
            return compute_zonal_power_spectrum(x, latitudes).mean(
                axis=-3
            )  # output is [batch, time, wavenumber] so this is a batch mean

        if ssim_kwargs is None:
            ssim_kwargs = {}

        def _compute_ssim(x, y):
            return compute_ssim(x, y, **ssim_kwargs)

        self._comparisons = {
            "rmse": Mean(metrics.root_mean_squared_error),
            "weighted_rmse": Mean(_area_weighted_rmse),
            "psnr": Mean(compute_psnr),
            "ssim": Mean(_compute_ssim),
        }
        self._intrinsics = {
            input_type: {
                "histogram": DynamicHistogram(n_bins=n_histogram_bins),
                "snapshot": Snapshot(),
                "spectrum": Mean(_compute_zonal_power_spectrum),
            }
            for input_type in ("target", "pred")
        }
        self.loss = Mean(torch.mean)

    @torch.no_grad()
    def record_batch(
        self, loss: torch.Tensor, target: TensorMapping, pred: TensorMapping
    ) -> None:
        """
        Records a batch of target and prediction tensors for metric computation.

        Args:
            target: Ground truth
            pred: Model outputs
        """
        for _, agg in self._comparisons.items():
            agg.record_batch(target, pred)

        for input, input_type in zip((target, pred), ("target", "pred")):
            for _, agg in self._intrinsics[input_type].items():  # type: ignore
                agg.record_batch(input)

        self.loss.record_batch({"loss": loss})

    def _get(self, getter: Literal["get", "get_wandb"]) -> Mapping[str, Any]:
        """Helper methods for accumulating all metrics into a single mapping
        with consistent keys. Values depend on the specified `getter`
        function."""
        ret = {}
        for metric_name, agg in self._comparisons.items():
            ret.update(
                {f"{metric_name}/{k}": v for (k, v) in getattr(agg, getter)().items()}
            )

        for input_type, aggs in self._intrinsics.items():
            for metric_name, agg in aggs.items():  # type: ignore
                ret.update(
                    {
                        f"{metric_name}/{var_name}_{input_type}": v
                        for (var_name, v) in getattr(agg, getter)().items()
                    }
                )

        return ret

    def get(self) -> Mapping[str, torch.Tensor]:
        """
        Returns a mapping of aggregated metrics and statistics.

        Returns:
            Mapping of aggregated metrics and statistics with keys corresponding
            to the metric and variable name (e.g. for passing to wandb).
        """
        return self._get(getter="get")

    def get_wandb(
        self,
    ) -> Mapping[str, Union[float, np.ndarray, wandb.Histogram, wandb.Image]]:
        return self._get(getter="get_wandb")
