"""Contains classes for aggregating evaluation metrics and various statistics."""

from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from fme.core import metrics
from fme.core.aggregator.one_step.snapshot import (
    SnapshotAggregator as CoreSnapshotAggregator,
)
from fme.core.aggregator.plotting import get_cmap_limits, plot_imshow
from fme.core.data_loading.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.histogram import ComparedDynamicHistograms
from fme.core.typing_ import TensorMapping
from fme.core.wandb import WandB
from fme.downscaling.metrics_and_maths import (
    compute_crps,
    compute_mae_error,
    compute_zonal_power_spectrum,
    interpolate,
    map_tensor_mapping,
)
from fme.downscaling.models import ModelOutputs


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
            data: the data to record.
        """
        data = _detach_and_to_cpu(data)
        metric = self._mapped_metric(data)

        if self._sum is None:
            self._sum = {
                k: torch.zeros_like(v, dtype=torch.float64) for k, v in metric.items()
            }

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
        return {k: v.detach().cpu().numpy() for k, v in self.get().items()}


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
            target: the 'truth' target values.
            prediction: the predicted values.
        """
        target = _detach_and_to_cpu(target)
        prediction = _detach_and_to_cpu(prediction)
        metric = self._mapped_metric(target, prediction)

        if self._sum is None:
            self._sum = {
                k: torch.zeros_like(v, dtype=torch.float64) for k, v in metric.items()
            }

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
        return {k: v.detach().cpu().numpy() for k, v in self.get().items()}


class ZonalPowerSpectrum:
    def __init__(self, latitudes: torch.Tensor) -> None:
        self._wandb = WandB.get_instance()

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

    def _plot_spectrum(self, spectrum: np.ndarray) -> Any:
        fig = plt.figure()
        plt.loglog(spectrum)
        plt.grid()
        plt.close(fig)
        return fig

    def get_wandb(self) -> Mapping[str, Any]:
        aggregated = self._mean_aggregator.get_wandb()
        ret = {}
        for name, values in aggregated.items():
            ret[name] = self._plot_spectrum(values)
        return ret


class SnapshotAggregator:
    def __init__(
        self, variable_metadata: Optional[Mapping[str, VariableMetadata]]
    ) -> None:
        self._snapshot_aggregator = CoreSnapshotAggregator(variable_metadata)

    def _tile_time_dim(self, x: torch.Tensor) -> torch.Tensor:
        time_dim = -3
        return x.unsqueeze(time_dim).repeat_interleave(2, dim=time_dim)

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping) -> None:
        # core SnapshotAggregator expects a time dimension of length two, so we
        # provide it by tiling the data.
        target = map_tensor_mapping(self._tile_time_dim)(target)
        prediction = map_tensor_mapping(self._tile_time_dim)(prediction)
        self._snapshot_aggregator.record_batch(-1.0, target, prediction, {}, {})

    def _remove_leading_slash(self, s: str) -> str:
        if s.startswith("/"):
            return s[1:]
        else:
            return s

    def get(self, label: str = "") -> Mapping[str, Any]:
        logs = self._snapshot_aggregator.get_logs(label)
        ret = {}
        for k, v in logs.items():
            # residual is meaningless for single steps
            if "residual" not in k:
                # The core SnapshotAggregator returns {label}/{key} even when
                # label == "". In this case, removing the leading slash.
                ret[self._remove_leading_slash(k)] = v
        return ret

    def get_wandb(self, label: str = "") -> Mapping[str, Any]:
        return self.get(label)


class MeanMapAggregator:
    """
    Aggregates time mean maps of target and prediction tensors, and time mean
    bias maps.

    Args:
        variable_metadata: metadata for the variables.
        gap_width: Width between the prediction and target images.
    """

    def __init__(
        self,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        gap_width: int = 4,
    ):
        self.gap_width = gap_width
        if variable_metadata is None:
            self._variable_metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._variable_metadata = variable_metadata

        def batch_mean(x: torch.Tensor) -> torch.Tensor:
            assert (
                len(x.shape) == 3
            ), f"Expected input (batch, height, width) but got {x.shape}."
            return x.mean(dim=0)

        self._mean_target = Mean(batch_mean)
        self._mean_prediction = Mean(batch_mean)

    def record_batch(self, target: TensorMapping, prediction: TensorMapping) -> None:
        """Updates the average of target, prediction, and bias time mean maps.

        Args:
            target: Tensor of shape (n_examples, n_lat, n_lon)
            prediction: Tensor of shape (n_examples, n_lat, n_lon)
        """
        assert set(target.keys()) == set(prediction.keys()), "Keys do not match"
        self._mean_target.record_batch(target)
        self._mean_prediction.record_batch(prediction)

    def get(self) -> TensorMapping:
        prediction = self._mean_prediction.get()
        target = self._mean_target.get()

        ret = {}
        for var_name in target.keys():
            gap = torch.full(
                (target[var_name].shape[-2], self.gap_width),
                float(target[var_name].min()),
            )
            ret[f"full-field/{var_name}"] = torch.cat(
                (prediction[var_name], gap, target[var_name]), dim=1
            )
            ret[f"error/{var_name}"] = prediction[var_name] - target[var_name]
        return ret

    _captions = {
        "full-field": (
            "{name} one step mean full field; "
            "(left) generated and (right) target [{units}]"
        ),
        "error": "{name} one step mean full field error (generated - target) [{units}]",
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

    def get_wandb(self) -> Mapping[str, Any]:
        ret = {}
        wandb = WandB.get_instance()
        for key, data in self.get().items():
            if "error" in key:
                diverging, cmap = True, "RdBu_r"
            else:
                diverging, cmap = False, None
            data = data.cpu().numpy()
            vmin, vmax = get_cmap_limits(data, diverging=diverging)
            map_name, var_name = key.split("/")
            caption = self._get_caption(map_name, var_name, vmin, vmax)
            fig = plot_imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
            ret[key] = wandb.Image(fig, caption=caption)
            plt.close(fig)
        return ret

    def get_dataset(self) -> xr.Dataset:
        prediction = self._mean_prediction.get()
        target = self._mean_target.get()
        dataset: Dict[str, xr.DataArray] = {}
        dims = ("source", "lat", "lon")
        coords = {
            "source": ["target", "prediction"],
        }
        for var_name in target:
            metadata = self._variable_metadata.get(
                var_name, VariableMetadata("unknown_units", var_name)
            )._asdict()
            datum = torch.stack(
                (
                    target[var_name],
                    prediction[var_name],
                ),
                dim=0,
            )
            dataset[var_name] = xr.DataArray(
                datum,
                dims=dims,
                attrs=metadata,
            )
        return xr.Dataset(dataset, coords=coords)


class RelativeMSEInterpAggregator:
    def __init__(self, downscale_factor: int) -> None:
        self.downscale_factor = downscale_factor
        self._prediction_mse = MeanComparison(torch.nn.MSELoss())
        self._interpolated_mse = MeanComparison(torch.nn.MSELoss())

    def _interpolate(self, tensor: torch.Tensor, downscale_factor: int):
        channel_dim = -3
        return interpolate(tensor.unsqueeze(channel_dim), downscale_factor).squeeze(
            channel_dim
        )

    def record_batch(
        self, target: TensorMapping, prediction: TensorMapping, coarse: TensorMapping
    ) -> None:
        self._prediction_mse.record_batch(target, prediction)
        interpolated = {
            k: self._interpolate(v, self.downscale_factor) for k, v in coarse.items()
        }
        self._interpolated_mse.record_batch(target, interpolated)

    def get_wandb(self) -> Mapping[str, Any]:
        prediction_mse = self._prediction_mse.get()
        interpolated_mse = self._interpolated_mse.get()
        return {k: prediction_mse[k] / interpolated_mse[k] for k in prediction_mse}


class LatentStepAggregator:
    def __init__(self) -> None:
        self._latent_steps: List[torch.Tensor] = []

    def _tile_time_dim(self, x: torch.Tensor) -> torch.Tensor:
        time_dim = -3
        return x.unsqueeze(time_dim).repeat_interleave(2, dim=time_dim)

    @torch.no_grad()
    def record_batch(self, latent_steps: List[torch.Tensor]) -> None:
        self._latent_steps = latent_steps

    def get(self) -> List[Any]:
        if len(self._latent_steps) == 0:
            return []

        wandb = WandB.get_instance()
        images = []
        latent_shape = self._latent_steps[0].shape
        output_channels = latent_shape[-3]
        for i, im in enumerate(self._latent_steps):
            for j in range(output_channels):
                image = im[0, j].cpu().numpy()  # select 1st example
                fig = plot_imshow(image, use_colorbar=False)
                images.append(
                    wandb.Image(fig, caption=f"latent step {i}, out channel {j}")
                )
                plt.close(fig)
        return images

    def get_wandb(self) -> List[Any]:
        return self.get()


class _ComparisonAggregator(Protocol):
    def record_batch(
        self, target: TensorMapping, prediction: TensorMapping
    ) -> None: ...

    def get_wandb(self) -> Mapping[str, Any]: ...


def _fold_sample_dim(
    truth: TensorMapping,
    pred: TensorMapping,
    coarse: TensorMapping,
) -> Tuple[TensorMapping, TensorMapping, TensorMapping]:
    """
    Takes truth and coarse data with only a [batch, ...] dimension and predictions
    with [batch, samples, ...] dimensions, and returns dictionaries
    which each have a combined [batch * samples, ...] dimension.

    This is used to pass data to aggregators written to expect a single
    batch dimension, or which don't care whether two values come
    from the same input or not.

    Args:
        truth: Dictionary of truth data, with only a batch dimension
        pred: Dictionary of prediction data, with a batch and sample dimension
        coarse: Dictionary of coarse data, with only a batch dimension

    Returns:
        Tuple of dictionaries with the same keys as the input dictionaries
        but each with a combined [batch * samples, ...] dimension.
    """
    return_truth = {}
    return_pred = {}
    return_coarse = {}
    for key, pred_value in pred.items():
        samples = pred_value.shape[1]
        if len(pred_value.shape) != len(truth[key].shape) + 1:
            raise ValueError(
                "expected pred to have a sample dimension, "
                f"has shape {pred_value.shape} "
                f"with truth shape {truth[key].shape}"
            )
        return_truth[key] = (
            truth[key]
            .unsqueeze(1)
            .repeat_interleave(samples, dim=1)
            .reshape(truth[key].shape[0] * samples, *truth[key].shape[1:])
        )
        return_coarse[key] = (
            coarse[key]
            .unsqueeze(1)
            .repeat_interleave(samples, dim=1)
            .reshape(coarse[key].shape[0] * samples, *coarse[key].shape[1:])
        )
        return_pred[key] = pred_value.reshape(
            pred_value.shape[0] * samples, *pred_value.shape[2:]
        )
    return return_truth, return_pred, return_coarse


class Aggregator:
    """
    Class for aggregating evaluation metrics and intrinsic statistics.

    Args:
        area_weights: Tensor of area weights.
        latitudes: Tensor of latitudes.
        downscale_factor: Downscaling factor.
        n_histogram_bins: Number of bins for histogram comparisons.
        percentiles: Percentiles for histogram comparisons.
        ssim_kwargs: Optional keyword arguments for SSIM computation.
        variable_metadata: Metadata for each variable.
    """

    def __init__(
        self,
        area_weights: Optional[torch.Tensor],
        latitudes: torch.Tensor,
        downscale_factor: int,
        n_histogram_bins: int = 300,
        percentiles: Optional[List[float]] = None,
        ssim_kwargs: Optional[Mapping[str, Any]] = None,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ) -> None:
        self.downscale_factor = downscale_factor
        if area_weights is not None:
            area_weights = area_weights.to(get_device())

        def _area_weighted_rmse(truth, pred):
            return metrics.root_mean_squared_error(
                truth.to(get_device()), pred.to(get_device()), area_weights
            )

        if ssim_kwargs is None:
            ssim_kwargs = {}

        self._comparisons: Mapping[str, _ComparisonAggregator] = {
            "rmse": MeanComparison(metrics.root_mean_squared_error),
            "weighted_rmse": MeanComparison(_area_weighted_rmse),
            "snapshot": SnapshotAggregator(variable_metadata),
            "time_mean_map": MeanMapAggregator(variable_metadata),
            "histogram": ComparedDynamicHistograms(
                n_bins=n_histogram_bins, percentiles=percentiles
            ),
        }
        self._intrinsics: Mapping[
            str, Mapping[str, Union[ComparedDynamicHistograms, ZonalPowerSpectrum]]
        ] = {
            input_type: {
                "spectrum": ZonalPowerSpectrum(latitudes),
            }
            for input_type in ("target", "prediction")
        }
        self._coarse_comparisons = {
            "relative_mse_bicubic": RelativeMSEInterpAggregator(downscale_factor)
        }
        self._probabilistic_comparisons = {
            "crps": MeanComparison(compute_crps),
            "mae_error": MeanComparison(compute_mae_error),
        }
        self._latent_step_aggregator = LatentStepAggregator()
        self.loss = Mean(torch.mean)

    @torch.no_grad()
    def record_batch(
        self,
        outputs: ModelOutputs,
        coarse: TensorMapping,
    ) -> None:
        """
        Records a batch of target and prediction tensors for metric computation.

        Args:
            outputs: the model predictions and target data.
            coarse: the coarse data used as input for downscaling.
        """
        for _, prob_comparison_aggregator in self._probabilistic_comparisons.items():
            prob_comparison_aggregator.record_batch(outputs.target, outputs.prediction)

        folded_target, folded_prediction, folded_coarse = _fold_sample_dim(
            outputs.target, outputs.prediction, coarse
        )
        for _, comparison_aggregator in self._comparisons.items():
            comparison_aggregator.record_batch(folded_target, folded_prediction)

        for _, coarse_comparison_aggregator in self._coarse_comparisons.items():
            coarse_comparison_aggregator.record_batch(
                folded_target, folded_prediction, folded_coarse
            )

        for input, input_type in zip(
            (folded_target, folded_prediction), ("target", "prediction")
        ):
            for _, intrinsic_aggregator in self._intrinsics[input_type].items():
                intrinsic_aggregator.record_batch(input)

        self._latent_step_aggregator.record_batch(outputs.latent_steps)

        self.loss.record_batch({"loss": outputs.loss})

    @torch.no_grad()
    def get_wandb(
        self,
        prefix: str = "",
    ) -> Mapping[str, Any]:
        if prefix != "":
            prefix += "/"

        ret: Dict[str, Any] = {f"{prefix}loss": self.loss.get()["loss"]}
        for metric_name, comparison_agg in self._comparisons.items():
            ret.update(
                {
                    f"{prefix}{metric_name}/{k}": v
                    for (k, v) in comparison_agg.get_wandb().items()
                }
            )

        for metric_name, relative_agg in self._coarse_comparisons.items():
            ret.update(
                {
                    f"{prefix}{metric_name}/{k}": v
                    for (k, v) in relative_agg.get_wandb().items()
                }
            )

        for data_role in ("target", "prediction"):
            _aggregators = self._intrinsics[data_role]
            for metric_name, intrinsic_agg in _aggregators.items():
                _metric_values = {
                    f"{prefix}{metric_name}/{var_name}_{data_role}": v
                    for (var_name, v) in intrinsic_agg.get_wandb().items()
                }
                ret.update(_metric_values)

        latent_steps = self._latent_step_aggregator.get_wandb()

        if len(latent_steps) > 0:
            ret[f"{prefix}snapshot/latent_steps"] = latent_steps

        return ret

    @torch.no_grad()
    def get_datasets(self) -> Mapping[str, xr.Dataset]:
        datasets = {
            # get_dataset is not yet implemented for all _ComparisonAggregator
            k: self._comparisons[k].get_dataset()  # type: ignore
            for k in ("time_mean_map", "histogram")
        }
        return datasets
