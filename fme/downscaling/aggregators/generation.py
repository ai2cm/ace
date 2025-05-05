from collections.abc import Collection, Mapping
from typing import Any

import matplotlib.pyplot as plt
import torch
import xarray as xr

from fme.ace.aggregator.plotting import get_cmap_limits, plot_imshow
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.typing_ import TensorMapping
from fme.core.wandb import WandB

from ..datasets import PairedBatchData
from ..metrics_and_maths import compute_crps, filter_tensor_mapping
from ..models import ModelOutputs
from .main import (
    Aggregator,
    MeanComparison,
    SumComparison,
    ensure_trailing_slash,
    interpolate,
)
from .shape_helpers import _check_batch_dims_for_recording, _fold_sample_dim
from .typing import _CoarseComparisonAggregator


def _batch_mean_crps(
    target: TensorMapping,
    prediction: TensorMapping,
    batch_dim: int = 0,
    sample_dim: int = 1,
) -> torch.Tensor:
    return compute_crps(target, prediction, sample_dim=sample_dim).mean(dim=batch_dim)


def _batch_mean_mae_loss(
    target, prediction, batch_dim=0, sample_dim: int = 1
) -> torch.Tensor:
    mae_loss = torch.nn.functional.l1_loss(prediction, target, reduction="none")
    return torch.mean(mae_loss, dim=(batch_dim, sample_dim), keepdim=False)


def _get_map_caption(key: str, data: torch.Tensor) -> str:
    avg = data.mean()
    vmin, vmax = data.min(), data.max()
    caption = f"{key},  mean: {avg:.4g}, min: {vmin:.4g}, max: {vmax:.4g}"
    return caption


class RelativeCRPSInterpAggregator:
    """
    Records the running relative continuous ranked probability score (CRPS) of the
    prediction and the interpolated coarse field.

    Args:
        downscale_factor: The factor by which the data has been downscaled.
        name: The name to use for the metric in the wandb output.
    """

    def __init__(
        self,
        downscale_factor: int,
        name: str = "",
        include_positional_comparisons: bool = True,
    ) -> None:
        self.downscale_factor = downscale_factor
        self._prediction_crps = MeanComparison(_batch_mean_crps)
        self._interpolated_mae = MeanComparison(_batch_mean_mae_loss)
        self._name = ensure_trailing_slash(name)
        self._include_positional_comparisons = include_positional_comparisons

    def _interpolate(
        self, data: TensorMapping, keys: Collection[str] | None = None
    ) -> TensorMapping:
        # Expect CRPS coarse to have sample dimension, so no adjustment needed
        keys = keys if keys is not None else data.keys()
        return {
            k: interpolate(v, self.downscale_factor)
            for k, v in data.items()
            if k in keys
        }

    @torch.no_grad()
    def record_batch(
        self, target: TensorMapping, prediction: TensorMapping, coarse: TensorMapping
    ) -> None:
        """
        Record the relative CRPS for the current batch.
        """
        self._prediction_crps.record_batch(target, prediction)
        self._interpolated_mae.record_batch(
            target, self._interpolate(coarse, keys=target.keys())
        )

    def _get(self) -> Mapping[str, Any]:
        prediction_crps = self._prediction_crps.get()
        interpolated_mae = self._interpolated_mae.get()
        ret = {}
        for k, crps in prediction_crps.items():
            ret[f"{self._name}{k}"] = crps / interpolated_mae[k].squeeze()
        return ret

    def _plot_image(self, data: torch.Tensor, key: str) -> Any:
        wandb = WandB.get_instance()
        fig = plot_imshow(data, vmin=-1, vmax=1, cmap="RdBu_r", use_colorbar=True)
        caption = (
            "Relative CRPS (CRPS / MAE of interpolated coarse field) scale [-1, 1]"
            f" mean: {data.mean():.4g} min: {data.min():.4g}, max: {data.max():.4g}"
        )
        wandb_image = wandb.Image(fig, caption=caption)
        plt.close(fig)
        return wandb_image

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the relative CRPS logs for wandb.
        """
        prefix = ensure_trailing_slash(prefix)
        ret = {}
        for k, v in self._get().items():
            v = v.cpu().numpy()
            if self._include_positional_comparisons:
                ret[f"{prefix}maps/{k}"] = self._plot_image(v, k)
            ret[f"{prefix}metrics/{k}"] = v.mean()
        return ret


class LatentStepAggregator:
    """
    Aggregates latent steps for visualization.

    The latent space expects that each output of the downscaling model is stacked in a
    'channel' dimension (index=-3) and that only selects the first example from the
    leading dimension to record.

    Args:
        name: The name to use for the latent steps in the wandb output.
    """

    def __init__(self, name: str = "") -> None:
        self._latent_steps: list[torch.Tensor] = []
        self._name = ensure_trailing_slash(name)

    @torch.no_grad()
    def record_batch(self, latent_steps: list[torch.Tensor]) -> None:
        self._latent_steps = latent_steps

    def _get(self) -> list[Any]:
        if len(self._latent_steps) == 0:
            return []

        wandb = WandB.get_instance()
        images = []

        # latents have a folded bach x sample dim
        # [batch * n_samples, output_channels, height, width]
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

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the latent step logs for wandb.
        """
        prefix = ensure_trailing_slash(prefix)
        return {f"{prefix}{self._name}": self._get()}


class SplitMapAndMetricLogs:
    """
    Wraps a comparison aggregator and splits the wandb logs into maps and metrics
    based on the shape of the stored metric tensor. Map (2D) metrics will also have
    an average value calculated and logged.

    Args:
        agg: The aggregator that provides a mix of multi-dimensional metrics.
        name: The name to use for the maps in the wandb output.
    """

    def __init__(
        self,
        agg: MeanComparison | SumComparison,
        name: str,
        include_positional_comparisons: bool = True,
    ) -> None:
        self._agg = agg
        self._name = ensure_trailing_slash(name)
        self._include_positional_comparisons = include_positional_comparisons

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping) -> None:
        """
        Record the metric values of a batch in the underlying comparison aggregator.
        """
        self._agg.record_batch(target, prediction)

    def _plot_map(self, data: torch.Tensor, key: str) -> Any:
        # assume an image metric if multiple dimensions
        if data.ndim != 2:
            raise ValueError("Expected data to have 2 dimensions")
        wandb = WandB.get_instance()
        caption = _get_map_caption(key, data)
        diverging = data.min() < 0 and data.max() > 0
        vmin, vmax = get_cmap_limits(data, diverging=diverging)
        fig = plot_imshow(data, vmin=vmin, vmax=vmax, use_colorbar=True)
        image = wandb.Image(fig, caption=caption)
        plt.close(fig)
        return image

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the wandb output to log. The names of the output items are segmented
        into {prefix}/maps for 2D data,a nd {prefix}/metrics for single-value data.
        """
        prefix = ensure_trailing_slash(prefix)
        results = self._agg.get_wandb()
        ret = {}
        for k, v in results.items():
            if len(v.shape) > 1:
                if self._include_positional_comparisons:
                    ret[f"{prefix}maps/{self._name}{k}"] = self._plot_map(v, k)
                ret[f"{prefix}metrics/{self._name}{k}"] = v.mean()
            else:
                ret[f"{prefix}metrics/{self._name}{k}"] = v
        return ret


class GenerationAggregator:
    """
    Aggregator for generated outputs from a denoising diffusion model
    that includes a sample dimension (batch, sample, ...). Utilizes
    an Aggregator for the metrics on folded (sample x batch dim) data.

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
        dims: list[str],
        downscale_factor: int,
        n_histogram_bins: int = 300,
        percentiles: list[float] | None = None,
        ssim_kwargs: Mapping[str, Any] | None = None,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        include_positional_comparisons: bool = True,
    ) -> None:
        self._agg = Aggregator(
            dims,
            downscale_factor,
            n_histogram_bins,
            percentiles,
            ssim_kwargs,
            variable_metadata,
            include_positional_comparisons,
        )

        # Non-folded sample dimension quantities
        self._probabilistic_comparisons = [
            SplitMapAndMetricLogs(
                MeanComparison(_batch_mean_crps), "crps", include_positional_comparisons
            ),
        ]

        self._probabilistic_coarse_comparisons: list[_CoarseComparisonAggregator] = [
            RelativeCRPSInterpAggregator(
                downscale_factor,
                name="relative_crps_bicubic",
                include_positional_comparisons=include_positional_comparisons,
            )
        ]

        self._latent_step_aggregator = LatentStepAggregator(
            name="maps/snapshot/latent_steps"
        )

    @torch.no_grad()
    def record_batch(
        self, outputs: ModelOutputs, coarse: TensorMapping, batch: PairedBatchData
    ) -> None:
        _check_batch_dims_for_recording(outputs, coarse, 4)
        target, prediction = outputs.target, outputs.prediction
        target = filter_tensor_mapping(target, prediction.keys())

        for prob_comparison_aggregator in self._probabilistic_comparisons:
            prob_comparison_aggregator.record_batch(target, prediction)

        for prob_coarse_comp_agg in self._probabilistic_coarse_comparisons:
            prob_coarse_comp_agg.record_batch(target, prediction, coarse)

        self._latent_step_aggregator.record_batch(outputs.latent_steps)

        sample_dim = 1
        folded_target, folded_prediction, folded_coarse = _fold_sample_dim(
            [target, prediction, coarse], sample_dim=sample_dim
        )
        folded_outputs = ModelOutputs(
            folded_prediction, folded_target, outputs.loss, []
        )
        num_samples = next(iter(prediction.values())).shape[sample_dim]
        expanded_batch = batch.expand_and_fold(
            num_samples, sample_dim, "folded_samples"
        )
        self._agg.record_batch(folded_outputs, folded_coarse, expanded_batch)

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the wandb output to log from all sub aggregators.
        """
        ret = {**self._agg.get_wandb(prefix)}
        for comparison in self._probabilistic_comparisons:
            ret.update(comparison.get_wandb(prefix))
        for coarse_comparison in self._probabilistic_coarse_comparisons:
            ret.update(coarse_comparison.get_wandb(prefix))
        ret.update(self._latent_step_aggregator.get_wandb(prefix))
        return ret

    def get_dataset(self) -> xr.Dataset:
        """
        Get the dataset output from the underlying aggregator.
        """
        return self._agg.get_dataset()
