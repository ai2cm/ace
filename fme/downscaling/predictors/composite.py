import dataclasses

import torch

from fme.core.typing_ import TensorDict
from fme.downscaling.data import BatchData, PairedBatchData, Topography, scale_tuple
from fme.downscaling.data.patching import Patch, get_patches
from fme.downscaling.data.utils import null_generator
from fme.downscaling.models import DiffusionModel, ModelOutputs
from fme.downscaling.predictors import CascadePredictor


@dataclasses.dataclass
class PatchPredictionConfig:
    """
    Configuration to enable predictions on multiple patches for evaluation.

    Args:
        divide_generation: enables the patched prediction of the full
            input data extent for generation.
        composite_prediction: if True, recombines the smaller prediction
            regions into the original full region as a single sample.
        coarse_horizontal_overlap: number of pixels to overlap in the
            coarse data.
    """

    divide_generation: bool = False
    composite_prediction: bool = False
    coarse_horizontal_overlap: int = 1

    @property
    def needs_patch_data_generator(self):
        # If final predictions are not composited together, the BatchData is divided
        # into patches before being passed to the top level no-target generation call.
        # If final predictions are composited together, the BatchData is divided
        # into patches within the PatchPredictor's generation method.
        if self.divide_generation and not self.composite_prediction:
            return True
        return False

    @property
    def needs_patch_predictor(self):
        if self.divide_generation and self.composite_prediction:
            return True
        return False


class PatchPredictor:
    """
    Model prediction wrapper for generating a full-extent prediction
    by dividing the input into a grid of patches.
    """

    def __init__(
        self,
        model: DiffusionModel | CascadePredictor,
        coarse_yx_patch_extent: tuple[int, int] | None = None,
        coarse_horizontal_overlap: int = 1,
    ):
        """
        Args:
            model: the model to use for generating predictions.
            coarse_yx_patch_extent: The shape of the coarse region passed
                to the downscaling model for prediction. If None, will be
                inferred from model.coarse_shape.
            coarse_horizontal_overlap: the number of pixels to overlap
                between patches in the coarse data.
        """
        self.model = model
        self.modules = self.model.modules

        if coarse_yx_patch_extent is None:
            coarse_yx_patch_extent = self.model.coarse_shape

        self.coarse_yx_patch_extent = coarse_yx_patch_extent
        self.downscale_factor = self.model.downscale_factor
        self.coarse_horizontal_overlap = coarse_horizontal_overlap

    @property
    def coarse_shape(self):
        return self.coarse_yx_patch_extent

    def _get_patches(
        self, coarse_yx_extent, fine_yx_extent
    ) -> tuple[list[Patch], list[Patch]]:
        coarse_patches = get_patches(
            yx_extent=coarse_yx_extent,
            yx_patch_extent=self.coarse_yx_patch_extent,
            overlap=self.coarse_horizontal_overlap,
            drop_partial_patches=False,
        )
        fine_yx_patch_extent = scale_tuple(
            self.coarse_yx_patch_extent, self.downscale_factor
        )

        fine_patches = get_patches(
            yx_extent=fine_yx_extent,
            yx_patch_extent=fine_yx_patch_extent,
            overlap=self.coarse_horizontal_overlap * self.downscale_factor,
            drop_partial_patches=False,
        )

        return coarse_patches, fine_patches

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        topography: Topography | None,
        n_samples: int = 1,
    ) -> ModelOutputs:
        predictions = []
        loss = 0.0

        coarse_patches, fine_patches = self._get_patches(
            coarse_yx_extent=batch.coarse.horizontal_shape,
            fine_yx_extent=batch.fine.horizontal_shape,
        )
        batch_generator = batch.generate_from_patches(
            coarse_patches=coarse_patches, fine_patches=fine_patches
        )
        if topography is not None:
            topography_generator = topography.generate_from_patches(fine_patches)
        else:
            topography_generator = null_generator(len(fine_patches))

        for data_patch, topography_patch in zip(batch_generator, topography_generator):
            model_output = self.model.generate_on_batch(
                data_patch, topography_patch, n_samples
            )
            predictions.append(model_output.prediction)
            loss = loss + model_output.loss

        prediction = composite_patch_predictions(predictions, fine_patches)
        # add ensemble dim to the target since not provided by the model
        target = {k: v.unsqueeze(1) for k, v in batch.fine.data.items()}
        outputs = ModelOutputs(
            prediction=prediction,
            target=target,
            loss=loss,
        )
        return outputs

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        topography: Topography | None,
        n_samples: int = 1,
    ) -> TensorDict:
        coarse_yx_extent = batch.horizontal_shape
        fine_yx_extent = scale_tuple(coarse_yx_extent, self.downscale_factor)
        coarse_patches, fine_patches = self._get_patches(
            coarse_yx_extent=coarse_yx_extent, fine_yx_extent=fine_yx_extent
        )
        predictions = []
        batch_generator = batch.generate_from_patches(coarse_patches)
        if topography is not None:
            topography_generator = topography.generate_from_patches(fine_patches)
        else:
            topography_generator = null_generator(len(fine_patches))
        for data_patch, topo_patch in zip(batch_generator, topography_generator):
            predictions.append(
                self.model.generate_on_batch_no_target(
                    batch=data_patch,
                    topography=topo_patch,
                    n_samples=n_samples,
                )
            )
        prediction = composite_patch_predictions(predictions, fine_patches)

        return prediction


def _get_full_extent_from_patches(patches: list[Patch]) -> tuple[int, int]:
    # input patches should have int start/stop values
    y_max = max(patch.input_slice.y.stop for patch in patches)
    x_max = max(patch.input_slice.x.stop for patch in patches)
    return y_max, x_max


def _composite_patch_tensors(
    predictions: list[torch.Tensor], patches: list[Patch]
) -> TensorDict:
    if len(predictions) != len(patches):
        raise ValueError("The number of predictions must match the number of patches.")

    y_size, x_size = _get_full_extent_from_patches(patches)
    # list elements have shape (n_batch, n_generated_sample, patch_lat, patch_lon)
    n_batch, n_gen_sample = predictions[0].shape[:2]
    output_sum = torch.zeros(
        n_batch, n_gen_sample, y_size, x_size, device=predictions[0].device
    )
    output_count = torch.zeros(
        n_batch, n_gen_sample, y_size, x_size, device=predictions[0].device
    )

    for i, pred in enumerate(predictions):
        in_slice = patches[i].input_slice
        out_slice = patches[i].output_slice
        # Adjust the input slice start if the output slice is trimmed
        adjusted_in_slice_y = slice(
            in_slice.y.start + (out_slice.y.start or 0), in_slice.y.stop
        )
        adjusted_in_slice_x = slice(
            in_slice.x.start + (out_slice.x.start or 0), in_slice.x.stop
        )
        trimmed_pred = pred[..., out_slice.y, out_slice.x]
        output_sum[..., adjusted_in_slice_y, adjusted_in_slice_x] += trimmed_pred
        output_count[..., adjusted_in_slice_y, adjusted_in_slice_x] += 1
    return output_sum / output_count


def composite_patch_predictions(
    predictions: list[TensorDict], patches: list[Patch]
) -> TensorDict:
    """
    Take the predictions from patches and combine them into a single
    tensor with the full extent of the patches. The predictions are
    averaged in overlapping patch regions.
    """
    combined_data = {}
    predicted_vars = list(predictions[0].keys())
    for var in predicted_vars:
        var_predictions = [pred[var] for pred in predictions]
        combined_data[var] = _composite_patch_tensors(var_predictions, patches)
    return combined_data
