import dataclasses
from typing import Literal

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.typing_ import TensorDict
from fme.downscaling.data import BatchData, PairedBatchData, scale_tuple
from fme.downscaling.data.patching import Patch, get_patches
from fme.downscaling.models import (
    DiffusionModel,
    ModelOutputs,
    _repeat_batch_by_samples,
)
from fme.downscaling.predictors.multidiffusion import generate_multidiffusion
from fme.downscaling.predictors.serial_denoising import DenoisingMoEPredictor


@dataclasses.dataclass
class PatchPredictionConfig:
    """
    Configuration to enable predictions on multiple patches for evaluation.

    Args:
        divide_generation: enables the patched prediction of the full
            input data extent for generation.
        composite_prediction: if True, recombines the smaller prediction
            regions into the original full region as a single sample.
            Both divide_generation and composite_prediction must be
            True or both False. Previously, divide_generation=True and
            composite_prediction=False were allowed; this is no longer
            supported.
        coarse_horizontal_overlap: number of pixels to overlap in the
            coarse data.
        multidiffusion: if True, generate a single full-extent sample with
            MultiDiffusion (https://arxiv.org/abs/2302.08113): at every
            denoising step, each patch is denoised at the model's trained
            patch shape and overlapping patch estimates are averaged, instead
            of generating each patch independently and averaging the finished
            patch samples. Requires divide_generation=True and
            composite_prediction=True. Larger coarse_horizontal_overlap values
            give smoother joins between patches at the cost of more model
            calls.
        multidiffusion_sampler_dtype: dtype of the full-domain MultiDiffusion
            sampler state and updates. Requires multidiffusion=True.
        multidiffusion_sampler_state_on_cpu: if True, keep the full-domain
            MultiDiffusion sampler state and per-patch conditioning in host
            memory, moving each patch to the GPU only for its network call,
            so GPU memory does not grow with the domain size. Requires
            multidiffusion=True.
    """

    divide_generation: bool = False
    composite_prediction: bool = False
    coarse_horizontal_overlap: int = 1
    multidiffusion: bool = False
    multidiffusion_sampler_dtype: Literal["float32", "float64"] = "float64"
    multidiffusion_sampler_state_on_cpu: bool = False

    def __post_init__(self):
        if self.divide_generation and not self.composite_prediction:
            raise ValueError(
                "divide_generation=True requires composite_prediction=True."
            )
        if not self.divide_generation and self.composite_prediction:
            raise ValueError(
                "composite_prediction=True requires divide_generation=True."
            )
        if self.multidiffusion and not self.divide_generation:
            raise ValueError(
                "multidiffusion=True requires divide_generation=True and "
                "composite_prediction=True."
            )
        if not self.multidiffusion and (
            self.multidiffusion_sampler_dtype != "float64"
            or self.multidiffusion_sampler_state_on_cpu
        ):
            raise ValueError(
                "multidiffusion_sampler_dtype and multidiffusion_sampler_state_on_cpu "
                "require multidiffusion=True."
            )

    @property
    def needs_patch_predictor(self):
        if self.divide_generation and self.composite_prediction:
            return True
        return False


class PatchPredictor:
    """
    Model prediction wrapper for generating a full-extent prediction
    by dividing the input into a grid of patches.

    Patch size is inferred from the model's coarse_shape used in training.
    """

    def __init__(
        self,
        model: DiffusionModel | DenoisingMoEPredictor,
        coarse_horizontal_overlap: int = 1,
        multidiffusion: bool = False,
        sampler_dtype: torch.dtype = torch.float64,
        sampler_state_on_cpu: bool = False,
    ):
        """
        Args:
            model: the model to use for generating predictions.
            coarse_horizontal_overlap: the number of pixels to overlap
                between patches in the coarse data.
            multidiffusion: if True, jointly denoise the full extent with
                MultiDiffusion instead of averaging independently generated
                patches. See PatchPredictionConfig.
            sampler_dtype: MultiDiffusion sampler state dtype.
            sampler_state_on_cpu: if True, keep the MultiDiffusion sampler
                state in host memory.
        """
        self.model = model
        self.modules = self.model.modules

        self.coarse_yx_patch_extent = self.model.coarse_shape
        self.downscale_factor = self.model.downscale_factor
        self.coarse_horizontal_overlap = coarse_horizontal_overlap
        self.multidiffusion = multidiffusion
        self.sampler_dtype = sampler_dtype
        self.sampler_state_on_cpu = sampler_state_on_cpu

    @classmethod
    def from_config(
        cls,
        model: DiffusionModel | DenoisingMoEPredictor,
        config: PatchPredictionConfig,
    ) -> "PatchPredictor":
        return cls(
            model,
            coarse_horizontal_overlap=config.coarse_horizontal_overlap,
            multidiffusion=config.multidiffusion,
            sampler_dtype=getattr(torch, config.multidiffusion_sampler_dtype),
            sampler_state_on_cpu=config.multidiffusion_sampler_state_on_cpu,
        )

    @property
    def coarse_shape(self) -> tuple[int, int]:
        return self.coarse_yx_patch_extent

    @property
    def static_inputs(self):
        return self.model.static_inputs

    @property
    def full_fine_coords(self):
        return self.model.full_fine_coords

    def get_fine_coords_for_batch(self, batch: BatchData) -> LatLonCoordinates:
        return self.model.get_fine_coords_for_batch(batch)

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
        n_samples: int = 1,
    ) -> ModelOutputs:
        coarse_patches, fine_patches = self._get_patches(
            coarse_yx_extent=batch.coarse.horizontal_shape,
            fine_yx_extent=batch.fine.horizontal_shape,
        )
        if self.multidiffusion:
            return self._generate_on_batch_multidiffusion(
                batch, coarse_patches, n_samples
            )

        predictions = []
        loss = 0.0
        batch_generator = batch.generate_from_patches(
            coarse_patches=coarse_patches, fine_patches=fine_patches
        )

        for data_patch in batch_generator:
            model_output = self.model.generate_on_batch(data_patch, n_samples)
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
        n_samples: int = 1,
    ) -> TensorDict:
        coarse_yx_extent = batch.horizontal_shape
        fine_yx_extent = scale_tuple(coarse_yx_extent, self.downscale_factor)
        coarse_patches, fine_patches = self._get_patches(
            coarse_yx_extent=coarse_yx_extent, fine_yx_extent=fine_yx_extent
        )
        if self.multidiffusion:
            generated, _ = generate_multidiffusion(
                self.model,
                batch,
                coarse_patches,
                n_samples,
                sampler_dtype=self.sampler_dtype,
                sampler_state_on_cpu=self.sampler_state_on_cpu,
            )
            return generated

        predictions = []
        batch_generator = batch.generate_from_patches(coarse_patches)
        for data_patch in batch_generator:
            predictions.append(
                self.model.generate_on_batch_no_target(
                    batch=data_patch,
                    n_samples=n_samples,
                )
            )
        prediction = composite_patch_predictions(predictions, fine_patches)

        return prediction

    def _generate_on_batch_multidiffusion(
        self,
        batch: PairedBatchData,
        coarse_patches: list[Patch],
        n_samples: int,
    ) -> ModelOutputs:
        base_model = self.model.base_model
        generated, generated_norm = generate_multidiffusion(
            self.model,
            batch.coarse,
            coarse_patches,
            n_samples,
            sampler_dtype=self.sampler_dtype,
            sampler_state_on_cpu=self.sampler_state_on_cpu,
        )
        targets_norm = base_model.out_packer.pack(
            base_model.normalizer.fine.normalize(dict(batch.fine.data)),
            axis=base_model._channel_axis,
        )
        targets_norm = _repeat_batch_by_samples(targets_norm, n_samples)
        [loss_component] = base_model.loss(generated_norm, targets_norm)
        # add ensemble dim to the target since not provided by the model
        target = {k: v.unsqueeze(1) for k, v in batch.fine.data.items()}
        return ModelOutputs(
            prediction=generated,
            target=target,
            loss=loss_component.loss.mean(),
        )


def check_input_shape_supported(
    model_coarse_shape: tuple[int, int],
    input_shape: tuple[int, int],
    patch: PatchPredictionConfig,
    name: str = "",
) -> None:
    suffix = f" for {name}" if name else ""
    if model_coarse_shape == input_shape:
        return
    if any(
        model_input_size > data_input_size
        for model_input_size, data_input_size in zip(model_coarse_shape, input_shape)
    ):
        raise ValueError(
            f"Model coarse shape {model_coarse_shape} is larger than "
            f"actual input shape {input_shape}{suffix}. "
            "We do not support generating outputs with a smaller spatial extent"
            " than the model's trained patch size. Please adjust the spatial extent"
            " to be at least as large as the model's input patch size "
            f"{model_coarse_shape}."
        )
    # If data shape is larger than model input shape, currently require patch prediction
    # as the models predict very biased results for out-of-sample input shapes.
    if not patch.needs_patch_predictor:
        raise ValueError(
            f"Input datashape {input_shape}{suffix} is larger than the model input "
            f"patch size {model_coarse_shape} and patch prediction is not configured. "
            "Generation for larger domains requires patch prediction configured with "
            "composite_prediction=True and divide_generation=True."
        )


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
