import dataclasses
from itertools import product
from typing import Generator, List, Tuple, Union

import torch

from fme.core.device import get_device
from fme.core.typing_ import TensorDict
from fme.downscaling.datasets import BatchData, PairedBatchData
from fme.downscaling.models import DiffusionModel, Model, ModelOutputs


@dataclasses.dataclass
class _HorizontalSlice:
    y: slice
    x: slice


@dataclasses.dataclass
class Patch:
    """
    Describes slices for a patch of data with a specific extent
    for the input fields and the generated output.  We store input
    and output because the input slice may be shifted to keep within
    bounds of the available input, and the output field will be trimmed
    to keep a consistent overlap between patches.
    """

    input_slice: _HorizontalSlice
    output_slice: _HorizontalSlice

    def patch_batch_data(self, batch_data: BatchData) -> BatchData:
        return batch_data.latlon_slice(
            lat_slice=self.input_slice.y, lon_slice=self.input_slice.x
        )


def _get_patch_slices(full_coords_size: int, patch_slice: slice):
    """
    Get the input and output slices for a patch. Adjust the
    input and output slices when the patch stop is out of bounds
    to account for the full-tile input requirements and the need
    to trim the output to the correct overlap.
    """
    if patch_slice.stop > full_coords_size:
        oob_size = patch_slice.stop - full_coords_size
        # Adjust the stop of the input to avoid out of bounds
        input_slice = slice(
            patch_slice.start - oob_size,
            full_coords_size,
        )
        # Slice to apply to the output to avoid extra overlap with prior patch
        output_slice = slice(oob_size, None)
    else:
        input_slice = patch_slice
        output_slice = slice(None, None)
    return input_slice, output_slice


def get_patches(
    yx_extents: Tuple[int, int],
    yx_patch_extents: Tuple[int, int],
    overlap: int,
    drop_partial_patches: bool = True,
) -> List[Patch]:
    """
    Generate a list of patches for the given extents and patch size.

    Args:
        yx_extents: the full horizontal (y, x) size of the data
        yx_patch_extents: the horizontal (y, x) size of the patches to generate
        overlap: the number of pixels to overlap between patches
        drop_partial_patches: if True, drop patches that are not full
            patches (i.e., the last patch in each dimension if not
            fully included by yx_extent bounds)
    """
    y_slices = _divide_into_slices(yx_extents[0], yx_patch_extents[0], overlap)
    x_slices = _divide_into_slices(yx_extents[1], yx_patch_extents[1], overlap)

    if drop_partial_patches:
        if y_slices[-1].stop > yx_extents[0]:
            y_slices.pop()
        if x_slices[-1].stop > yx_extents[1]:
            x_slices.pop()

    patches = []
    for y_sl, x_sl in product(y_slices, x_slices):
        y_in_slice, y_out_slice = _get_patch_slices(yx_extents[0], y_sl)
        x_in_slice, x_out_slice = _get_patch_slices(yx_extents[1], x_sl)
        patches.append(
            Patch(
                input_slice=_HorizontalSlice(y_in_slice, x_in_slice),
                output_slice=_HorizontalSlice(y_out_slice, x_out_slice),
            )
        )
    return patches


def generate_patched_data(
    data: BatchData, patches: List[Patch]
) -> Generator[BatchData, None, None]:
    """
    Generate patches from the given data and patches.
    """
    # TODO: Generalize to BatchData / BatchItem, maybe use ABC
    for patch in patches:
        patch_data = patch.patch_batch_data(data)
        yield patch_data


def paired_patch_generator(
    batch: PairedBatchData,
    coarse_patches: List[Patch],
    fine_patches: List[Patch],
) -> Generator[PairedBatchData, None, None]:
    """
    Generate patches from paired fine/coarse data.
    """
    coarse_gen = generate_patched_data(
        batch.coarse,
        coarse_patches,
    )
    fine_gen = generate_patched_data(
        batch.fine,
        fine_patches,
    )

    for coarse_patch, fine_patch in zip(coarse_gen, fine_gen):
        yield PairedBatchData(
            fine=fine_patch,
            coarse=coarse_patch,
        )


def _get_full_extent_from_patches(patches: List[Patch]) -> Tuple[int, int]:
    # input patches should have int start/stop values
    y_max = max(patch.input_slice.y.stop for patch in patches)
    x_max = max(patch.input_slice.x.stop for patch in patches)
    return y_max, x_max


def composite_patch_predictions(
    predictions: List[TensorDict], patches: List[Patch]
) -> TensorDict:
    """
    Take the predictions from patches and combine them into a single
    tensor with the full extent of the patches. The predictions are
    averaged in overlapping patch regions.
    """
    if len(predictions) != len(patches):
        raise ValueError("The number of predictions must match the number of patches.")

    y_size, x_size = _get_full_extent_from_patches(patches)

    example_data_tensor = list(predictions[0].values())[0]
    prediction_vars = list(predictions[0].keys())

    # prediction tensors have dims [batch, generated_sample, lat, lon]
    # a temporary patch dimension is added at axis 0 and stacked before returning
    empty_tensor = torch.full(
        (
            len(predictions),
            example_data_tensor.shape[0],
            example_data_tensor.shape[1],
            y_size,
            x_size,
        ),
        torch.nan,
        device=get_device(),
    )
    combined_data = {var: empty_tensor for var in prediction_vars}

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
        for var in prediction_vars:
            combined_data[var][i, :, :, adjusted_in_slice_y, adjusted_in_slice_x] = (
                pred[var][..., out_slice.y, out_slice.x]
            )

    for var in prediction_vars:
        combined_data[var] = torch.nanmean(combined_data[var], dim=0)

    return combined_data


class PatchPredictor:
    """
    Model prediction wrapper for generating a full-extent prediction
    by dividing the input into a grid of patches.
    """

    def __init__(
        self,
        model: Union[DiffusionModel, Model],
        coarse_extent: Tuple[int, int],
        coarse_horizontal_overlap: int = 1,
    ):
        """
        Args:
            model: the model to use for generating predictions.
            coarse_extent: the full extent of the coarse data.
            coarse_horizontal_overlap: the number of pixels to overlap
                between patches in the coarse data.
        """
        self.model = model
        self.modules = self.model.modules

        self.downscale_factor = self.model.downscale_factor
        coarse_extent = coarse_extent
        fine_extent = (
            coarse_extent[0] * self.downscale_factor,
            coarse_extent[1] * self.downscale_factor,
        )
        coarse_patch_extent = self.model.coarse_shape
        fine_patch_extent = (
            coarse_patch_extent[0] * self.downscale_factor,
            coarse_patch_extent[1] * self.downscale_factor,
        )
        coarse_horizontal_overlap = coarse_horizontal_overlap
        fine_horizontal_overlap = coarse_horizontal_overlap * self.downscale_factor

        self._fine_patches = get_patches(
            yx_extents=fine_extent,
            yx_patch_extents=fine_patch_extent,
            overlap=fine_horizontal_overlap,
            drop_partial_patches=False,
        )
        self._coarse_patches = get_patches(
            yx_extents=coarse_extent,
            yx_patch_extents=coarse_patch_extent,
            overlap=coarse_horizontal_overlap,
            drop_partial_patches=False,
        )

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        predictions = []
        loss = 0.0
        for patch_data in paired_patch_generator(
            batch, self._coarse_patches, self._fine_patches
        ):
            model_output = self.model.generate_on_batch(patch_data, n_samples)
            predictions.append(model_output.prediction)
            loss = loss + model_output.loss

        prediction = composite_patch_predictions(predictions, self._fine_patches)
        # add ensemble dim to the target since not provided by the model
        target = {k: v.unsqueeze(1) for k, v in batch.fine.data.items()}
        outputs = ModelOutputs(
            prediction=prediction,
            target=target,
            loss=loss,
        )
        return outputs


def _divide_into_slices(full_size: int, patch_size: int, overlap: int) -> List[slice]:
    # Size covered by N patches = patch_size * N - (N-1)*overlap
    # The end of the last slice might extend past the end of full_size,
    # this is ok as it is adjusted during runtime to avoid out of bounds
    slices = [slice(0, patch_size)]
    _stop = slices[-1].stop
    while _stop < full_size:
        next_start = _stop - overlap
        next_stop = next_start + patch_size
        slices.append(slice(next_start, next_stop))
        _stop = next_stop
    return slices
