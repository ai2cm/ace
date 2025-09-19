import dataclasses
from itertools import product


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
    yx_extent: tuple[int, int],
    yx_patch_extent: tuple[int, int],
    overlap: int,
    drop_partial_patches: bool = True,
    y_offset: int = 0,
    x_offset: int = 0,
) -> list[Patch]:
    """
    Generate a list of patches for the given extents and patch size.

    Args:
        yx_extent: the full horizontal (y, x) size of the data
        yx_patch_extent: the horizontal (y, x) size of the patches to generate
        overlap: the number of pixels to overlap between patches
        drop_partial_patches: if True, drop patches that are not full
            patches (i.e., the last patch in each dimension if not
            fully included by yx_extent bounds)
        y_offset: added to the start of y slices
        x_offset: added to the start of x slices
    """
    y_slices = _divide_into_slices(yx_extent[0], yx_patch_extent[0], overlap)
    x_slices = _divide_into_slices(yx_extent[1], yx_patch_extent[1], overlap)

    y_slices = [slice(s.start + y_offset, s.stop + y_offset) for s in y_slices]
    x_slices = [slice(s.start + x_offset, s.stop + x_offset) for s in x_slices]

    if drop_partial_patches:
        if y_slices[-1].stop > yx_extent[0]:
            y_slices.pop()
        if x_slices[-1].stop > yx_extent[1]:
            x_slices.pop()

    patches = []
    for y_sl, x_sl in product(y_slices, x_slices):
        y_in_slice, y_out_slice = _get_patch_slices(yx_extent[0], y_sl)
        x_in_slice, x_out_slice = _get_patch_slices(yx_extent[1], x_sl)
        patches.append(
            Patch(
                input_slice=_HorizontalSlice(y_in_slice, x_in_slice),
                output_slice=_HorizontalSlice(y_out_slice, x_out_slice),
            )
        )
    return patches


def _divide_into_slices(full_size: int, patch_size: int, overlap: int) -> list[slice]:
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
