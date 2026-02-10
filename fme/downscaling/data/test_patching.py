import numpy as np
import pytest
import torch

from fme.downscaling.data import PairedBatchData
from fme.downscaling.data.datasets import patched_batch_gen_from_paired_loader
from fme.downscaling.data.patching import (
    _divide_into_slices,
    _get_patch_slices,
    get_patches,
)
from fme.downscaling.predictors.test_composite import get_paired_test_data


@pytest.mark.parametrize(
    "full_coords_size, patch_slice, expected_input_slice, expected_output_slice",
    [
        (6, slice(4, 6), slice(4, 6), slice(None, None)),
        (6, slice(5, 7), slice(4, 6), slice(1, None)),
    ],
)
def test__adjust_out_of_bounds_slice(
    full_coords_size, patch_slice, expected_input_slice, expected_output_slice
):
    input_slice, output_slice = _get_patch_slices(full_coords_size, patch_slice)
    assert expected_input_slice == input_slice
    assert expected_output_slice == output_slice


@pytest.mark.parametrize(
    "full_size, patch_size, overlap, expected_slices",
    [
        pytest.param(6, 3, 0, [slice(0, 3), slice(3, 6)], id="overlap_0"),
        pytest.param(6, 3, 1, [slice(0, 3), slice(2, 5), slice(4, 7)], id="overlap_1"),
        pytest.param(
            6,
            3,
            2,
            [slice(0, 3), slice(1, 4), slice(2, 5), slice(3, 6)],
            id="overlap_2",
        ),
    ],
)
def test__divide_into_slices(full_size, patch_size, overlap, expected_slices):
    slices = _divide_into_slices(full_size, patch_size, overlap)
    assert slices == expected_slices


@pytest.mark.parametrize(
    "patch_size, expected_num_patches",
    [
        pytest.param((2, 2), 25, id="patch_size_2_no_drop"),  # 5 x 5 patches
        pytest.param((3, 3), 4, id="patch_size_3_drop"),  # 2 x 2 patches
    ],
)
def test_get_patches_drops_partial(patch_size, expected_num_patches):
    yx_extents = (6, 6)
    yx_patch_extents = patch_size
    overlap = 1

    patches = get_patches(
        yx_extent=yx_extents,
        yx_patch_extent=yx_patch_extents,
        overlap=overlap,
        drop_partial_patches=True,
    )
    assert len(patches) == expected_num_patches
    for patch in patches:
        assert patch.output_slice.y == slice(None, None)
        assert patch.output_slice.x == slice(None, None)


def test_get_patches_with_offset():
    yx_extents = (4, 4)
    yx_patch_extents = (2, 2)
    overlap = 0

    offset_patches = get_patches(
        yx_extent=yx_extents,
        yx_patch_extent=yx_patch_extents,
        overlap=overlap,
        drop_partial_patches=True,
        y_offset=1,
        x_offset=1,
    )

    assert len(offset_patches) == 1
    assert offset_patches[0].input_slice.x == slice(1, 3)
    assert offset_patches[0].input_slice.y == slice(1, 3)
    assert offset_patches[0].output_slice.x == slice(None, None)
    assert offset_patches[0].output_slice.y == slice(None, None)


def _mock_data_loader(
    n_batches, coarse_y_size, coarse_x_size, downscale_factor, batch_size
):
    batch_data = get_paired_test_data(
        coarse_y_size, coarse_x_size, downscale_factor, batch_size
    )
    for batch in range(n_batches):
        yield batch_data


@pytest.mark.parametrize("overlap", [0, 2])
def test_paired_patches_with_random_offset_consistent(overlap):
    coarse_shape = (20, 20)
    downscale_factor = 2
    batch_size = 3
    loader = _mock_data_loader(
        10, *coarse_shape, downscale_factor=downscale_factor, batch_size=batch_size
    )

    full_data = next(iter(loader))
    full_coarse_coords = full_data.coarse.latlon_coordinates
    full_fine_coords = full_data.fine.latlon_coordinates

    y_offsets = []
    x_offsets = []
    batch_generator = patched_batch_gen_from_paired_loader(
        loader=loader,
        coarse_yx_extent=coarse_shape,
        coarse_yx_patch_extent=(10, 10),
        downscale_factor=downscale_factor,
        coarse_overlap=overlap,
        drop_partial_patches=True,
        random_offset=True,
    )
    paired_batch: PairedBatchData
    for paired_batch in batch_generator:
        assert paired_batch.coarse.data["x"].shape == (batch_size, 10, 10)
        assert paired_batch.fine.data["x"].shape == (batch_size, 20, 20)

        coarse_patch_coords = paired_batch.coarse.latlon_coordinates
        fine_patch_coords = paired_batch.fine.latlon_coordinates

        # Lookup the index of the coordinate in the full coords
        # that corresponds to the first patch coordinate in order to
        # determine the offset applied
        coarse_y_offset = torch.where(
            full_coarse_coords.lat[0] == coarse_patch_coords.lat[0, 0]
        )[0].item()
        coarse_x_offset = torch.where(
            full_coarse_coords.lon[0] == coarse_patch_coords.lon[0, 0]
        )[0].item()
        fine_y_offset = torch.where(
            full_fine_coords.lat[0] == fine_patch_coords.lat[0, 0]
        )[0].item()
        fine_x_offset = torch.where(
            full_fine_coords.lon[0] == fine_patch_coords.lon[0, 0]
        )[0].item()

        assert fine_y_offset == coarse_y_offset * downscale_factor
        assert fine_x_offset == coarse_x_offset * downscale_factor

        y_offsets.append(coarse_y_offset)
        x_offsets.append(coarse_x_offset)

    assert len(np.unique(y_offsets)) > 1
    assert len(np.unique(x_offsets)) > 1


@pytest.mark.parametrize("shuffle", [True, False])
def test_paired_patches_shuffle(shuffle):
    coarse_shape = (8, 8)
    downscale_factor = 2
    batch_size = 3
    loader = _mock_data_loader(
        10, *coarse_shape, downscale_factor=downscale_factor, batch_size=batch_size
    )
    generator0 = patched_batch_gen_from_paired_loader(
        loader=loader,
        coarse_yx_extent=coarse_shape,
        coarse_yx_patch_extent=(2, 2),
        downscale_factor=downscale_factor,
        coarse_overlap=0,
        drop_partial_patches=True,
        random_offset=False,
        shuffle=shuffle,
    )
    generator1 = patched_batch_gen_from_paired_loader(
        loader=loader,
        coarse_yx_extent=coarse_shape,
        coarse_yx_patch_extent=(2, 2),
        downscale_factor=downscale_factor,
        coarse_overlap=0,
        drop_partial_patches=True,
        random_offset=False,
        shuffle=shuffle,
    )

    patches0: list[PairedBatchData] = []
    patches1: list[PairedBatchData] = []
    for i in range(4):
        patches0.append(next(generator0))
        patches1.append(next(generator1))

    data0 = torch.concat([patch.coarse.data["x"] for patch in patches0], dim=0)
    data1 = torch.concat([patch.coarse.data["x"] for patch in patches1], dim=0)

    if shuffle:
        assert not torch.equal(data0, data1)
    else:
        assert torch.equal(data0, data1)
