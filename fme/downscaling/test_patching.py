import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.packer import Packer
from fme.downscaling.datasets_new import (
    BatchData,
    BatchedLatLonCoordinates,
    PairedBatchData,
)
from fme.downscaling.models import ModelOutputs
from fme.downscaling.patching import (
    PatchPredictor,
    _divide_into_slices,
    _get_patch_slices,
    composite_patch_predictions,
    get_patches,
)


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
        yx_extents=yx_extents,
        yx_patch_extents=yx_patch_extents,
        overlap=overlap,
        drop_partial_patches=True,
    )
    assert len(patches) == expected_num_patches
    for patch in patches:
        assert patch.output_slice.y == slice(None, None)
        assert patch.output_slice.x == slice(None, None)


def test_composite_predictions():
    patch_yx_size = (2, 2)
    patches = get_patches((4, 4), patch_yx_size, overlap=0)
    batch_size, n_sample = 3, 2
    predictions = [
        {
            "x": (i + 1) * torch.ones(batch_size, n_sample, *patch_yx_size),
        }
        for i in range(4)
    ]
    composited = composite_patch_predictions(predictions, patches)["x"]
    assert composited.shape == (batch_size, n_sample, 4, 4)
    for batch_element in composited:
        assert torch.allclose(
            batch_element[0],
            torch.tensor(
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0],
                ],
                device=get_device(),
            ),
        )


class DummyModel:
    def __init__(self, coarse_shape, downscale_factor):
        self.coarse_shape = coarse_shape
        self.downscale_factor = downscale_factor
        self.modules = []
        self.out_packer = Packer(["x"])

    def generate_on_batch(self, batch, n_samples=1):
        prediction_data = {
            k: v.unsqueeze(1).expand(-1, n_samples, -1, -1)
            for k, v in batch.fine.data.items()
        }
        return ModelOutputs(
            prediction=prediction_data, target=prediction_data, loss=torch.tensor(1.0)
        )


def get_paired_test_data(
    coarse_lat_size, coarse_lon_size, downscale_factor, batch_size
):
    fine_lat_size, fine_lon_size = (
        coarse_lat_size * downscale_factor,
        coarse_lon_size * downscale_factor,
    )
    coarse_data = {
        "x": torch.rand(
            batch_size, coarse_lat_size, coarse_lon_size, device=get_device()
        )
    }
    coarse_lat_coords = (
        torch.linspace(0, 10, coarse_lat_size).unsqueeze(0).expand(batch_size, -1)
    )
    coarse_lon_coords = (
        torch.linspace(0, 10, coarse_lon_size).unsqueeze(0).expand(batch_size, -1)
    )

    fine_data = {
        "x": torch.rand(batch_size, fine_lat_size, fine_lon_size, device=get_device())
    }
    fine_lat_coords = (
        torch.linspace(0.0, 10.0, fine_lat_size).unsqueeze(0).expand(batch_size, -1)
    )
    fine_lon_coords = (
        torch.linspace(0.0, 10.0, fine_lon_size).unsqueeze(0).expand(batch_size, -1)
    )
    coarse_batch_data = BatchData(
        data=coarse_data,
        latlon_coordinates=BatchedLatLonCoordinates(
            lat=coarse_lat_coords, lon=coarse_lon_coords
        ),
        time=xr.DataArray(np.arange(batch_size), dims=["time"]),
    )
    fine_batch_data = BatchData(
        data=fine_data,
        latlon_coordinates=BatchedLatLonCoordinates(
            lat=fine_lat_coords, lon=fine_lon_coords
        ),
        time=xr.DataArray(np.arange(batch_size), dims=["time"]),
    )
    return PairedBatchData(coarse=coarse_batch_data, fine=fine_batch_data)


@pytest.mark.parametrize(
    "patch_size_coarse",
    [
        pytest.param((2, 2), id="no_partial_patch"),
        pytest.param((3, 3), id="partial_patch"),
    ],
)
def test_SpatialCompositePredictor(patch_size_coarse):
    batch_size = 3
    coarse_extent = (4, 4)
    paired_batch_data = get_paired_test_data(
        *coarse_extent, downscale_factor=2, batch_size=batch_size
    )
    predictor = PatchPredictor(
        DummyModel(coarse_shape=patch_size_coarse, downscale_factor=2),  # type: ignore
        coarse_extent,
        coarse_horizontal_overlap=1,
    )
    n_samples_generate = 2
    outputs = predictor.generate_on_batch(
        paired_batch_data, n_samples=n_samples_generate
    )
    assert outputs.prediction["x"].shape == (batch_size, n_samples_generate, 8, 8)
    # dummy model predicts same value as fine data for all samples
    for s in range(n_samples_generate):
        assert torch.equal(outputs.prediction["x"][:, s], outputs.target["x"][:, 0])
