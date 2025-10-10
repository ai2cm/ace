import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.packer import Packer
from fme.downscaling.aggregators.shape_helpers import upsample_tensor
from fme.downscaling.data import BatchData, PairedBatchData, Topography
from fme.downscaling.data.patching import get_patches
from fme.downscaling.data.utils import BatchedLatLonCoordinates
from fme.downscaling.models import ModelOutputs
from fme.downscaling.predictors.composite import (
    PatchPredictor,
    composite_patch_predictions,
)


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
                device=predictions[0]["x"].device,
            ),
        )


class DummyModel:
    def __init__(self, coarse_shape, downscale_factor):
        self.coarse_shape = coarse_shape
        self.downscale_factor = downscale_factor
        self.modules = []
        self.out_packer = Packer(["x"])

    def generate_on_batch(
        self, batch: PairedBatchData, topography: Topography | None, n_samples=1
    ):
        prediction_data = {
            k: v.unsqueeze(1).expand(-1, n_samples, -1, -1)
            for k, v in batch.fine.data.items()
        }
        return ModelOutputs(
            prediction=prediction_data, target=prediction_data, loss=torch.tensor(1.0)
        )

    def generate_on_batch_no_target(
        self, batch: BatchData, topography: Topography | None, n_samples=1
    ):
        prediction_data = {
            k: upsample_tensor(
                v.unsqueeze(1).expand(-1, n_samples, -1, -1),
                upsample_factor=self.downscale_factor,
            )
            for k, v in batch.data.items()
        }
        return prediction_data


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
def test_SpatialCompositePredictor_generate_on_batch(patch_size_coarse):
    batch_size = 3
    coarse_extent = (4, 4)
    downscale_factor = 2
    paired_batch_data = get_paired_test_data(
        *coarse_extent, downscale_factor=downscale_factor, batch_size=batch_size
    )
    topography = Topography(
        torch.randn(
            coarse_extent[0] * downscale_factor, coarse_extent[1] * downscale_factor
        ),
        paired_batch_data.fine.latlon_coordinates[0],
    )

    predictor = PatchPredictor(
        DummyModel(coarse_shape=patch_size_coarse, downscale_factor=downscale_factor),  # type: ignore
        coarse_extent,
        coarse_horizontal_overlap=1,
    )
    n_samples_generate = 2
    outputs = predictor.generate_on_batch(
        paired_batch_data, topography, n_samples=n_samples_generate
    )
    assert outputs.prediction["x"].shape == (batch_size, n_samples_generate, 8, 8)
    # dummy model predicts same value as fine data for all samples
    for s in range(n_samples_generate):
        assert torch.equal(outputs.prediction["x"][:, s], outputs.target["x"][:, 0])


@pytest.mark.parametrize(
    "patch_size_coarse",
    [
        pytest.param((2, 2), id="no_partial_patch"),
        pytest.param((3, 3), id="partial_patch"),
    ],
)
def test_SpatialCompositePredictor_generate_on_batch_no_target(patch_size_coarse):
    batch_size = 3
    coarse_extent = (4, 4)
    downscale_factor = 2
    paired_batch_data = get_paired_test_data(
        *coarse_extent, downscale_factor=downscale_factor, batch_size=batch_size
    )
    topography = Topography(
        torch.randn(
            coarse_extent[0] * downscale_factor, coarse_extent[1] * downscale_factor
        ),
        paired_batch_data.fine.latlon_coordinates[0],
    )
    predictor = PatchPredictor(
        DummyModel(coarse_shape=patch_size_coarse, downscale_factor=2),  # type: ignore
        coarse_extent,
        coarse_horizontal_overlap=1,
    )
    n_samples_generate = 2
    coarse_batch_data = paired_batch_data.coarse
    prediction = predictor.generate_on_batch_no_target(
        coarse_batch_data, topography, n_samples=n_samples_generate
    )
    assert prediction["x"].shape == (batch_size, n_samples_generate, 8, 8)
