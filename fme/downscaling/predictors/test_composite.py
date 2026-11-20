import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.packer import Packer
from fme.downscaling.aggregators.shape_helpers import upsample_tensor
from fme.downscaling.data import BatchData, PairedBatchData
from fme.downscaling.data.patching import get_patches
from fme.downscaling.data.utils import BatchedLatLonCoordinates
from fme.downscaling.models import ModelOutputs
from fme.downscaling.predictors.composite import (
    PatchPredictionConfig,
    PatchPredictor,
    check_input_shape_supported,
    composite_patch_predictions,
    get_generation_model,
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


def test_composite_predictions_trims_shifted_last_patch():
    # On a 1x4 domain, 1x3 patches with overlap 1 start at columns 0 and 2; the
    # second runs past the edge, so it is shifted back to columns 1-3 and its
    # first column is trimmed, leaving the configured one-column overlap.
    patches = get_patches((1, 4), (1, 3), overlap=1, drop_partial_patches=False)
    assert patches[1].output_slice.x == slice(1, None)
    predictions = [{"x": torch.full((2, 3, 1, 3), value)} for value in (1.0, 3.0)]
    composited = composite_patch_predictions(predictions, patches)["x"]
    # without the trim, column 1 would also average the two patches, to 2.0
    expected = torch.tensor([1.0, 1.0, 2.0, 3.0]).expand(2, 3, 1, 4)
    torch.testing.assert_close(composited, expected)


_COMPOSITE_PATCH = PatchPredictionConfig(
    divide_generation=True, composite_prediction=True
)
_NO_PATCH = PatchPredictionConfig()


@pytest.mark.parametrize("requires_training_patch_shape", [True, False])
def test_check_input_shape_supported_exact_match(requires_training_patch_shape):
    """Exact match is supported regardless of patch config (does not raise)."""
    check_input_shape_supported(
        (16, 16), (16, 16), _NO_PATCH, requires_training_patch_shape
    )


@pytest.mark.parametrize("requires_training_patch_shape", [True, False])
def test_check_input_shape_supported_larger_with_composite(
    requires_training_patch_shape,
):
    """A larger extent is supported when composite patch prediction is configured."""
    check_input_shape_supported(
        (16, 16), (32, 32), _COMPOSITE_PATCH, requires_training_patch_shape
    )


@pytest.mark.parametrize("requires_training_patch_shape", [True, False])
@pytest.mark.parametrize("patch", [_NO_PATCH, _COMPOSITE_PATCH])
@pytest.mark.parametrize("input_shape", [(8, 16), (16, 8), (8, 8), (8, 32)])
def test_check_input_shape_supported_raises_when_too_small(
    input_shape, patch, requires_training_patch_shape
):
    """Raise when the input is smaller than the model patch in any dimension."""
    with pytest.raises(ValueError, match="smaller spatial extent"):
        check_input_shape_supported(
            (16, 16), input_shape, patch, requires_training_patch_shape
        )


def test_check_input_shape_supported_legacy_raises_when_larger_without_patching():
    """
    Raise when a model with align_corners=True upsampling gets a larger extent but
    patching isn't configured.
    """
    with pytest.raises(ValueError, match="require patch prediction"):
        check_input_shape_supported(
            (16, 16), (32, 32), _NO_PATCH, requires_training_patch_shape=True
        )


def test_check_input_shape_supported_larger_without_patching():
    """
    A larger extent without patching is supported when the model does not
    require the training patch shape.
    """
    check_input_shape_supported(
        (16, 16), (32, 32), _NO_PATCH, requires_training_patch_shape=False
    )


class DummyModel:
    def __init__(
        self, coarse_shape, downscale_factor, requires_training_patch_shape=True
    ):
        self.coarse_shape = coarse_shape
        self.downscale_factor = downscale_factor
        self.requires_training_patch_shape = requires_training_patch_shape
        self.modules = []
        self.out_packer = Packer(["x"])

    def generate_on_batch(self, batch: PairedBatchData, n_samples=1):
        prediction_data = {
            k: v.unsqueeze(1).expand(-1, n_samples, -1, -1)
            for k, v in batch.fine.data.items()
        }
        return ModelOutputs(
            prediction=prediction_data, target=prediction_data, loss=torch.tensor(1.0)
        )

    def generate_on_batch_no_target(self, batch: BatchData, n_samples=1):
        prediction_data = {
            k: upsample_tensor(
                v.unsqueeze(1).expand(-1, n_samples, -1, -1),
                upsample_factor=self.downscale_factor,
            )
            for k, v in batch.data.items()
        }
        return prediction_data


@pytest.mark.parametrize("patch", [_NO_PATCH, _COMPOSITE_PATCH])
def test_get_generation_model_returns_model_for_exact_match(patch):
    model = DummyModel(coarse_shape=(16, 16), downscale_factor=2)
    # DummyModel duck-types the model interface used by get_generation_model
    assert get_generation_model(model, (16, 16), patch) is model  # type: ignore


def test_get_generation_model_wraps_larger_input_with_patch_predictor():
    model = DummyModel(coarse_shape=(16, 16), downscale_factor=2)
    generation_model = get_generation_model(
        model,  # type: ignore
        (32, 32),
        _COMPOSITE_PATCH,
    )
    assert isinstance(generation_model, PatchPredictor)
    assert generation_model.model is model
    assert generation_model.coarse_shape == (16, 16)


def test_get_generation_model_returns_model_for_larger_input_without_patching():
    """
    Models without align_corners=True upsampling generate on the full
    larger extent.
    """
    model = DummyModel(
        coarse_shape=(16, 16), downscale_factor=2, requires_training_patch_shape=False
    )
    generation_model = get_generation_model(
        model,  # type: ignore
        (32, 32),
        _NO_PATCH,
    )
    assert generation_model is model


def test_get_generation_model_legacy_raises_for_larger_input_without_patching():
    model = DummyModel(
        coarse_shape=(16, 16), downscale_factor=2, requires_training_patch_shape=True
    )
    with pytest.raises(ValueError, match="align_corners=True"):
        get_generation_model(model, (32, 32), _NO_PATCH)  # type: ignore


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

    predictor = PatchPredictor(
        DummyModel(coarse_shape=patch_size_coarse, downscale_factor=downscale_factor),  # type: ignore
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
    predictor = PatchPredictor(
        DummyModel(coarse_shape=patch_size_coarse, downscale_factor=2),  # type: ignore
        coarse_horizontal_overlap=1,
    )
    n_samples_generate = 2
    coarse_batch_data = paired_batch_data.coarse
    prediction = predictor.generate_on_batch_no_target(
        coarse_batch_data, n_samples=n_samples_generate
    )
    assert prediction["x"].shape == (batch_size, n_samples_generate, 8, 8)


@pytest.mark.parametrize(
    "divide_generation, composite_prediction, error_submessage",
    [
        (True, False, "divide_generation=True requires composite_prediction=True."),
        (False, True, "composite_prediction=True requires divide_generation=True."),
    ],
)
def test_PatchPredictionConfig_disallowed_configs(
    divide_generation, composite_prediction, error_submessage
):
    with pytest.raises(ValueError, match=error_submessage):
        PatchPredictionConfig(
            divide_generation=divide_generation,
            composite_prediction=composite_prediction,
        )


class _ShapeRecordingModel(DummyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shapes: list[tuple[int, int]] = []

    def generate_on_batch(self, batch: PairedBatchData, n_samples=1):
        self.input_shapes.append(tuple(batch.coarse.horizontal_shape))
        return super().generate_on_batch(batch, n_samples)


def test_PatchPredictor_larger_coarse_patch_extent():
    """Patches larger than the training patch tile the domain and composite."""
    paired_batch_data = get_paired_test_data(6, 6, downscale_factor=2, batch_size=1)
    model = _ShapeRecordingModel(
        coarse_shape=(2, 2), downscale_factor=2, requires_training_patch_shape=False
    )
    predictor = PatchPredictionConfig(
        divide_generation=True,
        composite_prediction=True,
        coarse_horizontal_overlap=1,
        coarse_patch_extent_lat=4,
        coarse_patch_extent_lon=4,
    ).build(model)  # type: ignore
    outputs = predictor.generate_on_batch(paired_batch_data, n_samples=1)
    # 6 = 4 + 4 - 2 (overlap 1, last patch shifted back): a 2 x 2 grid of 4 x 4.
    assert model.input_shapes == [(4, 4)] * 4
    assert torch.equal(outputs.prediction["x"][:, 0], outputs.target["x"][:, 0])


def test_PatchPredictor_coarse_patch_extent_smaller_than_model_raises():
    model = DummyModel(
        coarse_shape=(4, 4), downscale_factor=2, requires_training_patch_shape=False
    )
    with pytest.raises(ValueError, match="smaller than"):
        PatchPredictor(model, coarse_patch_extent=(2, 4))  # type: ignore


def test_PatchPredictor_coarse_patch_extent_requires_training_patch_shape():
    model = DummyModel(
        coarse_shape=(2, 2), downscale_factor=2, requires_training_patch_shape=True
    )
    with pytest.raises(ValueError, match="align_corners=True"):
        PatchPredictor(model, coarse_patch_extent=(4, 4))  # type: ignore


def test_PatchPredictionConfig_patch_extent_set_together():
    with pytest.raises(ValueError, match="set together"):
        PatchPredictionConfig(coarse_patch_extent_lat=4)
