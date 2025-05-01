import pytest
import torch
import xarray as xr

from fme.core import get_device, metrics
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.testing.wandb import mock_wandb
from fme.core.typing_ import TensorMapping

from .. import metrics_and_maths
from ..datasets import BatchData, BatchedLatLonCoordinates, PairedBatchData
from ..models import ModelOutputs
from .generation import GenerationAggregator
from .main import Mean, MeanComparison, MeanMapAggregator, SnapshotAggregator
from .shape_helpers import (
    _check_all_datasets_compatible_sample_dim,
    _check_batch_dims_for_recording,
)


def assert_tensor_mapping_all_close(x: TensorMapping, y: TensorMapping):
    assert x.keys() == y.keys(), f"Keys do not match: {x.keys()} != {y.keys()}"
    for k in x:
        assert x[k].shape == y[k].shape, f"Shapes do not match for {k}"
        assert torch.allclose(x[k], y[k]), f"Values do not match for {k}"


@pytest.mark.parametrize("input_", [{"x": torch.tensor([1.0, 2.0, 3.0])}])
@pytest.mark.parametrize(
    "metric, expected_outputs, n_batches",
    [
        (
            torch.mean,
            {"x": torch.tensor(2.0)},
            1,
        ),
        (
            torch.mean,
            {"x": torch.tensor(2.0)},
            2,
        ),
        (torch.sum, {"x": torch.tensor(6.0)}, 1),
    ],
)
def test_mean_values(metric, input_, expected_outputs, n_batches):
    aggregator = Mean(metric)
    for _ in range(n_batches):
        aggregator.record_batch(input_)
    result = aggregator.get()
    assert_tensor_mapping_all_close(result, expected_outputs)


@pytest.mark.parametrize(
    "target, prediction, metric, expected_outputs",
    [
        (
            {"x": torch.tensor([0.0, 0.0])},
            {"x": torch.tensor([1.0, 1.0])},
            metrics.root_mean_squared_error,
            {"x": torch.tensor(1.0)},
        )
    ],
)
@pytest.mark.parametrize("n_batches", [1, 2])
def test_mean_comparison_values(
    metric, target, prediction, expected_outputs, n_batches
):
    aggregator = MeanComparison(metric)
    for _ in range(n_batches):
        aggregator.record_batch(target, prediction)
    result = aggregator.get()
    assert_tensor_mapping_all_close(
        result, {k: v for (k, v) in expected_outputs.items()}
    )


def test_mean_comparison_dynamic_metric():
    def add(x, y):
        return (x + y).sum()

    def multiply(x, y):
        return (x * y).sum()

    target = {"x": torch.tensor([1.0, 1.0])}
    prediction = {"x": torch.tensor([2.0, 2.0])}
    aggregator = MeanComparison(add)
    aggregator.record_batch(target, prediction)
    result = aggregator.get()
    assert result["x"] == 6.0

    # overrides default metric
    aggregator.record_batch(target, prediction, dynamic_metric=multiply)
    result = aggregator.get()
    assert result["x"] == 5.0

    # Test that no function fails
    aggregator = MeanComparison()
    with pytest.raises(ValueError):
        aggregator.record_batch(target, prediction)


@pytest.mark.parametrize(
    "metric, input_shape, output_shape",
    [
        (torch.mean, (1,), ()),
        (torch.mean, (1, 2, 3), ()),
        (metrics_and_maths.compute_zonal_power_spectrum, (2, 1, 16, 32), (2, 1, 17)),
    ],
)
def test_mean_shapes(metric, input_shape, output_shape):
    aggregator = Mean(metric)
    n_batches = 2
    for _ in range(n_batches):
        aggregator.record_batch({"x": torch.ones(*input_shape)})
    result = aggregator.get()
    assert result["x"].shape == output_shape


def test_snapshot_runs():
    variable_metadata = {
        "x": VariableMetadata("foo/sec", "bary bar bar"),
        "y": VariableMetadata("bar/m", "fooey foo"),
    }
    snapshot = SnapshotAggregator(["height", "width"], variable_metadata)
    batch_size, height, width = 2, 4, 8

    target = {
        "x": torch.rand(batch_size, height, width),
        "y": torch.rand(batch_size, height, width),
    }
    prediction = {
        "x": torch.rand(batch_size, height, width),
        "y": torch.rand(batch_size, height, width),
    }
    snapshot.record_batch(target, prediction)
    snapshot._get()


@pytest.mark.parametrize("n_steps", (1, 2))
def test_map_aggregator(n_steps: int):
    batch_size, height, width = 3, 4, 5
    aggregator = MeanMapAggregator()
    for _ in range(n_steps):
        target = {
            "x": torch.rand(batch_size, height, width),
            "y": torch.rand(batch_size, height, width),
        }
        prediction = {
            "x": torch.rand(batch_size, height, width),
            "y": torch.rand(batch_size, height, width),
        }
        aggregator.record_batch(target, prediction)

    _, maps = aggregator._get_metrics_and_maps()
    for var_name in ("x", "y"):
        assert maps[f"maps/full-field/{var_name}"].shape == (
            height,
            width * 2 + aggregator.gap_width,
        )
        assert maps[f"maps/error/{var_name}"].shape == (height, width)

    aggregator.get_wandb()


@pytest.mark.parametrize("n_latent_steps", [0, 2])
def test_aggregator_integration(n_latent_steps, percentiles=[99.999]):
    downscale_factor = 2
    n_batch = 2
    n_samples = 3
    n_lat, n_lon = 16, 32
    coarse_n_lat = n_lat // downscale_factor
    coarse_n_lon = n_lon // downscale_factor
    fine_shape = (n_batch, n_lat, n_lon)
    coarse_shape = (n_batch, coarse_n_lat, coarse_n_lon)
    n_bins = 300
    prediction = {"x": torch.ones(*fine_shape, device=get_device())}
    target = {"x": torch.ones(*fine_shape, device=get_device())}
    coarse = {"x": torch.ones(*coarse_shape, device=get_device())}
    fine_coordinates = BatchedLatLonCoordinates(
        lat=torch.randn(n_batch, n_lat),
        lon=torch.randn(n_batch, n_lon),
        dims=["batch", "lat", "lon"],
    ).to_device()
    coarse_coordinates = BatchedLatLonCoordinates(
        lat=torch.randn(n_batch, coarse_n_lat),
        lon=torch.randn(n_batch, coarse_n_lon),
        dims=["batch", "lat", "lon"],
    ).to_device()
    time = xr.DataArray(torch.zeros(n_batch), dims=["batch"])
    batch = PairedBatchData(
        fine=BatchData(target, time, fine_coordinates),
        coarse=BatchData(coarse, time, coarse_coordinates),
    )
    # latent steps should include an output channel dimension since latent
    # contains stacked outputs, first dim is interleaved batch/samples
    latent_steps = [
        torch.zeros(n_batch * n_samples, 1, n_lat, n_lon) for _ in range(n_latent_steps)
    ]

    # Add sample dimension to fields for aggregator
    sample_dim = 1
    prediction = {
        k: v.unsqueeze(sample_dim).expand(-1, n_samples, -1, -1)
        for k, v in prediction.items()
    }
    target = {k: v.unsqueeze(sample_dim) for k, v in target.items()}
    coarse = {k: v.unsqueeze(sample_dim) for k, v in coarse.items()}

    with mock_wandb():
        aggregator = GenerationAggregator(
            ["lat", "lon"],
            downscale_factor,
            n_histogram_bins=n_bins,
            percentiles=percentiles,
        )
        aggregator.record_batch(
            outputs=ModelOutputs(
                prediction=prediction,
                target=target,
                latent_steps=latent_steps,
                loss=torch.tensor(0.0),
            ),
            coarse=coarse,
            batch=batch,
        )
        aggregator.get_wandb(prefix="test")
        aggregator.get_dataset()


@pytest.mark.parametrize("valid_shape", [(2, 4, 8), (2, 3, 4, 8)])
def test_check_batch_dims_valid_cases(valid_shape):
    outputs = ModelOutputs(
        target={"x": torch.zeros(*valid_shape)},
        prediction={"x": torch.zeros(*valid_shape)},
        latent_steps=[],
        loss=torch.tensor(0.0),
    )
    coarse = {"x": torch.zeros(*valid_shape)}
    _check_batch_dims_for_recording(outputs, coarse, len(valid_shape))


@pytest.mark.parametrize(
    "outputs, coarse",
    [
        pytest.param(
            ModelOutputs(
                target={"x": torch.zeros(2, 3, 4, 8)},
                prediction={"x": torch.zeros(2, 4, 8)},
                latent_steps=[],
                loss=torch.tensor(0.0),
            ),
            {"x": torch.zeros(2, 4, 8)},
            id="invalid_target_dims",
        ),
        pytest.param(
            ModelOutputs(
                target={"x": torch.zeros(2, 4, 8)},
                prediction={"x": torch.zeros(2, 3, 4, 8)},
                latent_steps=[],
                loss=torch.tensor(0.0),
            ),
            {"x": torch.zeros(2, 4, 8)},
            id="invalid_prediction_dims",
        ),
        pytest.param(
            ModelOutputs(
                target={"x": torch.zeros(2, 4, 8)},
                prediction={"x": torch.zeros(2, 4, 8)},
                latent_steps=[],
                loss=torch.tensor(0.0),
            ),
            {"x": torch.zeros(2, 3, 4, 8)},
            id="invalid_coarse_dims",
        ),
    ],
)
def test_check_batch_dims_for_recording_invalid_cases(outputs, coarse):
    with pytest.raises(ValueError):
        _check_batch_dims_for_recording(outputs, coarse, 3)


def test_check_all_datasets_compatible_sample_dim():
    prediction = {"x": torch.zeros(4, 2, 8, 16)}
    target = {"x": torch.zeros(4, 1, 8, 16)}

    # Invalid case with 3 samples (incompatible with tensor1)
    invalid_samples = {"x": torch.zeros(4, 3, 8, 16)}

    # invalid case with only 3 dimensions
    invalid_dims = {"x": torch.zeros(4, 8, 16)}

    n_samples = _check_all_datasets_compatible_sample_dim([prediction, target])
    assert n_samples == 2

    with pytest.raises(ValueError):
        _check_all_datasets_compatible_sample_dim([prediction, target, invalid_samples])

    with pytest.raises(ValueError):
        _check_all_datasets_compatible_sample_dim([prediction, invalid_dims])
