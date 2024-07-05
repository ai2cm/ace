import pytest
import torch

from fme.core import metrics
from fme.core.data_loading.data_typing import VariableMetadata
from fme.core.testing.wandb import mock_wandb
from fme.core.typing_ import TensorMapping
from fme.downscaling import metrics_and_maths
from fme.downscaling.aggregators import (
    Aggregator,
    Mean,
    MeanComparison,
    MeanMapAggregator,
    SnapshotAggregator,
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
    assert_tensor_mapping_all_close(result, expected_outputs)


def _compute_zonal_spectrum_for_testing(x):
    """Hard codes latitudes."""
    lats = torch.linspace(-89.5, 89.5, x.shape[-2])
    return metrics_and_maths.compute_zonal_power_spectrum(x, lats)


@pytest.mark.parametrize(
    "metric, input_shape, output_shape",
    [
        (torch.mean, (1,), ()),
        (torch.mean, (1, 2, 3), ()),
        (_compute_zonal_spectrum_for_testing, (2, 1, 16, 32), (2, 1, 17)),
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
    metadata = {
        "x": VariableMetadata("foo/sec", "bary bar bar"),
        "y": VariableMetadata("bar/m", "fooey foo"),
    }
    snapshot = SnapshotAggregator(metadata)
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
    snapshot.get()


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

    values = aggregator.get()
    for var_name in ("x", "y"):
        assert values[f"full-field/{var_name}"].shape == (
            height,
            width * 2 + aggregator.gap_width,
        )
        assert values[f"error/{var_name}"].shape == (height, width)

    aggregator.get_wandb()
    ds = aggregator.get_dataset()
    all(ds.coords["source"] == ["target", "prediction"])


@pytest.mark.parametrize(
    "prefix, expected_prefix",
    [
        pytest.param("", "", id="no_prefix"),
        pytest.param("foo", "foo/", id="prefix=foo"),
    ],
)
def test_performance_metrics(prefix, expected_prefix, percentiles=[99.999]):
    downscale_factor = 2
    shape = (2, 16, 32)
    _, n_lat, n_lon = shape
    area_weights = torch.ones(n_lon)
    latitudes = torch.linspace(-89.5, 89.5, n_lat)
    n_bins = 300
    target = {"x": torch.zeros(*shape)}
    prediction = {"x": torch.ones(*shape)}
    coarse = {
        "x": torch.ones(2, shape[1] // downscale_factor, shape[2] // downscale_factor)
    }

    with mock_wandb():
        aggregator = Aggregator(
            area_weights,
            latitudes,
            downscale_factor,
            n_histogram_bins=n_bins,
            percentiles=percentiles,
        )
        aggregator.record_batch(torch.tensor(0.0), target, prediction, coarse)
        wandb_metrics = aggregator.get_wandb(prefix=prefix)

    assert f"{expected_prefix}loss" in wandb_metrics
    num_metrics = 1
    percentile_names = [
        f"histogram/{data_type}/{p}th-percentile"
        for p in percentiles
        for data_type in ("target", "prediction")
    ]
    for metric_name in [
        "rmse",
        "weighted_rmse",
        "relative_mse_bicubic",
        "time_mean_map/error",
        "time_mean_map/full-field",
        "histogram/target",
        "histogram/prediction",
    ] + percentile_names:
        num_metrics += 1

        key = f"{expected_prefix}{metric_name}/x"

        if "histogram" in key and "percentile" not in key:
            wandb_key = f"{expected_prefix}histogram/x"
        else:
            wandb_key = key
        assert wandb_key in wandb_metrics

    for metric_name in [
        "snapshot/image-error",
        "snapshot/image-full-field",
    ]:
        num_metrics += 1

    for data_type in ["target", "prediction"]:
        num_metrics += 1
        key = f"{expected_prefix}spectrum/x_{data_type}"
        assert key in wandb_metrics

    # in wandb target and prediction histograms are plotted on the same figure
    assert len(wandb_metrics) + 1 == num_metrics
