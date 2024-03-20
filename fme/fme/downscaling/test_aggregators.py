import contextlib

import pytest
import torch

from fme.core import metrics
from fme.core.data_loading.data_typing import VariableMetadata
from fme.core.testing.wandb import mock_wandb
from fme.core.typing_ import TensorMapping
from fme.core.wandb import Image
from fme.downscaling import metrics_and_maths
from fme.downscaling.aggregators import (
    Aggregator,
    DynamicHistogram,
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


@pytest.mark.parametrize(
    "shape,err",
    [
        pytest.param((2, 8, 16), contextlib.nullcontext(), id="no_time_dim"),
        pytest.param((2, 1, 8, 16), pytest.raises(AssertionError), id="no_time_dim"),
    ],
)
@pytest.mark.parametrize("percentiles", [[], [99.0], [99.0, 99.99]])
def test_dynamic_histogram(shape, err, percentiles):
    n_bins = 300
    histogram = DynamicHistogram(n_bins, percentiles=percentiles)
    target = {"x": torch.ones(*shape), "y": torch.zeros(*shape)}
    prediction = {"x": torch.rand(*shape), "y": torch.rand(*shape)}
    with err:
        histogram.record_batch(target, prediction)
        result = histogram.get()

        percentile_names = []
        for p in percentiles:
            for data_type in ("target", "prediction"):
                for var_name in ("x", "y"):
                    percentile_names.append(f"{data_type}/{p}th-percentile/{var_name}")

        assert sorted(list(result.keys())) == sorted(
            [
                "prediction/x",
                "prediction/y",
                "target/x",
                "target/y",
            ]
            + percentile_names
        )
        for var_name in ["x", "y"]:
            for data_type in ["target", "prediction"]:
                counts, bin_edges = result[f"{data_type}/{var_name}"]
                assert counts.shape == (n_bins,)
                assert bin_edges.shape == (n_bins + 1,)
                for p in percentiles:
                    assert (
                        result[f"{data_type}/{p}th-percentile/{var_name}"].shape == ()
                    )


@pytest.mark.parametrize("n_steps", (1, 2))
def test_map_aggregator_shape(n_steps: int):
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


@pytest.mark.parametrize(
    "prefix, expected_prefix",
    [
        pytest.param("", "", id="no_prefix"),
        pytest.param("foo", "foo/", id="prefix=foo"),
    ],
)
def test_performance_metrics(prefix, expected_prefix, percentiles=[99.999]):
    shape = (2, 16, 32)
    _, n_lat, n_lon = shape
    area_weights = torch.ones(n_lon)
    latitudes = torch.linspace(-89.5, 89.5, n_lat)
    n_bins = 300
    target = {"x": torch.zeros(*shape)}
    prediction = {"x": torch.ones(*shape)}

    with mock_wandb():
        aggregator = Aggregator(
            area_weights, latitudes, n_histogram_bins=n_bins, percentiles=percentiles
        )
        aggregator.record_batch(torch.tensor(0.0), target, prediction)
        all_metrics = aggregator.get(prefix=prefix)
        wandb_metrics = aggregator.get_wandb(prefix=prefix)

    gap_width = aggregator._comparisons["time_mean_map"].gap_width  # type: ignore
    assert f"{expected_prefix}loss" in all_metrics
    assert f"{expected_prefix}loss" in wandb_metrics
    num_metrics = 1  # loss
    percentile_names_and_shapes = [
        (f"histogram/{data_type}/{p}th-percentile", ())
        for p in percentiles
        for data_type in ("target", "prediction")
    ]
    for metric_name, expected_shape in [
        ("rmse", ()),
        ("weighted_rmse", ()),
        ("psnr", ()),
        ("ssim", ()),
        ("time_mean_map/error", (n_lat, n_lon)),
        (
            "time_mean_map/full-field",
            (n_lat, 2 * n_lon + gap_width),
        ),
        ("histogram/target", (n_bins,)),
        ("histogram/prediction", (n_bins,)),
    ] + percentile_names_and_shapes:
        num_metrics += 1

        key = f"{expected_prefix}{metric_name}/x"

        assert key in all_metrics

        if "histogram" in key and "percentile" not in key:
            wandb_key = f"{expected_prefix}histogram/x"
            counts, _ = all_metrics[key]
            shape = counts.shape
        else:
            wandb_key = key
            shape = all_metrics[key].shape
        assert wandb_key in wandb_metrics
        assert shape == expected_shape

    for metric_name in [
        "snapshot/image-error",
        "snapshot/image-full-field",
    ]:
        num_metrics += 1
        assert isinstance(all_metrics[f"{expected_prefix}{metric_name}/x"], Image)

    for data_type in ["target", "prediction"]:
        num_metrics += 1
        key = f"{expected_prefix}spectrum/x_{data_type}"
        assert key in all_metrics
        assert key in wandb_metrics
        assert all_metrics[key].shape == (n_lon // 2 + 1,)

    # in wandb target and prediction histograms are plotted on the same figure
    assert len(all_metrics) == len(wandb_metrics) + 1 == num_metrics
