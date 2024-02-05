import pytest
import torch

import fme.core.histogram
from fme.core import metrics
from fme.core.typing_ import TensorMapping
from fme.downscaling import metrics_and_maths
from fme.downscaling.aggregators import (
    DynamicHistogram,
    InferenceAggregator,
    Mean,
    Snapshot,
)


def assert_tensor_mapping_all_close(x: TensorMapping, y: TensorMapping):
    assert x.keys() == y.keys(), f"Keys do not match: {x.keys()} != {y.keys()}"
    for k in x:
        assert x[k].shape == y[k].shape, f"Shapes do not match for {k}"
        assert torch.allclose(x[k], y[k]), f"Values do not match for {k}"


@pytest.mark.parametrize(
    "metric, input_values, expected_output_values",
    [
        (
            torch.mean,
            ({"x": torch.tensor([1.0, 2.0, 3.0])},),
            {"x": torch.tensor(2.0)},
        ),
        (
            metrics.root_mean_squared_error,
            (
                {"x": torch.tensor([1.0, 2.0, 3.0])},
                {"x": torch.tensor([1.0, 2.0, 3.0])},
            ),
            {"x": torch.tensor(0.0)},
        ),
        (
            metrics.root_mean_squared_error,
            ({"x": torch.ones(2, 1, 4, 8)}, {"x": torch.ones(2, 1, 4, 8)}),
            {"x": torch.tensor(0.0)},
        ),
    ],
)
def test_mean_values(metric, input_values, expected_output_values):
    aggregator = Mean(metric)
    n_batches = 2
    for _ in range(n_batches):
        aggregator.record_batch(*input_values)
    result = aggregator.get()
    assert_tensor_mapping_all_close(result, expected_output_values)


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


def test_snapshot_records_first_value():
    snapshot = Snapshot()
    values = {"x": torch.tensor([1, 2, 3]), "y": torch.tensor([4, 5, 6])}
    snapshot.record_batch(values)
    assert_tensor_mapping_all_close(snapshot.get(), values)
    snapshot.record_batch({"x": torch.tensor([7, 8, 9])})
    assert_tensor_mapping_all_close(snapshot.get(), values)


def test_dynamic_histogram():
    n_bins = 300
    histogram = DynamicHistogram(n_bins)
    shape = (2, 1, 8, 16)  # [batch, sample, height, width]
    data = {"x": torch.ones(*shape), "y": torch.zeros(*shape)}
    histogram.record_batch(data)
    result = histogram.get()
    assert sorted(list(result.keys())) == ["x", "y"]
    for var_name in ["x", "y"]:
        assert result[var_name].counts.shape == (1, n_bins)


def test_performance_metrics():
    shape = batch_size, n_lat, n_lon = 2, 16, 32  # no time dim for MappedTensors
    del batch_size  # unused
    area_weights = torch.ones(n_lon)
    latitudes = torch.linspace(-89.5, 89.5, n_lat)
    n_bins = 300
    aggregator = InferenceAggregator(area_weights, latitudes, n_histogram_bins=n_bins)

    target = {"x": torch.zeros(*shape)}
    pred = {"x": torch.ones(*shape)}
    aggregator.record_batch(torch.tensor(0.0), target, pred)

    all_metrics = aggregator.get()
    wandb_metrics = aggregator.get_wandb()
    num_metrics = 0
    for metric_name in [
        "rmse",
        "weighted_rmse",
        "psnr",
        "ssim",
    ]:
        num_metrics += 1
        assert f"{metric_name}/x" in all_metrics
        assert f"{metric_name}/x" in wandb_metrics
        assert all_metrics[f"{metric_name}/x"].shape == (), f"{metric_name}/x"

    for instrinsic_name, expected_shape in zip(
        ["spectrum", "snapshot", "histogram"],
        [(n_lon // 2 + 1,), shape, (1, n_bins)],
    ):
        for input_type in ["target", "pred"]:
            num_metrics += 1
            key = f"{instrinsic_name}/x_{input_type}"
            assert key in all_metrics
            assert key in wandb_metrics
            value = all_metrics[key]
            if isinstance(value, fme.core.histogram.DynamicHistogram):
                shape = value.counts.shape
            else:
                shape = value.shape
            assert shape == expected_shape

    assert len(all_metrics) == len(wandb_metrics) == num_metrics


if __name__ == "__main__":
    pytest.main()
