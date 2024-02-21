import contextlib

import pytest
import torch

from fme.core import metrics
from fme.core.testing.wandb import mock_wandb
from fme.core.typing_ import TensorMapping
from fme.downscaling import metrics_and_maths
from fme.downscaling.aggregators import Aggregator, DynamicHistogram, Mean, Snapshot


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


@pytest.mark.parametrize(
    "shape,err",
    [
        pytest.param((2, 8, 16), contextlib.nullcontext(), id="no_time_dim"),
        pytest.param((2, 1, 8, 16), pytest.raises(AssertionError), id="no_time_dim"),
    ],
)
def test_dynamic_histogram(shape, err):
    n_bins = 300
    histogram = DynamicHistogram(n_bins)
    data = {"x": torch.ones(*shape), "y": torch.zeros(*shape)}
    with err:
        histogram.record_batch(data)
        result = histogram.get()
        assert sorted(list(result.keys())) == ["x", "y"]
        for var_name in ["x", "y"]:
            counts, bin_edges = result[var_name]
            assert counts.shape == (n_bins,)
            assert bin_edges.shape == (n_bins + 1,)


@pytest.mark.parametrize(
    "prefix, expected_prefix",
    [
        pytest.param("", "", id="no_prefix"),
        pytest.param("foo", "foo/", id="prefix=foo"),
    ],
)
def test_performance_metrics(prefix, expected_prefix):
    shape = (2, 16, 32)
    _, n_lat, n_lon = shape
    area_weights = torch.ones(n_lon)
    latitudes = torch.linspace(-89.5, 89.5, n_lat)
    n_bins = 300
    target = {"x": torch.zeros(*shape)}
    prediction = {"x": torch.ones(*shape)}

    with mock_wandb():
        aggregator = Aggregator(area_weights, latitudes, n_histogram_bins=n_bins)
        aggregator.record_batch(torch.tensor(0.0), target, prediction)
        all_metrics = aggregator.get(prefix=prefix)
        wandb_metrics = aggregator.get_wandb(prefix=prefix)

    assert f"{expected_prefix}loss" in all_metrics
    assert f"{expected_prefix}loss" in wandb_metrics
    num_metrics = 1  # loss
    for metric_name in [
        "rmse",
        "weighted_rmse",
        "psnr",
        "ssim",
    ]:
        num_metrics += 1
        assert (
            f"{expected_prefix}{metric_name}/x" in all_metrics
        ), f"{expected_prefix}{metric_name}/x, {all_metrics.keys()}"
        assert f"{expected_prefix}{metric_name}/x" in wandb_metrics
        assert (
            all_metrics[f"{expected_prefix}{metric_name}/x"].shape == ()
        ), f"{metric_name}/x"

    expected_shapes = ((n_lon // 2 + 1,), shape, (n_bins,))
    for instrinsic_name, expected_shape in zip(
        ("spectrum", "snapshot", "histogram"), expected_shapes
    ):
        for input_type in ["target", "prediction"]:
            num_metrics += 1
            key = f"{expected_prefix}{instrinsic_name}/x_{input_type}"
            assert key in all_metrics
            assert key in wandb_metrics
            value = all_metrics[key]
            if instrinsic_name == "histogram":
                counts, _ = value
                shape = counts.shape
            else:
                shape = value.shape

            assert shape == expected_shape

    assert len(all_metrics) == len(wandb_metrics) == num_metrics


if __name__ == "__main__":
    pytest.main()
