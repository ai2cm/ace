import time

import matplotlib.figure
import numpy as np
import pytest
import torch

from fme.core.histogram import (
    ComparedDynamicHistograms,
    DynamicHistogram,
    _normalize_histogram,
)


def test__normalize_histogram():
    counts = np.array(
        [
            1,
            2,
            3,
            4,
        ]
    )
    bin_edges = np.array([0, 1, 2, 3, 5])
    normalized_counts = _normalize_histogram(counts, bin_edges)
    assert sum(np.diff(bin_edges) * normalized_counts) == 1.0


@pytest.mark.parametrize(
    "n_times, time_bin_len",
    [
        pytest.param(3, 3, id="one_time_bin"),
        pytest.param(6, 2, id="multiple_time_bins"),
    ],
)
def test_dynamic_histogram_random_values(n_times: int, time_bin_len: int):
    np.random.seed(0)
    n_values = 50_000
    n_window = 100
    # gradually increasing values in time ensure bins get expanded
    data = (
        np.random.uniform(low=0, high=1, size=(n_times, n_values))
        * np.arange(1, n_values + 1)[None, :]
    )
    histogram = DynamicHistogram(n_times=n_times, n_bins=20)
    start = time.time()
    for i_time in range(0, n_times, time_bin_len):
        for i in range(0, n_values, n_window):
            histogram.add(
                torch.as_tensor(data[i_time : i_time + time_bin_len, i : i + n_window]),
                i_time_start=i_time,
            )
    end = time.time()
    assert (end - start) < 0.8, "histogram too slow"
    assert np.sum(histogram.counts) == n_times * n_values
    direct_histogram = np.apply_along_axis(
        lambda arr: np.histogram(arr, bins=histogram.bin_edges)[0],
        axis=1,
        arr=data,
    )
    assert np.allclose(histogram.counts, direct_histogram)
    assert histogram.bin_edges is not None
    histogram_range = histogram.bin_edges[-1] - histogram.bin_edges[0]
    value_range = np.max(data) - np.min(data)

    # values may have been doubled to the left and then to the right,
    # and if the values were infinitesmally to the left and right of
    # the existing range when that happens, there will be 3 "empty"
    # sections to the one original "full" section:
    #     [ 0, #, 0, 0]
    # where there could be a value on the rightmost edge of the first
    # and leftmost edge of the third bin, with none outside that range.
    #
    # There should be no way to get a histogram range larger than this,
    # relative to the value range.
    assert histogram_range <= 4.0 * value_range


def test_dynamic_histogram_extends_as_expected():
    histogram = DynamicHistogram(n_times=1, n_bins=200)
    histogram.add(torch.as_tensor([[-1.0, 0.0, 1.0]]))
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    np.testing.assert_approx_equal(bin_edges[0], -1.0, significant=6)
    np.testing.assert_approx_equal(bin_edges[-1], 1.0, significant=6)
    histogram.add(torch.as_tensor([[-2.0]]))
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    # double in size to the left, length becomes 4, from -3 to 1.0
    np.testing.assert_approx_equal(bin_edges[0], -3.0, significant=6)
    np.testing.assert_approx_equal(bin_edges[-1], 1.0, significant=6)
    histogram.add(torch.as_tensor([[2.0]]))
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    # double in size to the right, length becomes 8, from -3 to 5.0
    np.testing.assert_approx_equal(bin_edges[0], -3.0, significant=6)
    np.testing.assert_approx_equal(bin_edges[-1], 5.0, significant=6)
    histogram.add(torch.as_tensor([[27.0]]))
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    # double in size twice to the right, length becomes 32, from -3 to 29.0
    np.testing.assert_approx_equal(bin_edges[0], -3.0, significant=6)
    np.testing.assert_approx_equal(bin_edges[-1], 29.0, significant=6)


def test_histogram_handles_uniform_field():
    histogram = DynamicHistogram(n_times=1, n_bins=200)
    histogram.add(torch.as_tensor([[1.0, 1.0, 1.0]]))  # has zero range
    histogram.add(torch.as_tensor([[1.0, 2.0, 3.0]]))  # has non-zero range


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 8, 16), id="no_time_dim"),
        pytest.param((2, 1, 8, 16), id="time_dim"),
    ],
)
@pytest.mark.parametrize("percentiles", [[], [99.0], [99.0, 99.99]])
def test_compared_dynamic_histograms(shape, percentiles):
    n_bins = 300
    histogram = ComparedDynamicHistograms(n_bins, percentiles=percentiles)
    target = {"x": torch.ones(*shape), "y": torch.zeros(*shape)}
    prediction = {"x": torch.rand(*shape), "y": torch.rand(*shape)}
    histogram.record_batch(target, prediction)
    wandb_result = histogram.get_wandb()

    percentile_names = []
    for p in percentiles:
        for data_type in ("target", "prediction"):
            for var_name in ("x", "y"):
                percentile_names.append(f"{data_type}/{p}th-percentile/{var_name}")

    assert sorted(list(wandb_result.keys())) == sorted(
        [
            "x",
            "y",
        ]
        + percentile_names
    )
    for var_name in ["x", "y"]:
        for data_type in ["target", "prediction"]:
            assert isinstance(wandb_result[f"{var_name}"], matplotlib.figure.Figure)
            for p in percentiles:
                assert isinstance(
                    wandb_result[f"{data_type}/{p}th-percentile/{var_name}"], float
                )

    ds = histogram.get_dataset()
    all(ds.coords["source"] == ["target", "prediction"])
