import time

import matplotlib.figure
import numpy as np
import pytest
import torch

from fme.core.histogram import (
    ComparedDynamicHistograms,
    ComparedDynamicTailsHistograms,
    DynamicHistogram,
    DynamicHistogramAggregator,
    _normalize_histogram,
    histogram,
    trim_zero_bins,
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


@pytest.mark.parametrize("n_series", [1, 3])
def test_dynamic_histogram_random_values(n_series: int):
    np.random.seed(0)
    n_values = 50_000
    n_window = 100
    # gradually increasing values ensure bins get expanded, with a different
    # range for each series so that the bins must span the range of all series
    data = (
        np.random.uniform(low=0, high=1, size=(n_series, n_values))
        * np.arange(1, n_values + 1)[None, :]
        * np.arange(1, n_series + 1)[:, None]
    )
    histogram = DynamicHistogram(n_series=n_series, n_bins=20)
    start = time.time()
    for i in range(0, n_values, n_window):
        histogram.add(
            [
                torch.as_tensor(data[i_series, i : i + n_window])
                for i_series in range(n_series)
            ]
        )
    end = time.time()
    assert (end - start) < 0.8, "histogram too slow"
    assert np.sum(histogram.counts) == n_series * n_values
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
    histogram = DynamicHistogram(n_series=1, n_bins=200)
    histogram.add([torch.as_tensor([-1.0, 0.0, 1.0])])
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    np.testing.assert_approx_equal(bin_edges[0], -1.0, significant=6)
    np.testing.assert_approx_equal(bin_edges[-1], 1.0, significant=6)
    histogram.add([torch.as_tensor([-2.0])])
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    # double in size to the left, length becomes 4, from -3 to 1.0
    np.testing.assert_approx_equal(bin_edges[0], -3.0, significant=6)
    np.testing.assert_approx_equal(bin_edges[-1], 1.0, significant=6)
    histogram.add([torch.as_tensor([2.0])])
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    # double in size to the right, length becomes 8, from -3 to 5.0
    np.testing.assert_approx_equal(bin_edges[0], -3.0, significant=6)
    np.testing.assert_approx_equal(bin_edges[-1], 5.0, significant=6)
    histogram.add([torch.as_tensor([27.0])])
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    # double in size twice to the right, length becomes 32, from -3 to 29.0
    np.testing.assert_approx_equal(bin_edges[0], -3.0, significant=6)
    np.testing.assert_approx_equal(bin_edges[-1], 29.0, significant=6)


def test_dynamic_histogram_initial_bins_span_all_series():
    histogram = DynamicHistogram(n_series=2, n_bins=200)
    histogram.add([torch.as_tensor([0.0, 1.0]), torch.as_tensor([-3.0, 5.0, 2.0])])
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    # no doubling is needed to fit the second series
    np.testing.assert_approx_equal(bin_edges[0], -3.0, significant=5)
    np.testing.assert_approx_equal(bin_edges[-1], 5.0, significant=5)
    np.testing.assert_array_equal(histogram.counts.sum(axis=1), [2, 3])


def test_dynamic_histogram_extends_all_series_together():
    histogram = DynamicHistogram(n_series=2, n_bins=200)
    histogram.add([torch.as_tensor([-1.0, 1.0]), torch.as_tensor([-1.0, 1.0])])
    # only the second series has a value outside of the existing bins
    histogram.add([torch.as_tensor([0.5]), torch.as_tensor([2.0])])
    bin_edges = histogram.bin_edges
    assert bin_edges is not None
    np.testing.assert_approx_equal(bin_edges[-1], 3.0, significant=5)
    for i_series, values in enumerate([[-1.0, 1.0, 0.5], [-1.0, 1.0, 2.0]]):
        np.testing.assert_array_equal(
            histogram.counts[i_series], np.histogram(values, bins=bin_edges)[0]
        )


def test_dynamic_histogram_wrong_number_of_series():
    histogram = DynamicHistogram(n_series=2, n_bins=200)
    with pytest.raises(ValueError, match="2 series"):
        histogram.add([torch.as_tensor([1.0])])


def test_dynamic_histogram_odd_n_bins():
    with pytest.raises(ValueError, match="even"):
        DynamicHistogram(n_series=1, n_bins=201)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_dynamic_histogram_ignores_non_finite_values(bad_value: float):
    values = torch.as_tensor([0.0, 1.0, 2.0])
    other = torch.as_tensor([0.5, 1.5])
    expected = DynamicHistogram(n_series=2, n_bins=200)
    expected.add([values, other])
    histogram = DynamicHistogram(n_series=2, n_bins=200)
    # non-finite values in one series don't affect it or the other series
    histogram.add([values, torch.as_tensor([0.5, bad_value, 1.5])])
    # a series with no finite values is skipped
    histogram.add([torch.as_tensor([]), torch.as_tensor([bad_value])])
    assert histogram.bin_edges is not None
    np.testing.assert_array_equal(histogram.bin_edges, expected.bin_edges)
    np.testing.assert_array_equal(histogram.counts, expected.counts)


def test_dynamic_histogram_range_larger_than_float32_max():
    # values of both signs give bins with a range that can't be represented
    # in float32, as can happen when a prediction diverges
    histogram = DynamicHistogram(n_series=2, n_bins=200)
    target = 250.0 + 50.0 * torch.rand(64, dtype=torch.float32)
    for bad_value in [-2.0e38, 2.0e38, 3.0e38]:
        prediction = target.clone()
        prediction[0] = bad_value
        histogram.add([target, prediction])
    assert histogram.bin_edges is not None
    assert histogram.bin_edges[-1] - histogram.bin_edges[0] > 3.5e38
    np.testing.assert_array_equal(histogram.counts.sum(axis=1), [3 * 64, 3 * 64])
    np.testing.assert_array_equal(
        histogram.counts[1],
        np.histogram(
            [-2.0e38, 2.0e38, 3.0e38] + [275.0] * (3 * 63), bins=histogram.bin_edges
        )[0],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_histogram_dtypes(dtype: torch.dtype):
    # the bin of this value is not exactly representable in half precision
    values = torch.as_tensor([1000.0]).to(dtype)
    counts = histogram(values, vmin=0.3, bin_size=0.1, n_bins=20000)
    assert counts.sum() == 1
    assert counts[9997] == 1


def test_histogram_counts_values_outside_of_bins_in_end_bins():
    values = torch.as_tensor([-5.0, 0.5, 20.0, 30.0])
    counts = histogram(values, vmin=0.0, bin_size=1.0, n_bins=10)
    np.testing.assert_array_equal(counts, [2, 0, 0, 0, 0, 0, 0, 0, 0, 2])


def test_dynamic_histogram_no_finite_values():
    histogram = DynamicHistogram(n_series=1, n_bins=200)
    histogram.add([torch.as_tensor([float("nan")])])
    assert histogram.bin_edges is None
    assert np.all(np.isnan(histogram.get_bin_edges()))
    assert np.all(histogram.counts == 0)


@pytest.mark.parametrize("value", [1.0, 300.0, 1.0e5, 1.0e12])
def test_histogram_handles_uniform_field(value: float):
    histogram = DynamicHistogram(n_series=1, n_bins=200)
    # has zero range, including after rounding of float32 values
    histogram.add([torch.as_tensor([value, value, value], dtype=torch.float32)])
    histogram.add([torch.as_tensor([value, 2 * value, 3 * value])])  # non-zero range
    assert histogram.counts.sum() == 6


def test_trim_zero_bins_trims_series_together():
    counts = np.array([[0, 0, 1, 2, 0, 0], [0, 3, 0, 0, 0, 0]])
    bin_edges = np.arange(7.0)
    trimmed_counts, trimmed_edges = trim_zero_bins(counts, bin_edges)
    np.testing.assert_array_equal(trimmed_counts, [[0, 1, 2], [3, 0, 0]])
    np.testing.assert_array_equal(trimmed_edges, [1.0, 2.0, 3.0, 4.0])


def test_compared_dynamic_histograms_share_bins():
    torch.manual_seed(0)
    n_bins = 300
    histogram = ComparedDynamicHistograms(n_bins, percentiles=[99.0])
    shape = (2, 3, 8, 16)
    targets, predictions = [], []
    for scale in [1.0, 3.0]:  # bins must be extended for the second batch
        # the prediction has a wider range of values than the target
        # float64 so that values are binned identically to np.histogram
        targets.append(scale * torch.rand(*shape, dtype=torch.float64))
        predictions.append(10.0 * scale * torch.rand(*shape, dtype=torch.float64) - 5.0)
        histogram.record_batch({"x": targets[-1]}, {"x": predictions[-1]})

    histograms = histogram._get_histograms()["x"]
    bin_edges = histograms["target"].bin_edges
    np.testing.assert_array_equal(histograms["prediction"].bin_edges, bin_edges)
    for source_histogram, values in (
        (histograms["target"], targets),
        (histograms["prediction"], predictions),
    ):
        np.testing.assert_array_equal(
            source_histogram.counts,
            np.histogram(torch.cat(values).numpy(), bins=bin_edges)[0],
        )

    wandb_result = histogram.get_wandb()
    bin_width = bin_edges[1] - bin_edges[0]
    for source, values in (("target", targets), ("prediction", predictions)):
        assert wandb_result[f"{source}/99.0th-percentile/x"] == pytest.approx(
            np.percentile(torch.cat(values).numpy(), 99.0), abs=bin_width
        )


def test_compared_dynamic_histograms_dataset():
    torch.manual_seed(0)
    n_bins = 300
    histogram = ComparedDynamicHistograms(n_bins, percentiles=[99.0])
    shape = (2, 8, 16)
    # float64 so that values are binned identically to np.histogram
    target = {"x": torch.rand(*shape, dtype=torch.float64)}
    prediction = {"x": 2.0 * torch.rand(*shape, dtype=torch.float64)}
    histogram.record_batch(target, prediction)
    ds = histogram.get_dataset()
    assert list(ds.coords["source"].values) == ["target", "prediction"]
    assert ds["x"].dims == ("source", "bin")
    assert ds["x"].shape == (2, n_bins)
    assert ds["x_bin_edges"].dims == ("source", "bin_edges")
    assert ds["x_bin_edges"].shape == (2, n_bins + 1)
    np.testing.assert_array_equal(
        ds["x_bin_edges"].sel(source="target").values,
        ds["x_bin_edges"].sel(source="prediction").values,
    )
    for source, data in (("target", target), ("prediction", prediction)):
        np.testing.assert_array_equal(
            ds["x"].sel(source=source).values,
            np.histogram(
                data["x"].numpy(),
                bins=ds["x_bin_edges"].sel(source=source).values,
            )[0],
        )


def test_compared_dynamic_histograms_non_finite_prediction():
    torch.manual_seed(0)
    n_bins = 300
    shape = (2, 8, 16)
    target = {"x": torch.rand(*shape)}
    prediction = {"x": torch.rand(*shape)}
    bad_prediction = {"x": prediction["x"].clone()}
    bad_prediction["x"][0, 0, 0] = float("inf")
    bad_prediction["x"][1, 2, 3] = float("nan")

    expected = ComparedDynamicHistograms(n_bins, percentiles=[99.0])
    expected.record_batch(target, prediction)
    histogram = ComparedDynamicHistograms(n_bins, percentiles=[99.0])
    histogram.record_batch(target, bad_prediction)

    ds, expected_ds = histogram.get_dataset(), expected.get_dataset()
    np.testing.assert_array_equal(ds["x_bin_edges"], expected_ds["x_bin_edges"])
    np.testing.assert_array_equal(
        ds["x"].sel(source="target"), expected_ds["x"].sel(source="target")
    )
    assert ds["x"].sel(source="prediction").sum() == prediction["x"].numel() - 2
    wandb_result = histogram.get_wandb()
    assert wandb_result["target/99.0th-percentile/x"] == pytest.approx(
        expected.get_wandb()["target/99.0th-percentile/x"]
    )
    assert np.isfinite(wandb_result["prediction/99.0th-percentile/x"])


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
    assert list(ds.coords["source"].values) == ["target", "prediction"]


def test_compared_dynamic_histograms_prediction_frac_is_prediction_over_target():
    n_bins = 300
    histogram = ComparedDynamicHistograms(
        n_bins,
        percentiles=[99.0],
        compute_percentile_frac=True,
    )
    shape = (2, 8, 16)
    target = {"x": torch.full(shape, 2.0)}
    prediction = {"x": torch.full(shape, 4.0)}
    histogram.record_batch(target, prediction)
    wandb_result = histogram.get_wandb()

    # percentiles are only accurate to within the width of a bin
    assert wandb_result[
        "prediction_frac_of_target/99.0th-percentile/x"
    ] == pytest.approx(2.0, rel=0.01)


@pytest.mark.parametrize(
    "target, prediction",
    [
        pytest.param(
            {"x": torch.ones(2, 8, 16), "y": torch.zeros(2, 8, 16)},
            {},
            id="non_empty_target_empty_prediction",
        ),
        pytest.param(
            {},
            {"x": torch.rand(2, 8, 16), "y": torch.rand(2, 8, 16)},
            id="empty_target_non_empty_prediction",
        ),
        pytest.param({}, {}, id="empty_target_empty_prediction"),
    ],
)
def test_compared_dynamic_histogram_empty_tensor_mapping(target, prediction):
    """
    Test that histogram aggregator handles empty tensor mappings.
    """
    n_bins = 300
    histogram = ComparedDynamicHistograms(n_bins, percentiles=[99.0])

    with pytest.raises(ValueError):
        histogram.record_batch(target, prediction)


def test_compared_dynamic_histogram_varying_variables_with_record_batch():
    n_bins = 300
    histogram = ComparedDynamicHistograms(n_bins, percentiles=[99.0])
    target = {"x": torch.ones(2, 8, 16), "y": torch.zeros(2, 8, 16)}
    prediction = {"x": torch.rand(2, 8, 16), "y": torch.rand(2, 8, 16)}

    # record one batch to set the recorded variables
    histogram.record_batch(target, prediction)

    target_missing_y = {"x": torch.ones(2, 8, 16)}
    prediction_missing_y = {"x": torch.rand(2, 8, 16)}
    target_extra_z = {**target, "z": torch.ones(2, 8, 16)}
    prediction_extra_w = {**prediction, "w": torch.rand(2, 8, 16)}

    # fails on variables missing from inital recorded batch
    with pytest.raises(ValueError):
        histogram.record_batch(target_missing_y, prediction)
    with pytest.raises(ValueError):
        histogram.record_batch(target, prediction_missing_y)

    # ignores non-matching variables
    histogram.record_batch(target_extra_z, prediction_extra_w)


def test_compared_dynamic_histogram_with_nans_record_batch():
    n_bins = 300
    histogram = ComparedDynamicHistograms(n_bins, percentiles=[99.0])

    # record one batch
    target = {"x": torch.ones(2, 2, 8, 16), "y": torch.zeros(2, 2, 8, 16)}
    target["x"][0, 0, 0] = float("nan")
    prediction = {"x": torch.rand(2, 2, 8, 16), "y": torch.rand(2, 2, 8, 16)}
    prediction["x"][0, 0, 0] = float("nan")

    histogram.record_batch(target, prediction)
    wandb_result = histogram.get_wandb()
    assert not np.isnan(wandb_result["target/99.0th-percentile/x"])
    assert not np.isnan(wandb_result["prediction/99.0th-percentile/x"])

    # record another batch with different time dimension size
    target = {"x": torch.ones(2, 1, 8, 16), "y": torch.zeros(2, 1, 8, 16)}
    target["x"][0, 0, 0] = float("nan")
    prediction = {"x": torch.rand(2, 1, 8, 16), "y": torch.rand(2, 1, 8, 16)}
    prediction["x"][0, 0, 0] = float("nan")

    histogram.record_batch(target, prediction)
    wandb_result = histogram.get_wandb()
    assert not np.isnan(wandb_result["target/99.0th-percentile/x"])
    assert not np.isnan(wandb_result["prediction/99.0th-percentile/x"])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 8, 16), id="no_time_dim"),
        pytest.param((2, 1, 8, 16), id="time_dim"),
    ],
)
@pytest.mark.parametrize("percentiles", [[], [99.0], [99.0, 99.99]])
def test_dynamic_histogram_aggregator(shape, percentiles):
    n_bins = 300
    histogram = DynamicHistogramAggregator(n_bins, percentiles=percentiles)
    prediction = {"x": torch.rand(*shape), "y": torch.rand(*shape)}
    histogram.record_batch(prediction)
    wandb_result = histogram.get_wandb()

    percentile_names = []
    for p in percentiles:
        for var_name in ("x", "y"):
            percentile_names.append(f"{p}th-percentile/{var_name}")

    assert sorted(list(wandb_result.keys())) == sorted(
        [
            "x",
            "y",
        ]
        + percentile_names
    )
    for var_name in ["x", "y"]:
        assert isinstance(wandb_result[f"{var_name}"], matplotlib.figure.Figure)
        for p in percentiles:
            assert isinstance(wandb_result[f"{p}th-percentile/{var_name}"], float)

    ds = histogram.get_dataset()
    for var in ["x", "y"]:
        assert var in ds
        assert f"{var}_bin_edges" in ds
        assert len(ds[var]) == n_bins


def test_compared_dynamic_tails_histograms():
    n_bins = 300
    histogram = ComparedDynamicTailsHistograms(
        n_bins,
        percentiles=[99.9999],
        two_tailed_variables=["y"],
        left_tailed_variables=["z"],
    )
    shape = (2, 8, 16)
    target = {
        "x": torch.ones(*shape),
        "y": torch.zeros(*shape),
        "z": torch.zeros(*shape),
    }
    prediction = {
        "x": torch.rand(*shape),
        "y": torch.rand(*shape),
        "z": torch.rand(*shape),
    }
    histogram.record_batch(target, prediction)
    wandb_result = histogram.get_wandb()

    for var in ["x", "y"]:
        assert f"99.9999th-percentile/{var}" in wandb_result
        assert f"prediction_frac_of_target/99.9999th-percentile/{var}" in wandb_result
    for var in ["y", "z"]:
        assert f"0.0001th-percentile/{var}" in wandb_result
        assert f"prediction_frac_of_target/0.0001th-percentile/{var}" in wandb_result


def test_compared_dynamic_tails_histograms_share_bins():
    histogram = ComparedDynamicTailsHistograms(n_bins=300, percentiles=[99.0])
    shape = (2, 8, 16)
    histogram.record_batch({"x": torch.rand(*shape)}, {"x": 5.0 * torch.rand(*shape)})
    histograms = histogram._get_histograms()["x"]
    np.testing.assert_array_equal(
        histograms["target"].bin_edges, histograms["prediction"].bin_edges
    )
    ds = histogram.get_dataset()
    np.testing.assert_array_equal(
        ds["x_bin_edges"].sel(source="target"),
        ds["x_bin_edges"].sel(source="prediction"),
    )


def test_compared_dynamic_tails_histograms_prediction_frac_is_prediction_over_target():
    n_bins = 300
    histogram = ComparedDynamicTailsHistograms(
        n_bins,
        percentiles=[99.0],
        left_tailed_variables=["lower"],
    )
    shape = (2, 8, 16)
    target = {
        "upper": torch.full(shape, 2.0),
        "lower": torch.full(shape, 2.0),
    }
    prediction = {
        "upper": torch.full(shape, 4.0),
        "lower": torch.full(shape, 4.0),
    }
    histogram.record_batch(target, prediction)
    wandb_result = histogram.get_wandb()

    # percentiles are only accurate to within the width of a bin
    assert wandb_result[
        "prediction_frac_of_target/99.0th-percentile/upper"
    ] == pytest.approx(2.0, rel=0.01)
    assert wandb_result[
        "prediction_frac_of_target/1.0th-percentile/lower"
    ] == pytest.approx(2.0, rel=0.01)
