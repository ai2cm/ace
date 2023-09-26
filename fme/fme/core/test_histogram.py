import numpy as np

from fme.core.histogram import DynamicHistogram


def test_dynamic_histogram_random_values():
    np.random.seed(0)
    n_times = 3
    n_values = 1000
    n_window = 100
    # gradually increasing values in time ensure bins get expanded
    data = (
        np.random.uniform(low=0, high=1, size=(n_times, n_values))
        * np.arange(1, 1001)[None, :]
    )
    histogram = DynamicHistogram(n_times=n_times, n_bins=20)
    for i in range(0, n_values, n_window):
        histogram.add(data[:, i : i + n_window])
    assert np.sum(histogram.counts) == n_times * n_values
    direct_histogram = np.apply_along_axis(
        lambda arr: np.histogram(arr, bins=histogram.bin_edges)[0],
        axis=1,
        arr=data,
    )
    assert np.allclose(histogram.counts, direct_histogram)
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
    histogram.add(np.array([[-1.0, 0.0, 1.0]]))
    np.testing.assert_approx_equal(histogram.bin_edges[0], -1.0)
    np.testing.assert_approx_equal(histogram.bin_edges[-1], 1.0)
    histogram.add(np.array([[-2.0]]))
    # double in size to the left, length becomes 4, from -3 to 1.0
    np.testing.assert_approx_equal(histogram.bin_edges[0], -3.0)
    np.testing.assert_approx_equal(histogram.bin_edges[-1], 1.0)
    histogram.add(np.array([[2.0]]))
    # double in size to the right, length becomes 8, from -3 to 5.0
    np.testing.assert_approx_equal(histogram.bin_edges[0], -3.0)
    np.testing.assert_approx_equal(histogram.bin_edges[-1], 5.0)
    histogram.add(np.array([[27.0]]))
    # double in size twice to the right, length becomes 32, from -3 to 29.0
    np.testing.assert_approx_equal(histogram.bin_edges[0], -3.0)
    np.testing.assert_approx_equal(histogram.bin_edges[-1], 29.0)
