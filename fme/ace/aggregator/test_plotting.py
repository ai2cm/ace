from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pytest

from .plotting import (
    _stitch_data_panels,
    clamp_date_axis,
    fold_healpix_data,
    format_period_axis,
    get_cmap_limits,
    plot_imshow,
    plot_paneled_data,
    plot_power_spectrum_by_period,
)


def test_cmap_limits():
    data = np.array([1, 2, 3])
    vmin, vmax = get_cmap_limits(data)
    assert vmin == 1
    assert vmax == 3


def test_cmap_limits_diverging():
    data = np.array([-1, 2, 3])
    vmin, vmax = get_cmap_limits(data, diverging=True)
    assert vmin == -3
    assert vmax == 3


@pytest.mark.parametrize("use_colorbar", [True, False])
def test_plot_imshow(use_colorbar):
    shape = [10, 15]
    data = np.random.randn(*shape)
    fig = plot_imshow(np.array(data), use_colorbar=use_colorbar)
    width, height = (fig.get_size_inches() * fig.dpi).astype(int)
    if use_colorbar:
        # colorbar is no more than 15% of the width but greater than 0 pixels
        assert shape[1] < width <= int(shape[1] * 1.15)
        assert height == shape[0]
    else:
        assert [height, width] == shape


def test_fold_healpix_data():
    face_shape = [2, 3]
    data = np.random.randn(12, *face_shape)
    folded = fold_healpix_data(data, fill_value=0)
    expected_shape = (6 * face_shape[0], 4 * face_shape[1])
    assert folded.shape == expected_shape


@pytest.mark.parametrize("use_colorbar", [True, False])
def test_plot_imshow_healpix(use_colorbar):
    face_shape = [4, 6]
    shape = [6 * face_shape[0], 4 * face_shape[1]]
    data = np.random.randn(12, *face_shape)
    fig = plot_imshow(np.array(data), use_colorbar=use_colorbar)
    width, height = (fig.get_size_inches() * fig.dpi).astype(int)
    if use_colorbar:
        # colorbar is no more than 15% of the width but greater than 0 pixels
        assert shape[1] < width <= int(shape[1] * 1.15)
        assert height == shape[0]
    else:
        assert [height, width] == shape


def test_stitch_data_panels():
    data = [
        [np.array([[1, 2]]), np.array([[3, 4]])],
        [np.array([[5, 6]]), np.array([[7, 8]])],
    ]
    stitched = _stitch_data_panels(data, fill_value=1)
    expected = np.array(
        [  # vertical orientation is swapped as data starts from bottom-left
            [5, 6, 1, 7, 8],
            [1, 1, 1, 1, 1],
            [1, 2, 1, 3, 4],
        ]
    )
    assert np.array_equal(stitched, expected)


@pytest.mark.parametrize(
    "shape, img_shape",
    [
        pytest.param(
            [12, 2, 3],
            [
                27,  # 3 * 4 + 1 (divider) + 3 * 4 + 2 (colorbar)
                25,  # 2 * 6 + 1 (divider) + 2 * 6 + 2 (colorbar)
            ],
            id="healpix",
        ),
        pytest.param(
            [2, 3],
            [
                9,  # 3 + 1 (divider) + 3 + 2 (colorbar)
                5,  # 2 + 1 (divider) + 2 (colorbar)
            ],
            id="latlon",
        ),
    ],
)
def test_plot_paneled_data(shape, img_shape):
    panel = np.random.uniform(size=shape)
    data = [
        [panel, panel],
        [panel, panel],
    ]
    fig = plot_paneled_data(data, diverging=False)
    assert fig.image is not None
    assert np.array_equal(fig.image.size, img_shape)
    fig = plot_paneled_data(data, diverging=True)
    assert fig.image is not None
    assert np.array_equal(fig.image.size, img_shape)


def test_clamp_date_axis_allows_a_140_year_series_starting_at_year_0001():
    """Regression: a long rollout starting at year 0001 must still draw.

    Matplotlib rejects dates before year 0001, and autoscale pads the x limits
    ~5% beyond the data -- 7 years on a 140-year series, which on a series
    starting in year 0001 lands at year -6 and raises when the figure is drawn.
    """
    times = [datetime(year, 1, 15) for year in range(1, 141)]
    values = np.arange(len(times), dtype=float)

    fig, ax = plt.subplots(1, 1)
    try:
        ax.plot(times, values)
        clamp_date_axis(ax)
        # the failure is raised when the date locator runs, i.e. at draw time
        fig.canvas.draw()
        left, right = ax.get_xlim()
        assert left == mdates.date2num(times[0])
        assert right == mdates.date2num(times[-1])
    finally:
        plt.close(fig)


def test_plot_power_spectrum_by_period_draws_mean_and_samples():
    """A line per sample plus a heavy sample-mean line, at power per octave."""
    freqs_per_year = np.array([0.0, 0.25, 0.5, 1.0])
    power_by_sample = np.array([[10.0, 4.0, 2.0, 1.0], [20.0, 8.0, 6.0, 3.0]])

    fig, ax = plt.subplots(1, 1)
    try:
        plot_power_spectrum_by_period(ax, freqs_per_year, power_by_sample, "mean")
        assert len(ax.lines) == 3  # two samples and their mean
        mean_line = ax.lines[-1]
        assert mean_line.get_label() == "mean"
        # the zero frequency has no finite period and is dropped
        np.testing.assert_allclose(mean_line.get_xdata(), [4.0, 2.0, 1.0])
        expected = np.array([6.0, 4.0, 2.0]) * np.array([0.25, 0.5, 1.0]) * np.log(2.0)
        np.testing.assert_allclose(mean_line.get_ydata(), expected)
    finally:
        plt.close(fig)


def test_format_period_axis_labels_octaves():
    fig, ax = plt.subplots(1, 1)
    try:
        format_period_axis(ax, max_period_years=16.0)
        fig.canvas.draw()
        assert ax.get_xscale() == "log"
        assert ax.get_xlim() == (0.5, 16.0)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == ["0.5", "1.0", "2.0", "4.0", "8.0", "16.0"]
    finally:
        plt.close(fig)


def test_format_period_axis_drops_ticks_beyond_a_short_record():
    fig, ax = plt.subplots(1, 1)
    try:
        format_period_axis(ax, max_period_years=3.0)
        fig.canvas.draw()
        assert ax.get_xlim() == (0.5, 3.0)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == ["0.5", "1.0", "2.0"]
    finally:
        plt.close(fig)
