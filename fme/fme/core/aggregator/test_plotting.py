import numpy as np

from .plotting import _stitch_data_panels, get_cmap_limits, plot_imshow


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


def test_plot_imshow():
    shape = [10, 15]
    data = np.random.randn(*shape)
    fig = plot_imshow(data)
    fig_size_pixels = fig.get_size_inches() * fig.dpi
    # matplotlib figsize is height then width, hence flipping shape order
    assert [int(x) for x in fig_size_pixels] == shape[::-1]


def test_stitch_data_panels():
    data = [
        [np.array([[1, 2]]), np.array([[3, 4]])],
        [np.array([[5, 6]]), np.array([[7, 8]])],
    ]
    stitched = _stitch_data_panels(data, vmin=1)
    expected = np.array(
        [
            [1, 2, 1, 3, 4],
            [1, 1, 1, 1, 1],
            [5, 6, 1, 7, 8],
        ]
    )
    assert np.array_equal(stitched, expected)
