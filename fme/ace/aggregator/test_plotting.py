import numpy as np
import pytest

from .plotting import (
    _stitch_data_panels,
    fold_healpix_data,
    get_cmap_limits,
    plot_imshow,
    plot_paneled_data,
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
