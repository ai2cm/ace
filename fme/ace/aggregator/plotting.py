from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from fme.core.wandb import Image, WandB


def get_cmap_limits(data: np.ndarray, diverging=False) -> tuple[float, float]:
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    if diverging:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    return vmin, vmax


def plot_imshow(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap | None = None,
    flip_lat: bool = True,
    use_colorbar: bool = True,
) -> Figure:
    """Plot a 2D array using imshow, ensuring figure size is same as array size."""
    min_ = np.nanmin(data) if vmin is None else vmin
    max_ = np.nanmax(data) if vmax is None else vmax
    if len(data.shape) == 3:
        data = fold_healpix_data(data, fill_value=0.5 * (min_ + max_))
    if flip_lat:
        lat_dim = -2
        data = np.flip(data, axis=lat_dim)

    if use_colorbar:
        height, width = data.shape
        colorbar_width = max(1, int(0.025 * width))
        range_ = np.linspace(min_, max_, height)
        range_ = np.repeat(range_[:, np.newaxis], repeats=colorbar_width, axis=1)
        range_ = np.flipud(range_)  # wandb images start from top (and left)
        padding = np.zeros((height, colorbar_width)) + np.nan
        data = np.concatenate((data, padding, range_), axis=1)

    # make figure size (in pixels) be the same as array size
    figsize = np.array(data.T.shape) / plt.rcParams["figure.dpi"]
    fig = Figure(figsize=figsize)  # create directly for cleanup when it leaves scope
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    return fig


def fold_healpix_data(data: np.ndarray, fill_value: float) -> np.ndarray:
    if data.shape[0] != 12:
        raise ValueError(
            "first dimension must be 12 (face) for healpix data, "
            f"got shape {data.shape}"
        )
    # we want to panel the data like this, numbered by first dimension index
    # -----------------
    # |   |   |   |   |
    # |   |   |   |3  |
    # -----------------
    # |   |   |   |   |
    # |   |   |2  |7  |
    # -----------------
    # |   |   |   |   |
    # |   |1  |6  |10 |
    # -----------------
    # |   |   |   |   |
    # |0  |5  |9  |   |
    # -----------------
    # |   |   |   |   |
    # |4  |8  |   |   |
    # -----------------
    # |   |   |   |   |
    # |11 |   |   |   |
    # -----------------
    blank_panel = np.full_like(data[0], fill_value)
    panels = [
        [blank_panel, blank_panel, blank_panel, data[3]],
        [blank_panel, blank_panel, data[2], data[7]],
        [blank_panel, data[1], data[6], data[10]],
        [data[0], data[5], data[9], blank_panel],
        [data[4], data[8], blank_panel, blank_panel],
        [data[11], blank_panel, blank_panel, blank_panel],
    ]
    return np.concatenate([np.concatenate(row, axis=1) for row in panels], axis=0)


def fold_if_healpix_data(data: np.ndarray, fill_value: float) -> np.ndarray:
    if data.shape[0] == 12:
        return fold_healpix_data(data, fill_value)
    return data


def plot_paneled_data(
    data: list[list[np.ndarray]],
    diverging: bool,
    caption: str | None = None,
) -> Image:
    """Plot a list of 2D data arrays in a paneled plot."""
    if diverging:
        cmap = "RdBu_r"
    else:
        cmap = None
    vmin = np.inf
    vmax = -np.inf
    for row in data:
        for arr in row:
            vmin = min(vmin, np.nanmin(arr))
            vmax = max(vmax, np.nanmax(arr))
    if diverging:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    if caption is not None:
        caption += " "
    else:
        caption = ""

    caption += f"vmin={vmin:.4g}, vmax={vmax:.4g}."

    if diverging:
        fill_value = 0.5 * (vmin + vmax)
    else:
        fill_value = vmin
    all_data = _stitch_data_panels(data, fill_value=fill_value)

    fig = plot_imshow(all_data, vmin=vmin, vmax=vmax, cmap=cmap)
    wandb = WandB.get_instance()
    wandb_image = wandb.Image(fig, caption=caption)
    plt.close(fig)

    return wandb_image


def _stitch_data_panels(data: list[list[np.ndarray]], fill_value) -> np.ndarray:
    for row in data:
        if len(row) != len(data[0]):
            raise ValueError("All rows must have the same number of panels.")
    data = [
        [fold_if_healpix_data(arr, fill_value=fill_value) for arr in row]
        for row in data
    ]
    n_rows = len(data)
    n_cols = len(data[0])
    for row in data:
        for arr in row:
            if arr.shape != data[0][0].shape:
                raise ValueError("All panels must have the same shape.")

    stitched_data = np.full(
        (
            n_rows * data[0][0].shape[0] + n_rows - 1,
            n_cols * data[0][0].shape[1] + n_cols - 1,
        ),
        fill_value=fill_value,
    )

    # iterate over rows backwards, as the image starts in the bottom left
    # and moves upwards
    for i, row in enumerate(reversed(data)):
        for j, arr in enumerate(row):
            start_row = i * (arr.shape[0] + 1)
            end_row = start_row + arr.shape[0]
            start_col = j * (arr.shape[1] + 1)
            end_col = start_col + arr.shape[1]
            stitched_data[start_row:end_row, start_col:end_col] = arr

    return stitched_data


def plot_mean_and_samples(
    ax: Any,
    data: xr.DataArray,
    mean_label: str,
    color: str = "cornflowerblue",
    plot_samples: bool = True,
    time_series_dim: str = "year",
    sample_dim: str = "sample",
):
    if time_series_dim in data.dims:
        data.mean(sample_dim).plot(
            ax=ax, x=time_series_dim, label=mean_label, color=color
        )
    if time_series_dim in data.dims and plot_samples:
        for i_sample in range(data.sizes[sample_dim]):
            data.isel(sample=i_sample).plot(
                ax=ax,
                x=time_series_dim,
                color=color,
                alpha=0.4,
                linestyle="-.",
                marker="x",
            )
