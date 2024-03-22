from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fme.core.wandb import WandB


def get_cmap_limits(data: np.ndarray, diverging=False) -> Tuple[float, float]:
    vmin = data.min()
    vmax = data.max()
    if diverging:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    return vmin, vmax


def plot_imshow(
    data: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    flip_lat: bool = True,
) -> plt.figure:
    """Plot a 2D array using imshow, ensuring figure size is same as array size."""
    if flip_lat:
        lat_dim = -2
        data = np.flip(data, axis=lat_dim)
    # make figure size (in pixels) be the same as array size
    figsize = np.array(data.T.shape) / plt.rcParams["figure.dpi"]
    fig = Figure(figsize=figsize)  # create directly for cleanup when it leaves scope
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    return fig


def plot_paneled_data(
    data: List[List[np.ndarray]],
    diverging: bool,
    caption: Optional[str] = None,
) -> Figure:
    """Plot a list of 2D data arrays in a paneled plot."""
    if diverging:
        cmap = "RdBu_r"
    else:
        cmap = None
    vmin = np.inf
    vmax = -np.inf
    for row in data:
        for arr in row:
            vmin = min(vmin, np.min(arr))
            vmax = max(vmax, np.max(arr))
    if diverging:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    if caption is not None:
        caption += " "
    else:
        caption = ""

    caption += f"vmin={vmin:.4g}, vmax={vmax:.4g}."

    all_data = _stitch_data_panels(data, vmin=vmin)

    fig = plot_imshow(all_data, vmin=vmin, vmax=vmax, cmap=cmap)
    wandb = WandB.get_instance()
    wandb_image = wandb.Image(fig, caption=caption)
    plt.close(fig)
    return wandb_image


def _stitch_data_panels(data: List[List[np.ndarray]], vmin) -> np.ndarray:
    for row in data:
        if len(row) != len(data[0]):
            raise ValueError("All rows must have the same number of panels.")
    n_rows = len(data)
    n_cols = len(data[0])
    for row in data:
        for arr in row:
            if arr.shape != data[0][0].shape:
                raise ValueError("All panels must have the same shape.")

    stitched_data = (
        np.zeros(
            (
                n_rows * data[0][0].shape[0] + n_rows - 1,
                n_cols * data[0][0].shape[1] + n_cols - 1,
            )
        )
        + vmin
    )

    for i, row in enumerate(data):
        for j, arr in enumerate(row):
            start_row = i * (arr.shape[0] + 1)
            end_row = start_row + arr.shape[0]
            start_col = j * (arr.shape[1] + 1)
            end_col = start_col + arr.shape[1]
            stitched_data[start_row:end_row, start_col:end_col] = arr

    return stitched_data
