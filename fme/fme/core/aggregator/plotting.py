from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    return fig
