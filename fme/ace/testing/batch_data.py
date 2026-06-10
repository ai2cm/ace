from collections.abc import Iterable

import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.core.device import get_device


def get_batch_data(
    names: Iterable[str],
    n_samples: int,
    n_time: int,
    img_shape: tuple[int, int] = (5, 5),
    epoch: int | None = None,
    horizontal_dims: list[str] | None = None,
) -> BatchData:
    """
    Create a BatchData of random values with an all-zero time coordinate.
    """
    data = {
        name: torch.rand(n_samples, n_time, *img_shape, device=get_device())
        for name in names
    }
    return BatchData.new_on_device(
        data=data,
        time=xr.DataArray(
            np.zeros((n_samples, n_time)),
            dims=["sample", "time"],
        ),
        labels=None,
        epoch=epoch,
        horizontal_dims=horizontal_dims,
    )
