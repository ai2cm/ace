import logging

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.xarray import XarrayDataset
from fme.core.typing_ import TensorDict


class XarraySubset(torch.utils.data.Dataset):
    def __init__(self, dataset: XarrayDataset, subset: slice | np.ndarray):
        indices = np.arange(len(dataset))[subset]
        logging.info(f"Subsetting dataset samples according to {subset}.")
        self._dataset = torch.utils.data.Subset(dataset, indices)
        self._sample_start_times = dataset.sample_start_times[indices]
        self._sample_n_times = dataset.sample_n_times
        self.dims = dataset.dims

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[TensorDict, xr.DataArray]:
        return self._dataset[idx]

    @property
    def sample_start_times(self):
        return self._sample_start_times

    @property
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        return self._sample_n_times
