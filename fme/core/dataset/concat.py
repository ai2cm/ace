from collections.abc import Sequence
from typing import Self

import torch
import xarray as xr

from fme.core.dataset.subset import XarraySubset
from fme.core.dataset.xarray import XarrayDataset
from fme.core.typing_ import TensorDict


class XarrayConcat(torch.utils.data.Dataset):
    def __init__(self, datasets: Sequence[XarrayDataset | XarraySubset | Self]):
        self._dataset = torch.utils.data.ConcatDataset(datasets)
        sample_start_times = datasets[0].sample_start_times
        for dataset in datasets[1:]:
            sample_start_times = sample_start_times.append(dataset.sample_start_times)
            assert dataset.sample_n_times == datasets[0].sample_n_times
            if not dataset.sample_n_times == datasets[0].sample_n_times:
                raise ValueError(
                    "All concatenated datasets \
                         must have the same number of steps per sample item."
                )
        self._sample_start_times = sample_start_times
        assert len(self._dataset) == len(sample_start_times)
        self._sample_n_times = datasets[0].sample_n_times

        for dataset in datasets[1:]:
            if dataset.dims != datasets[0].dims:
                raise ValueError(
                    "Datasets being concatenated do not have the same dimensions: "
                    f"{dataset.dims} != {datasets[0].dims}"
                )
        self.dims: list[str] = datasets[0].dims

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
