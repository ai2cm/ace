import dataclasses
from collections.abc import Sequence
from typing import Self

import torch
import xarray as xr

from fme.core.dataset.config import DatasetConfigABC
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.xarray import (
    XarrayDataConfig,
    XarrayDataset,
    XarraySubset,
    get_xarray_datasets,
)
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

    def __getitem__(self, idx: int) -> tuple[TensorDict, xr.DataArray, set[str]]:
        return self._dataset[idx]

    @property
    def sample_start_times(self):
        return self._sample_start_times

    @property
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        return self._sample_n_times


def get_dataset(
    dataset_configs: Sequence[XarrayDataConfig],
    names: Sequence[str],
    n_timesteps: int,
    strict: bool = True,
) -> tuple[XarrayConcat, DatasetProperties]:
    datasets, properties = get_xarray_datasets(
        dataset_configs, names, n_timesteps, strict=strict
    )
    ensemble = XarrayConcat(datasets)
    return ensemble, properties


@dataclasses.dataclass
class ConcatDatasetConfig(DatasetConfigABC):
    """
    Configuration for concatenating multiple datasets across time.

    Parameters:
        concat: List of XarrayDataConfig objects to concatenate.
        strict: Whether to enforce that the datasets to be concatenated
            have the same dimensions and spatial coordinates.
    """

    concat: Sequence[XarrayDataConfig]
    strict: bool = True

    def __post_init__(self):
        self.zarr_engine_used = any(ds.engine == "zarr" for ds in self.concat)

    def build(
        self,
        names: Sequence[str],
        n_timesteps: int,
    ) -> tuple[torch.utils.data.Dataset, DatasetProperties]:
        return get_dataset(
            self.concat,
            names,
            n_timesteps,
            strict=self.strict,
        )
