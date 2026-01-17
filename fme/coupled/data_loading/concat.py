from collections.abc import Sequence
from typing import Any

import torch

from fme.core.generics.dataset import GenericDataset
from fme.coupled.data_loading.data_typing import CoupledDataset, CoupledDatasetItem


class ConcatDataset(GenericDataset[CoupledDatasetItem]):
    def __init__(self, datasets: Sequence[CoupledDataset]):
        self._dataset = torch.utils.data.ConcatDataset(datasets)
        self._underlying_datasets = datasets

    def __getitem__(self, idx: int) -> CoupledDatasetItem:
        return self._dataset[idx]

    def __len__(self) -> int:
        return len(self._dataset)

    def set_epoch(self, epoch: int):
        for dataset in self._underlying_datasets:
            dataset.set_epoch(epoch)

    @property
    def first_time(self) -> Any:
        raise NotImplementedError("ConcatDataset does not support inference.")

    @property
    def last_time(self) -> Any:
        raise NotImplementedError("ConcatDataset does not support inference.")
