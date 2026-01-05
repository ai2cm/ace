import abc
from collections.abc import Sequence
from typing import Any, Generic, TypeVar, final

import torch

_T_co = TypeVar("_T_co", covariant=True)


class GenericDataset(Generic[_T_co], abc.ABC):
    """Type-safe generic dataset API."""

    @abc.abstractmethod
    def __getitem__(self, index) -> _T_co:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @final
    def torch_dataset(self) -> torch.utils.data.Dataset[_T_co]:
        return _PyTorchDataset(self)

    @final
    def subset(self, indices: Sequence[int]) -> "_Subset[_T_co]":
        return _Subset(self, indices)

    @abc.abstractmethod
    def set_epoch(self, epoch: int):
        pass

    @property
    @abc.abstractmethod
    def first_time(self) -> Any:
        """
        Start time of the first sample in the dataset.
        """
        pass

    @property
    @abc.abstractmethod
    def last_time(self) -> Any:
        """
        *Start* time of the last sample in the dataset.
        """
        pass


class _Subset(GenericDataset[_T_co]):
    def __init__(self, dataset: GenericDataset[_T_co], indices: Sequence[int]):
        self._dataset = dataset
        self._indices = indices

    def __getitem__(self, index) -> _T_co:
        """Reimplements torch.utils.data.Subset.__getitem__."""
        if isinstance(index, list):
            return self._dataset[[self._indices[i] for i in index]]
        return self._dataset[self._indices[index]]

    def __len__(self) -> int:
        return len(self._indices)

    def set_epoch(self, epoch: int):
        self._dataset.set_epoch(epoch)

    @property
    def first_time(self) -> Any:
        return self._dataset.first_time

    @property
    def last_time(self) -> Any:
        return self._dataset.last_time


class _PyTorchDataset(Generic[_T_co], torch.utils.data.Dataset[_T_co]):
    """Wraps GenericDataset with torch.utils.data.Dataset, generically.

    NOTE: This class is not type-safe, nor is it intended to be. It's usefulness
    is in deferring inheritance of torch.utils.data.Dataset as long as possible
    so that we can maintain interal type safety with GenericDataset.
    """

    def __init__(self, dataset: GenericDataset[_T_co]):
        self._dataset = dataset

    def __getitem__(self, index) -> _T_co:
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)
