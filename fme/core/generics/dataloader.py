import logging
from collections.abc import Callable, Iterator
from typing import Generic, Protocol, Self, TypeVar, final, runtime_checkable

import torch

from fme.core.generics.dataset import GenericDataset
from fme.core.rand import alternate_seed

_T = TypeVar("_T")
_BD = TypeVar("_BD", covariant=True)  # collate_fn return type, i.e. "batch data"


@runtime_checkable
class _EpochAwareSampler(Protocol):
    """A sampler whose draw order depends on the training epoch."""

    def set_epoch(self, epoch: int): ...


@runtime_checkable
class _AlternateShuffleSampler(Protocol):
    """A sampler that can produce an independent draw order on demand."""

    def alternate_shuffle(self): ...


class GenericDataLoader(Generic[_BD]):
    """
    Wrapper around torch.utils.data.DataLoader, for type safety.

    Handles initialization of the torch.utils.data.Dataset (via PyTorchDataset)
    and torch.utils.data.Sampler objects needed to initialize the wrapped
    torch.utils.data.DataLoader.
    """

    @final
    def __init__(
        self,
        dataset: GenericDataset[_T],
        collate_fn: Callable[[list[_T]], _BD],
        sampler: torch.utils.data.Sampler | None = None,
        **kwargs,
    ):
        """Initialize the torch.utils.data.DataLoader.

        Args:
            dataset: Generic fme dataset.
            collate_fn: Non-optional batch data collate function.
            sampler: Optional torch.utils.data.Sampler.
            **kwargs: Additional keyword arguments passed to
                torch.utils.data.DataLoader.
        """
        self._dataset = dataset
        self._collate_fn = collate_fn
        self._sampler = sampler
        self._kwargs = kwargs
        if "batch_sampler" in kwargs and kwargs["batch_sampler"] is not None:
            raise ValueError(
                "batch_sampler arg to torch.utils.data.DataLoader is not suppoerted"
            )
        self._torch_loader: torch.utils.data.DataLoader[_BD] = (
            torch.utils.data.DataLoader(
                self._dataset.torch_dataset,
                sampler=self._sampler,
                collate_fn=self._collate_fn,
                **kwargs,
            )
        )

    @final
    def __iter__(self) -> Iterator[_BD]:
        return iter(self._torch_loader)

    @final
    def __len__(self) -> int:
        return len(self._torch_loader)

    @final
    def __next__(self) -> _BD:
        return next(self._torch_loader)

    @property
    @final
    def batch_size(self) -> int:
        return self._torch_loader.batch_size

    @property
    @final
    def n_samples(self) -> int:
        return len(self) * self.batch_size

    @final
    def log_info(self, name: str):
        logging.info(f"{name} data: {self.n_samples} samples, {len(self)} batches")
        logging.info(
            f"{name} data: first sample's initial time: {self._dataset.first_time}"
        )
        logging.info(
            f"{name} data: last sample's initial time: {self._dataset.last_time}"
        )

    @final
    def subset(
        self, start_batch: int | None = None, stop_batch: int | None = None
    ) -> Self:
        if start_batch is None and stop_batch is None:
            return self

        # how many *examples* to drop, not batches
        n_skip = (
            start_batch * self._torch_loader.batch_size
            if start_batch is not None
            else 0
        )
        n_stop = (
            stop_batch * self._torch_loader.batch_size
            if stop_batch is not None
            else None
        )
        # freeze the order produced by the original sampler
        indices = list(self._torch_loader.sampler)[slice(n_skip, n_stop)]

        sub_ds = self._dataset.subset(indices)
        sub_sampler = torch.utils.data.SequentialSampler(sub_ds.torch_dataset)
        kwargs = {**self._kwargs}  # make a shallow copy

        return self.__class__(
            dataset=sub_ds,
            sampler=sub_sampler,
            collate_fn=self._collate_fn,
            **kwargs,
        )

    @final
    def set_epoch(self, epoch: int):
        self._dataset.set_epoch(epoch)
        if isinstance(self._sampler, _EpochAwareSampler):
            self._sampler.set_epoch(epoch)

    @final
    def alternate_shuffle(self):
        if isinstance(self._sampler, _AlternateShuffleSampler):
            self._sampler.alternate_shuffle()
        elif isinstance(self._sampler, torch.utils.data.DistributedSampler):
            self._sampler.set_epoch(alternate_seed(self._sampler.epoch))
