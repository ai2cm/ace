import abc
import logging
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar, final

import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict


def get_data_loader(
    dataset: torch.utils.data.Dataset[tuple[TensorDict, xr.DataArray]],
    batch_size: int,
    n_window_timesteps: int,
    time_buffer: int,
    num_workers: int,
    sampler: torch.utils.data.Sampler,
    shuffled: bool,
    drop_last: bool,
    pin_memory: bool,
    collate_fn: Callable,
    multiprocessing_context: str | None,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> "DataLoaderABC":
    if prefetch_factor is None:
        # DataLoader default is not None so we must leave it unset
        kwargs = {}
    else:
        kwargs = {"prefetch_factor": prefetch_factor}
    torch_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=persistent_workers,
        **kwargs,
    )
    dataloader: DataLoaderABC = TorchDataLoader(torch_dataloader, sampler, dataset)
    if time_buffer > 0:
        n_timesteps_preloaded = time_buffer + n_window_timesteps
        dataloader = SlidingWindowDataLoader(
            loader=dataloader,
            input_n_timesteps=n_timesteps_preloaded,
            output_n_timesteps=n_window_timesteps,
            shuffle=shuffled,
        )

    if len(dataloader) == 0:
        msg = (
            "No batches in dataloader: "
            f"{dataloader.n_samples} samples, "
            f"batch size is {dataloader.batch_size}"
        )
        if time_buffer > 0:
            msg += f", and an outer sample length is {n_timesteps_preloaded}"
        raise ValueError(msg)

    return dataloader


class DataLoaderABC(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[BatchData]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __next__(self) -> BatchData:
        pass

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        """
        Number of samples in a single batch.
        """
        pass

    @property
    @abc.abstractmethod
    def n_samples(self) -> int:
        pass

    @property
    @final
    def n_batches(self) -> int:
        # Note this method is not (at time of writing) used and has the same meaning
        # as __len__, it exists only so that the meaning of the n_samples property
        # is unambiguous. Previously there have been bugs caused by confusion between
        # the two.
        return len(self)

    @abc.abstractmethod
    def log_info(self, name: str):
        pass

    @abc.abstractmethod
    def subset(self, start: int) -> "DataLoaderABC":
        """
        Return a subset of the data loader starting at the given index.
        """
        pass

    @abc.abstractmethod
    def set_epoch(self, epoch: int):
        pass


T = TypeVar("T")


class GenericTorchDataLoader(Generic[T]):
    SelfType = TypeVar("SelfType", bound="GenericTorchDataLoader")

    @final
    def __init__(
        self,
        loader: torch.utils.data.DataLoader[T],
        sampler: torch.utils.data.Sampler,
        dataset: torch.utils.data.Dataset[tuple[TensorDict, xr.DataArray]],
    ):
        """
        Args:
            loader: The underlying torch data loader.
            sampler: The sampler associated with the loader.
            dataset: The underlying dataset associated with the loader. This is
                provided to allow direct access to the dataset.
        """
        self._loader = loader
        self._sampler = sampler
        self._dataset = dataset

    @final
    def __iter__(self) -> Iterator[T]:
        return iter(self._loader)

    @final
    def __len__(self) -> int:
        return len(self._loader)

    @final
    def __next__(self) -> T:
        return next(self._loader)

    @property
    @final
    def batch_size(self) -> int:
        return self._loader.batch_size

    @property
    @final
    def n_samples(self) -> int:
        return len(self) * self.batch_size

    @final
    def log_info(self, name: str):
        logging.info(f"{name} data: {self.n_samples} samples, {len(self)} batches")
        logging.info(f"{name} data: first sample's initial time: {self._first_time}")
        logging.info(f"{name} data: last sample's initial time: {self._last_time}")

    @property
    @final
    def _first_time(self) -> Any:
        return self._dataset[0][1].values[0]

    @property
    @final
    def _last_time(self) -> Any:
        return self._dataset[-1][1].values[0]

    @final
    def subset(self: SelfType, start: int) -> SelfType:
        if start == 0:
            return self
        # how many *examples* to drop, not batches
        n_skip = start * self._loader.batch_size

        # freeze the order produced by the original sampler
        indices = list(self._loader.sampler)[n_skip:]

        # dataset view that exposes only the remaining indices
        sub_ds = torch.utils.data.Subset(self._loader.dataset, indices)

        # iterate through the subset exactly as-is (no extra shuffling)
        sub_sampler = torch.utils.data.SequentialSampler(sub_ds)
        sub_loader = torch.utils.data.DataLoader(
            sub_ds,
            batch_size=self._loader.batch_size,
            num_workers=self._loader.num_workers,
            sampler=sub_sampler,
            drop_last=self._loader.drop_last,
            pin_memory=self._loader.pin_memory,
            collate_fn=self._loader.collate_fn,
            prefetch_factor=self._loader.prefetch_factor,
            multiprocessing_context=self._loader.multiprocessing_context,
            persistent_workers=self._loader.persistent_workers,
        )
        return self.__class__(
            sub_loader,
            sampler=sub_sampler,
            dataset=sub_ds,
        )

    @final
    def set_epoch(self, epoch: int):
        if isinstance(self._sampler, torch.utils.data.DistributedSampler):
            self._sampler.set_epoch(epoch)


class TorchDataLoader(GenericTorchDataLoader[BatchData], DataLoaderABC):
    pass


def random_columns_no_replacement(N, M, seed=None):
    """
    Return an (N, M) int array.
    Each column is an independent random permutation of range(N).
    """
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((N, M), dtype=np.float32), axis=1)


class SlidingWindowDataLoader(DataLoaderABC):
    """
    A wrapper for the torch data loader that provides additional properties.
    Create more batches by applying a sliding window to existing batches.
    """

    def __init__(
        self,
        loader: DataLoaderABC,
        input_n_timesteps: int,
        output_n_timesteps: int,
        shuffle: bool,
        n_skipped_input_batches: int = 0,
        skip_first_n_output_batches: int = 0,
    ):
        """
        Create a sliding window over the input data loader to generate new batches.
            We assume that the loader returns batches with the same number of timesteps
            equal to "input_n_timesteps". This is asserted for every batch in __next__.

        Args:
            loader: Pre-batched input data loader that returns batches of
                `input_n_timesteps` timesteps.
            input_n_timesteps: Number of timesteps in the input batch, i.e., the size of
                the sliding window.
            output_n_timesteps: Number of timesteps in the output batch, must be smaller
                than the number of input timesteps.
            shuffle: Whether to shuffle the start indices for the sliding window.
            n_skipped_input_batches: Number of input batches that have been skipped in
                the base loader. Helpful to ensure that the random order of samples
                is preserved.
            skip_first_n_output_batches: Number of output batches to skip at the start
                of the loader.
                Helpful for skipping a number of outputs smaller than the size of the
                sliding window.
        """
        self._input_loader_len = len(loader)
        self._loader = loader
        assert output_n_timesteps <= input_n_timesteps
        self._input_n_timesteps = input_n_timesteps
        self._output_n_timesteps = output_n_timesteps
        self._n_new_batches = input_n_timesteps - output_n_timesteps + 1
        self._shuffle = shuffle
        self._epoch = 0
        self._n_skipped_input_batches = n_skipped_input_batches
        self._i_batch = self._n_skipped_input_batches
        self._skip_first_n_output_batches = skip_first_n_output_batches
        self._current_batch: BatchData | None = None
        self._init_shuffle_indices()

    def __iter__(self) -> Iterator[BatchData]:
        # reset the iterator state
        self._loaditer = iter(self._loader)
        self._i_batch = self._n_skipped_input_batches
        try:
            self._current_batch = next(self._loaditer)
        except StopIteration:
            return iter([])
        self._counter = self._skip_first_n_output_batches
        self._init_shuffle_indices()
        return self

    def _init_shuffle_indices(self):
        dist = Distributed.get_instance()
        if self._shuffle:
            self._shuffle_indices = torch.from_numpy(
                random_columns_no_replacement(
                    self._n_skipped_input_batches + self._input_loader_len,
                    self._n_new_batches,
                    seed=self._epoch + dist.get_seed(),
                )
            )
        else:
            self._shuffle_indices = torch.arange(self._n_new_batches)[
                None, :
            ].broadcast_to(
                (
                    self._n_skipped_input_batches + self._input_loader_len,
                    self._n_new_batches,
                )
            )

    def __len__(self) -> int:
        return self._input_loader_len * self._n_new_batches

    def __next__(self) -> BatchData:
        if self._current_batch is None:
            self._current_batch = next(self._loaditer)
            self._i_batch += 1
            self._counter = 0
            assert self._current_batch.n_timesteps == self._input_n_timesteps
        start = self._shuffle_indices[self._i_batch, self._counter]
        slice_ = slice(start, start + self._output_n_timesteps)
        small_batch = self._current_batch.select_time_slice(slice_)
        self._counter += 1
        if self._counter == self._n_new_batches:
            self._current_batch = None
        return small_batch

    @property
    def batch_size(self) -> int:
        """
        Number of samples in a single batch.
        """
        return self._loader.batch_size

    @property
    def n_samples(self) -> int:
        return self._loader.n_samples

    def _n_samples_per_dataset_item(self) -> int:
        """
        Number of samples per dataset outer item, i.e., in a window.
        """
        return self._input_n_timesteps - self._output_n_timesteps + 1

    def log_info(self, name: str):
        logging.info(
            f"{name} data: {len(self) * self.batch_size} samples, {len(self)} batches"
        )
        logging.info(
            f"{name} data: An outer window size of "
            f"{self._n_samples_per_dataset_item()} samples was used to load "
            "the above samples and batches in chunks, coming from the following "
            "base loader:"
        )
        self._loader.log_info(name + " (base loader)")

    def subset(self, start_batch: int) -> "SlidingWindowDataLoader":
        sub_batches_per_contained_batch = (
            1 + self._input_n_timesteps - self._output_n_timesteps
        )
        n_batches_to_skip = (
            start_batch * self._loader.batch_size // sub_batches_per_contained_batch
        )
        subsetted_loader = self._loader.subset(n_batches_to_skip)
        self._loaditer = iter(subsetted_loader)
        n_sub_batches_to_skip = (
            start_batch * self._loader.batch_size % sub_batches_per_contained_batch
        )
        return SlidingWindowDataLoader(
            subsetted_loader,
            input_n_timesteps=self._input_n_timesteps,
            output_n_timesteps=self._output_n_timesteps,
            shuffle=self._shuffle,
            n_skipped_input_batches=n_batches_to_skip,
            skip_first_n_output_batches=n_sub_batches_to_skip,
        )

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._loader.set_epoch(epoch)
