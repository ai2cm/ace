import abc
import logging
from collections.abc import Callable, Iterator
from typing import cast, final

import numpy as np
import torch

from fme.ace.data_loading.batch_data import BatchData
from fme.core.dataset.dataset import DatasetItem
from fme.core.dataset.schedule import IntSchedule
from fme.core.distributed import Distributed
from fme.core.generics.dataloader import GenericDataLoader
from fme.core.generics.dataset import GenericDataset
from fme.core.rand import alternate_seed


def get_data_loader(
    dataset: GenericDataset[DatasetItem],
    batch_size: int,
    n_window_timesteps: IntSchedule,
    time_buffer: int,
    num_workers: int,
    sampler: torch.utils.data.Sampler,
    shuffled: bool,
    drop_last: bool,
    pin_memory: bool,
    collate_fn: Callable[[list[DatasetItem]], BatchData],
    multiprocessing_context: str | None,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> "DataLoaderABC":
    if prefetch_factor is None:
        # DataLoader default is not None so we must leave it unset
        kwargs = {}
    else:
        kwargs = {"prefetch_factor": prefetch_factor}
    torch_dataloader = TorchDataLoader(
        dataset,
        collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=pin_memory,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=persistent_workers,
        **kwargs,
    )
    dataloader: DataLoaderABC
    if time_buffer > 0:
        dataloader = SlidingWindowDataLoader(
            loader=torch_dataloader,
            output_n_timesteps=n_window_timesteps,
            time_buffer=time_buffer,
            shuffle=shuffled,
        )
    else:
        dataloader = torch_dataloader

    if len(dataloader) == 0:
        msg = (
            "No batches in dataloader: "
            f"{dataloader.n_samples} samples, "
            f"batch size is {dataloader.batch_size}"
        )
        if time_buffer > 0:
            msg += f", and time_buffer is {time_buffer}"
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
    def subset(
        self, start_batch: int | None = None, stop_batch: int | None = None
    ) -> "DataLoaderABC":
        """
        Return a subset of the data loader starting at the given batch index
        and stopping at the given batch index (exclusive).

        Args:
            start_batch: Index of the first batch to include.
            stop_batch: Index of the last batch to include (exclusive).

        Returns:
            A subset of the data loader.
        """
        pass

    @abc.abstractmethod
    def set_epoch(self, epoch: int):
        pass

    @abc.abstractmethod
    def alternate_shuffle(self):
        """
        Change the random shuffle of the data loader for the current epoch.
        """
        pass


class TorchDataLoader(GenericDataLoader[BatchData], DataLoaderABC):
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
        loader: TorchDataLoader,
        output_n_timesteps: IntSchedule,
        time_buffer: int,
        shuffle: bool,
        n_skipped_input_batches: int = 0,
        skip_first_n_output_batches: int = 0,
        skip_last_n_output_batches: int = 0,
    ):
        """
        Create a sliding window over the input data loader to generate new batches.
            We assume that the loader returns batches with the same number of timesteps
            equal to "input_n_timesteps". This is asserted for every batch in __next__.

        Args:
            loader: Pre-batched input data loader that returns batches of
                `output_n_timesteps + time_buffer` timesteps.
            output_n_timesteps: Number of timesteps in the output batch.
            time_buffer: Number of extra timesteps in the input batches to allow
                creating multiple output batches from a single input batch.
            shuffle: Whether to shuffle the start indices for the sliding window.
            n_skipped_input_batches: Number of input batches that have been skipped in
                the base loader. Helpful to ensure that the random order of samples
                is preserved.
            skip_first_n_output_batches: Number of output batches to skip at the start
                of the loader.
                Helpful for skipping a number of outputs smaller than the size of the
                sliding window.
            skip_last_n_output_batches: Number of output batches to skip at the end
                of the loader.
                Helpful for skipping a number of outputs smaller than the size of the
                sliding window.
        """
        self._input_loader_len = len(loader)
        self._loader = loader
        self._input_n_timesteps_schedule = output_n_timesteps.add(time_buffer)
        self._n_new_batches = time_buffer + 1
        self._time_buffer = time_buffer
        self._output_n_timesteps_schedule = output_n_timesteps
        self._update_n_timesteps_for_epoch(0)
        self._shuffle = shuffle
        self._n_skipped_input_batches = n_skipped_input_batches
        self._i_batch = self._n_skipped_input_batches
        self._skip_first_n_output_batches = skip_first_n_output_batches
        self._skip_last_n_output_batches = skip_last_n_output_batches
        self._current_batch: BatchData | None = None
        self.set_epoch(0)

    def _update_n_timesteps_for_epoch(self, epoch: int):
        self._input_n_timesteps = self._input_n_timesteps_schedule.get_value(epoch)
        self._output_n_timesteps = self._output_n_timesteps_schedule.get_value(epoch)
        assert self._output_n_timesteps <= self._input_n_timesteps
        self._stored_shuffle_indices = None  # needs to be re-generated

    def __iter__(self) -> Iterator[BatchData]:
        # reset the iterator state
        self._loaditer = iter(self._loader)
        self._i_batch = self._n_skipped_input_batches
        try:
            self._current_batch = next(self._loaditer).to_device()
        except StopIteration:
            return iter([])
        self._counter = self._skip_first_n_output_batches
        return self

    @property
    def _shuffle_indices(self):
        if self._stored_shuffle_indices is None:
            if self._shuffle:
                self._stored_shuffle_indices = torch.from_numpy(
                    random_columns_no_replacement(
                        self._n_skipped_input_batches + self._input_loader_len,
                        self._n_new_batches,
                        seed=self._seed,
                    )
                )
            else:
                self._stored_shuffle_indices = torch.arange(self._n_new_batches)[
                    None, :
                ].broadcast_to(
                    (
                        self._n_skipped_input_batches + self._input_loader_len,
                        self._n_new_batches,
                    )
                )
        return self._stored_shuffle_indices

    def __len__(self) -> int:
        return self._input_loader_len * self._n_new_batches

    def __next__(self) -> BatchData:
        if self._current_batch is None:
            self._current_batch = next(self._loaditer).to_device()
            self._i_batch += 1
            self._counter = 0
            assert self._current_batch.n_timesteps == self._input_n_timesteps
        start = self._shuffle_indices[self._i_batch, self._counter]
        slice_ = slice(start, start + self._output_n_timesteps)
        small_batch = self._current_batch.select_time_slice(slice_)
        self._counter += 1
        if self._counter == self._n_new_batches:
            self._current_batch = None
        # on the last batch, we need to skip the last n output batches
        if self._i_batch == self._input_loader_len - 1:
            if self._counter == self._n_new_batches - self._skip_last_n_output_batches:
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
        return self._n_new_batches

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

    def subset(
        self, start_batch: int | None = None, stop_batch: int | None = None
    ) -> "SlidingWindowDataLoader":
        if start_batch is None and stop_batch is None:
            return self
        sub_batches_per_contained_batch = (
            1 + self._input_n_timesteps - self._output_n_timesteps
        )
        n_batches_to_skip, n_sub_batches_to_skip = get_skip_batches(
            sub_batches_per_contained_batch, start_batch
        )
        n_batches_to_stop, n_sub_batches_to_skip_last = get_stop_batches(
            sub_batches_per_contained_batch, stop_batch
        )
        subsetted_loader = self._loader.subset(
            start_batch=n_batches_to_skip, stop_batch=n_batches_to_stop
        )
        self._loaditer = iter(subsetted_loader)
        loader = SlidingWindowDataLoader(
            cast(TorchDataLoader, subsetted_loader),
            output_n_timesteps=self._output_n_timesteps_schedule,
            time_buffer=self._time_buffer,
            shuffle=self._shuffle,
            n_skipped_input_batches=n_batches_to_skip,
            skip_first_n_output_batches=n_sub_batches_to_skip,
            skip_last_n_output_batches=n_sub_batches_to_skip_last,
        )
        loader.set_epoch(self._epoch)
        return loader

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        dist = Distributed.get_instance()
        self._seed = self._epoch + dist.get_seed()
        self._loader.set_epoch(epoch)
        self._update_n_timesteps_for_epoch(epoch)

    def alternate_shuffle(self):
        self._loader.alternate_shuffle()
        self._seed = alternate_seed(self._seed)
        self._stored_shuffle_indices = None  # needs to be re-generated


def get_skip_batches(
    sub_batches_per_contained_batch: int, start: int | None = None
) -> tuple[int, int]:
    """
    Get the number of batches and sub-batches to skip.

    Args:
        sub_batches_per_contained_batch: Number of sub-batches per contained batch.
        batch_size: Size of the batch.
        start: Number of samples (not batches) to skip.

    Returns:
        Number of batches to skip and number of sub-batches to skip.
    """
    if start is None:
        n_batches_to_skip = 0
        n_sub_batches_to_skip = 0
    else:
        n_batches_to_skip = start // sub_batches_per_contained_batch
        n_sub_batches_to_skip = start % sub_batches_per_contained_batch
    return n_batches_to_skip, n_sub_batches_to_skip


def get_stop_batches(
    sub_batches_per_contained_batch: int, stop: int | None = None
) -> tuple[int | None, int]:
    """
    Get the number of batches and sub-batches to stop.

    Args:
        sub_batches_per_contained_batch: Number of sub-batches per contained batch.
        batch_size: Size of the batch.
        stop: Number of batches to stop at (exclusive).

    Returns:
        Number of batches to stop at (exclusive)
        and which sub-batch to stop at (exclusive).
    """
    if stop is None:
        n_batches_to_stop: int | None = None
        n_sub_batches_to_skip_last = 0
    else:
        n_batches_to_stop = (stop - 1) // sub_batches_per_contained_batch + 1
        n_sub_batches_to_skip_last = (
            sub_batches_per_contained_batch - stop % sub_batches_per_contained_batch
        ) % sub_batches_per_contained_batch
    return n_batches_to_stop, n_sub_batches_to_skip_last
