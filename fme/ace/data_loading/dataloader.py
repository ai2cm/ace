import abc
import logging
from collections.abc import Callable, Iterator
from typing import final

import torch

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.pool import PooledSequence, build_pool_schedule
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
    time_buffer_pool_size: int,
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
            pool_size=time_buffer_pool_size,
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


class SlidingWindowDataLoader(DataLoaderABC):
    """
    A wrapper for the torch data loader that provides additional properties.
    Create more batches by applying a sliding window to existing batches.

    When ``pool_size`` > 1, multiple input batches are held in memory
    simultaneously and output batches are interleaved across them,
    reducing the temporal correlation between consecutive outputs.
    """

    def __init__(
        self,
        loader: TorchDataLoader,
        output_n_timesteps: IntSchedule,
        time_buffer: int,
        shuffle: bool,
        pool_size: int = 1,
        _subset_start: int = 0,
        _subset_stop: int | None = None,
    ):
        """
        Args:
            loader: Pre-batched input data loader that returns batches of
                ``output_n_timesteps + time_buffer`` timesteps.
            output_n_timesteps: Number of timesteps in each output batch.
            time_buffer: Number of extra timesteps in the input batches,
                allowing ``time_buffer + 1`` output batches per input batch.
            shuffle: Whether to randomize slot selection and sub-window order.
            pool_size: Number of input batches to hold in memory at once.
                Output batches are drawn by selecting a pool slot and then a
                sub-window within that slot; exhausted slots are replaced by
                the next input batch.
        """
        self._loader = loader
        self._input_n_timesteps_schedule = output_n_timesteps.add(time_buffer)
        self._n_new_batches = time_buffer + 1
        self._time_buffer = time_buffer
        self._output_n_timesteps_schedule = output_n_timesteps
        self._pool_size = pool_size
        self._shuffle = shuffle
        self._total_input_batches = len(loader)
        self._subset_start = _subset_start
        self._subset_stop = _subset_stop

        self._update_n_timesteps_for_epoch(0)
        self.set_epoch(0)

    def _update_n_timesteps_for_epoch(self, epoch: int):
        self._input_n_timesteps = self._input_n_timesteps_schedule.get_value(epoch)
        self._output_n_timesteps = self._output_n_timesteps_schedule.get_value(epoch)
        assert self._output_n_timesteps <= self._input_n_timesteps
        self._pooled: PooledSequence[BatchData, BatchData] | None = None

    def _prepare(self, batch: BatchData) -> BatchData:
        assert batch.n_timesteps == self._input_n_timesteps
        return batch.to_device()

    def _extract(self, batch: BatchData, offset: int) -> BatchData:
        return batch.select_time_slice(slice(offset, offset + self._output_n_timesteps))

    @property
    def _pooled_sequence(self) -> PooledSequence[BatchData, BatchData]:
        if self._pooled is None:
            schedule = build_pool_schedule(
                n_input_batches=self._total_input_batches,
                n_sub_batches=self._n_new_batches,
                pool_size=self._pool_size,
                shuffle=self._shuffle,
                seed=self._seed,
            )
            pooled: PooledSequence[BatchData, BatchData] = PooledSequence(
                source=self._loader,
                schedule=schedule,
                pool_size=self._pool_size,
                extract=self._extract,
                prepare=self._prepare,
            )
            if self._subset_start > 0 or self._subset_stop is not None:
                pooled = pooled.subset(start=self._subset_start, stop=self._subset_stop)
            self._pooled = pooled
        return self._pooled

    def __iter__(self) -> Iterator[BatchData]:
        self._iter = iter(self._pooled_sequence)
        return self

    def __len__(self) -> int:
        return len(self._pooled_sequence)

    def __next__(self) -> BatchData:
        return next(self._iter)

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
            f"{name} data: pool_size={self._pool_size}, "
            f"{self._n_new_batches} sub-batches per window, "
            "coming from the following base loader:"
        )
        self._loader.log_info(name + " (base loader)")

    def subset(
        self, start_batch: int | None = None, stop_batch: int | None = None
    ) -> "SlidingWindowDataLoader":
        if start_batch is None and stop_batch is None:
            return self
        new_start = self._subset_start + (start_batch or 0)
        if stop_batch is not None:
            effective_stop = (
                self._subset_stop
                if self._subset_stop is not None
                else self._total_input_batches * self._n_new_batches
            )
            new_stop: int | None = min(self._subset_start + stop_batch, effective_stop)
        else:
            new_stop = self._subset_stop
        loader = SlidingWindowDataLoader(
            self._loader,
            output_n_timesteps=self._output_n_timesteps_schedule,
            time_buffer=self._time_buffer,
            shuffle=self._shuffle,
            pool_size=self._pool_size,
            _subset_start=new_start,
            _subset_stop=new_stop,
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
        self._pooled = None
