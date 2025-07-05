import logging
import random
from collections.abc import Iterator

import torch

from fme.ace.data_loading.batch_data import BatchData
from fme.core.dataset.xarray import XarrayDataset


class SlidingWindowDataLoader:
    """
    A wrapper for the torch data loader that provides additional properties.
    Create more batches by applying a sliding window to existing batches.
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader[BatchData],
        input_n_timesteps: int,
        output_n_timesteps: int,
        shuffle: bool,
        dataset: XarrayDataset,
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
            dataset: The underlying dataset associated with the loader.
        """
        self._input_loader_len = len(loader)
        self._loader = loader
        assert output_n_timesteps <= input_n_timesteps
        self._input_n_timesteps = input_n_timesteps
        self._output_n_timesteps = output_n_timesteps
        self._n_new_batches = input_n_timesteps - output_n_timesteps + 1
        self._shuffle = shuffle
        self.dataset = dataset

    def __iter__(self) -> Iterator[BatchData]:
        # reset the iterator state
        self._loaditer = iter(self._loader)
        self._current_batch: BatchData | None = None
        return self

    def __len__(self) -> int:
        return self._input_loader_len * self._n_new_batches

    def __next__(self) -> BatchData:
        if self._current_batch is None:
            self._current_batch = next(self._loaditer)
            self._counter = 0
            assert self._current_batch.n_timesteps == self._input_n_timesteps
            self._start_steps = list(range(0, self._n_new_batches))
            if self._shuffle:
                random.shuffle(self._start_steps)
        start = self._start_steps[self._counter]
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
            "the above samples and batches in chunks."
        )
