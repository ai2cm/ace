from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product

import torch
import xarray as xr
from torch.utils.data import DataLoader

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.generics.data import SizedMap

from ..data import BatchData, Topography
from ..data.config import BatchItemDatasetAdapter
from .constants import ENSEMBLE_NAME, TIME_NAME


@dataclass
class SliceWorkItem:
    """
    Work item specification for generation: which time and ensemble slices to process.

    This is an immutable specification of work to be done. To attach batch data,
    use the `with_batch()` class method to create a `LoadedWorkItem`.
    """

    time_slice: slice  # times to grab from the dataset
    ens_slice: slice  # ensemble members to generate
    is_padding: bool = False  # Used for even GPU work distribution

    @staticmethod
    def _check_slice_no_none(slice_: slice) -> None:
        # check for valid work slices, not None, positive, and monotonic
        if slice_.start is None or slice_.stop is None:
            raise ValueError("Slice start and stop must be defined (not None)")
        if slice_.start < 0 or slice_.stop < 0:
            raise ValueError("Slice start and stop must be positive")
        if slice_.start >= slice_.stop:
            raise ValueError("Slice start must be less than stop")

    def __post_init__(self):
        self._check_slice_no_none(self.time_slice)
        self._check_slice_no_none(self.ens_slice)
        self.n_ens = self.ens_slice.stop - self.ens_slice.start

    @property
    def time_indices(self) -> list[int]:
        """Get list of time indices to load from the dataset."""
        sl_ = self.time_slice
        return list(range(sl_.start, sl_.stop))

    @property
    def dim_insert_slices(self) -> dict[str, slice]:
        """Get zarr position slices for writing output."""
        return {
            TIME_NAME: self.time_slice,
            ENSEMBLE_NAME: self.ens_slice,
        }

    @classmethod
    def with_batch(
        cls, work_item: "SliceWorkItem", batch: BatchData
    ) -> "LoadedSliceWorkItem":
        """
        Create a LoadedWorkItem with batch data attached.
        """
        return LoadedSliceWorkItem(
            time_slice=work_item.time_slice,
            ens_slice=work_item.ens_slice,
            is_padding=work_item.is_padding,
            batch=batch,
        )


@dataclass
class LoadedSliceWorkItem(SliceWorkItem):
    """
    Work item with batch data attached, ready for generation.

    Created via _SliceWorkItem.with_batch() after loading data from the dataset.
    """

    def __init__(self, *args, batch: BatchData, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = batch

    def __post_init__(self):
        super().__post_init__()

    def to_device(self) -> "LoadedSliceWorkItem":
        return LoadedSliceWorkItem(
            time_slice=self.time_slice,
            ens_slice=self.ens_slice,
            is_padding=self.is_padding,
            batch=self.batch.to_device(),
        )


class SliceItemDataset:
    """
    Dataset that loads a batch of data with metadata about time
    and ensemble slices for generation and saving out to Zarr.

    Intended to be used with batch_size=1 because each individual
    item loads the specified batch for a specified time and
    ensemble slice.
    """

    def __init__(
        self,
        slice_items: list[SliceWorkItem],
        dataset: BatchItemDatasetAdapter,
        spatial_shape: tuple[int, int] | None = None,
    ) -> None:
        self.slice_items = slice_items
        self.dataset = dataset
        self._dtype = None

        if spatial_shape is None:
            sample_batch_item = self.dataset[0]
            self.spatial_shape = sample_batch_item.horizontal_shape
        else:
            self.spatial_shape = spatial_shape

    def __len__(self) -> int:
        return len(self.slice_items)

    def __getitem__(self, idx: int) -> LoadedSliceWorkItem:
        work_spec = self.slice_items[idx]
        data_items = [self.dataset[i] for i in work_spec.time_indices]
        batch = BatchData.from_sequence(data_items)
        loaded_item = SliceWorkItem.with_batch(work_spec, batch)
        return loaded_item

    @property
    def max_output_shape(self):
        first_item = self.slice_items[0]
        n_times = first_item.time_slice.stop - first_item.time_slice.start
        n_ensembles = first_item.ens_slice.stop - first_item.ens_slice.start
        return (n_times, n_ensembles, *self.spatial_shape)

    @property
    def dtype(self) -> torch.dtype:
        if self._dtype is not None:
            return self._dtype

        sample_item = self.dataset[0]
        sample_tensor = next(iter(sample_item.data.values()))
        self._dtype = sample_tensor.dtype
        return self._dtype


def _generate_slices(total: int, step: int) -> list[slice]:
    """Generate list of slices to cover a total range using a fixed step size."""
    slices = []
    start = 0
    while start < total:
        end = min(start + step, total)
        slices.append(slice(start, end))
        start = end
    return slices


def _get_time_ens_slices_step_single_gpu(
    n_ens: int, max_samples_per_gpu: int
) -> tuple[int, int]:
    n_ens_per_slice = min(n_ens, max_samples_per_gpu)
    n_time_per_slice = max(1, max_samples_per_gpu % n_ens_per_slice)

    return n_time_per_slice, n_ens_per_slice


def _get_slice_step_size_for_distributed(total_size: int, requested_step: int) -> int:
    # If not evenly divisible, search for a step size that would only
    # produce a difference of 1 between the regular slice step size
    # and the remainder slice size.

    new_step = max(1, min(total_size, requested_step))
    remainder_work = total_size % new_step

    if remainder_work == 0 or new_step - remainder_work == 1:
        # evenly divides total size or the remainder (size of work item)
        # is only 1 less than the requested step size meaning trailing
        # slice portions are nearly balanced
        return new_step
    elif new_step - remainder_work == 1:
        return new_step
    else:
        # keep searching by incrementing down
        return _get_slice_step_size_for_distributed(total_size, new_step - 1)


def _get_time_ens_slice_step_multi_gpu(
    n_times: int,
    n_ens: int,
    max_samples_per_gpu: int,
) -> tuple[int, int]:
    """
    Get the time and ensemble slice step sizes for multi-GPU distributed generation.

    We try to balance the work item sizes across GPUs by reducing the slice step
    sizes to minimize the size difference of any remainder work compared to the
    size of a regular work item.  This is to avoid having very small work items
    that may appear as "hanging" and cause NCCL timeouts during generation.

    The general strategy is as follows:
    1. Determine ensemble slice step size first, since we prioritize ensemble
       dimension splits.
    2. Given the ensemble slice step size, determine the time slice step size
       if there is remaining capacity for multiple samples in the time dimension
       based on the max samples per GPU.

    I think the worst case relative difference would be having a remaining work item
    of size 1 (1 time, 1 ens) compared to a regular work item with size 4 (2 time,
    2 ens) for a relative difference of 75%.  Hopefully, small work items would
    compute quickly enough to not cause timeouts.

    Unfortunately, this will not help if the compute time difference for generating
    a work item is very large (perhaps generating global tiles?).  In that case,
    the only answer if the difference causes NCCL timeouts would be to manually select
    the time / ensemble sizes of the dataset to perfectly align with max samples,
    increase the NCCL timeout, or just increase the max samples per GPU to try and
    reduce the compute difference (if it finds a sensible slice step size).
    We will see if this is actually an issue in practice.
    """
    n_ens_per_slice = _get_slice_step_size_for_distributed(n_ens, max_samples_per_gpu)
    n_time_per_slice = _get_slice_step_size_for_distributed(
        n_times, max_samples_per_gpu // n_ens_per_slice
    )
    return n_time_per_slice, n_ens_per_slice


def _get_work_item_padding(
    work_items: list[SliceWorkItem], dist: Distributed
) -> list[SliceWorkItem]:
    remainder = len(work_items) % dist.world_size
    if remainder == 0:
        return []

    n_padding = dist.world_size - remainder
    # repeat last item as padding
    padding_item = SliceWorkItem(
        time_slice=work_items[-1].time_slice,
        ens_slice=work_items[-1].ens_slice,
        is_padding=True,
    )
    return [padding_item] * n_padding


def get_work_items(
    n_times: int, n_ens: int, max_samples_per_gpu: int, dist: Distributed | None = None
) -> list[SliceWorkItem]:
    """
    Create work items for generation based on time and ensemble slices.

    Args:
        n_times: Number of time steps in the data
        n_ens: Total number of ensemble members to generate
        max_samples_per_gpu: Max number of time and ensemble samples per GPU batch.
            Splits are done prioritizing the ensemble dimension first.  If distributed,
            the splits are adjusted to try and make sure the last slice in
            time and ensemble dimensions are not too small compared to the others.
        dist: Distributed instance for inferring padding work items (optional)
    """
    work_items: list[SliceWorkItem] = []

    dist = dist or Distributed.get_instance()
    if dist.is_distributed():
        n_time_per_slice, n_ens_per_slice = _get_time_ens_slice_step_multi_gpu(
            n_times, n_ens, max_samples_per_gpu
        )
    else:
        n_time_per_slice, n_ens_per_slice = _get_time_ens_slices_step_single_gpu(
            n_ens, max_samples_per_gpu
        )

    ens_slices = _generate_slices(n_ens, n_ens_per_slice)
    time_slices = _generate_slices(n_times, n_time_per_slice)

    work_items = [
        SliceWorkItem(time_sl, ens_sl)
        for (time_sl, ens_sl) in product(time_slices, ens_slices)
    ]

    # Pad work items to evenly distribute across GPUs
    dist = dist or Distributed.get_instance()
    if dist.is_distributed():
        work_items.extend(_get_work_item_padding(work_items, dist))

    return work_items


@dataclass
class SliceWorkItemGriddedData:
    _loader: DataLoader[LoadedSliceWorkItem]
    variable_metadata: Mapping[str, VariableMetadata]
    all_times: xr.CFTimeIndex
    dtype: torch.dtype
    max_output_shape: tuple[int, ...]
    topography: Topography

    # TODO: currently no protocol or ABC for gridded data objects
    #       if we want to unify, we will need one and just raise
    #       NotImplementedError for unsupported methods.

    @property
    def loader(self) -> DataLoader[LoadedSliceWorkItem]:
        def on_device(work_item: LoadedSliceWorkItem) -> LoadedSliceWorkItem:
            return work_item.to_device()

        return SizedMap(on_device, self._loader)

    def get_generator(self) -> Iterator[tuple[LoadedSliceWorkItem, Topography]]:
        work_item: LoadedSliceWorkItem
        for work_item in self.loader:
            yield work_item, self.topography
