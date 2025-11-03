from dataclasses import dataclass
from itertools import product

import torch

from fme.core.distributed import Distributed
from fme.core.device import get_device

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
    is_padding: bool = False  # For even GPU distribution

    @staticmethod
    def _check_slice_no_none(slice_: slice) -> None:
        # check for valid work slices, no None and positive, monotonic
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
    def insert_slices(self) -> dict[str, slice]:
        """Get zarr position slices for writing output."""
        return {
            TIME_NAME: self.time_slice,
            ENSEMBLE_NAME: self.ens_slice,
        }

    @classmethod
    def with_batch(
        cls, work_item: "SliceWorkItem", batch: BatchData
    ) -> "LoadedWorkItem":
        """
        Create a LoadedWorkItem with batch data attached.
        """
        return LoadedWorkItem(
            time_slice=work_item.time_slice,
            ens_slice=work_item.ens_slice,
            is_padding=work_item.is_padding,
            batch=batch,
        )


@dataclass
class LoadedWorkItem(SliceWorkItem):
    """
    Work item with batch data attached, ready for generation.

    Created via _SliceWorkItem.with_batch() after loading data from the dataset.
    """

    batch: BatchData | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.batch is None:
            raise ValueError(
                "LoadedWorkItem must be created with batch data via with_batch()"
            )


class SliceItemDataset:
    """
    Dataset that loads a batch of data with metadata about time
    and ensemble slices for generation and saving out to Zarr.
    """

    def __init__(
        self,
        slice_items: list[SliceWorkItem],
        dataset: BatchItemDatasetAdapter,
        topography: Topography,
    ) -> None:
        self.slice_items = slice_items
        self.dataset = dataset
        self.topography = topography
        self._dtype = None

    def __len__(self) -> int:
        return len(self.slice_items)

    def __getitem__(self, idx: int) -> tuple[LoadedWorkItem, Topography]:
        work_spec = self.slice_items[idx]
        data_items = [self.dataset[i] for i in work_spec.time_indices]
        batch = BatchData.from_sequence(data_items).to_device()
        loaded_item = SliceWorkItem.with_batch(work_spec, batch)
        topography = self.topography.to_device(device=get_device())
        return loaded_item, topography

    @property
    def max_output_shape(self):
        first_item = self.slice_items[0]
        n_times = first_item.time_slice.stop - first_item.time_slice.start
        n_ensembles = first_item.ens_slice.stop - first_item.ens_slice.start
        spatial = self.topography.data.shape
        return (n_times, n_ensembles, *spatial)

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


def get_work_items(
    n_times: int, n_ens: int, n_items_per_gpu: int, dist: Distributed | None = None
) -> list[SliceWorkItem]:
    """
    Create work items for generation based on time and ensemble slices.

    Args:
        n_times: Number of time steps in the data
        n_ens: Total number of ensemble members to generate
        n_items_per_gpu: Number of time×ensemble items per GPU batch
        dist: Distributed instance for inferring padding work items (optional)
    """
    # 6 ens, 4 times, 4 items per gpu
    # item 0: (time=[0], ens=[0, 1, 2, 3])
    # item 1: (time=[0], ens=[4, 5])
    # item 2:
    work_items: list[SliceWorkItem] = []
    n_ens_per_slice = min(n_ens, n_items_per_gpu)
    n_time_per_slice = max(1, n_items_per_gpu // n_ens_per_slice)

    ens_slices = _generate_slices(n_ens, n_ens_per_slice)
    time_slices = _generate_slices(n_times, n_time_per_slice)

    work_items = [
        SliceWorkItem(time_sl, ens_sl)
        for (time_sl, ens_sl) in product(time_slices, ens_slices)
    ]

    # Pad work items to evenly distribute across GPUs
    dist = dist or Distributed.get_instance()
    if dist.is_distributed():
        remainder = len(work_items) % dist.world_size
        if remainder != 0:
            n_padding = dist.world_size - remainder
            # repeat last item as padding
            padding_item = SliceWorkItem(
                time_slice=work_items[-1].time_slice,
                ens_slice=work_items[-1].ens_slice,
                is_padding=True,
            )
            work_items.extend([padding_item] * n_padding)

    return work_items
