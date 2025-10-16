import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.dataloader import SlidingWindowDataLoader, TorchDataLoader
from fme.ace.data_loading.getters import CollateFn
from fme.core.distributed import Distributed


def get_sample_tuples(start: int, end: int, times_per_batch: int):
    return [
        (
            {},
            xr.DataArray(
                data=[list(range(i, i + times_per_batch))],
                dims=["sample", "time"],
            ),
            set(),
        )
        for i in range(start, end)
    ]


def get_batch_time(batch: BatchData):
    return int(batch.time.values[0, 0])


class ListDataset(torch.utils.data.Dataset[BatchData]):
    def __init__(self, data: list[BatchData]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> BatchData:
        return self.data[idx]


def get_data_loader(
    start: int,
    end: int,
    times_per_batch: int = 1,
    time_buffer: int = 0,
    shuffle: bool = False,
):
    inner_times_per_batch = times_per_batch + time_buffer
    n_skip = time_buffer + 1
    dataset = ListDataset(
        get_sample_tuples(start, end, inner_times_per_batch)[::n_skip]
    )
    dist = Distributed.get_instance()
    sampler = dist.get_sampler(dataset, shuffle=shuffle)
    return TorchDataLoader(
        loader=torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=CollateFn(horizontal_dims=[]),
            batch_size=1,
        ),
        sampler=sampler,
        dataset=dataset,
    )


def get_sliding_window_data_loader(
    start: int, end: int, time_buffer: int, shuffle: bool = False
):
    if (end - start) % (time_buffer + 1) != 0:
        raise ValueError(
            "end - start must be divisible by time_buffer + 1, "
            f"got {end - start} and {time_buffer}"
        )
    loader = get_data_loader(
        start, end, times_per_batch=1, time_buffer=time_buffer, shuffle=shuffle
    )
    return SlidingWindowDataLoader(
        loader=loader,
        input_n_timesteps=time_buffer + 1,
        output_n_timesteps=1,
        shuffle=shuffle,
    )


def test_torch_data_loader_subset_sequential():
    loader = get_data_loader(0, 10, shuffle=False)
    subset = loader.subset(start=2)
    assert len(subset) == 8
    times = [get_batch_time(batch) for batch in subset]
    assert times == [2, 3, 4, 5, 6, 7, 8, 9]


def test_torch_data_loader_subset_random():
    loader = get_data_loader(0, 10, shuffle=True)
    original_times = [get_batch_time(batch) for batch in loader]
    reproduced_times = [get_batch_time(batch) for batch in loader]
    assert original_times == reproduced_times, "times not reproducible"
    subset = loader.subset(start=5)
    assert len(subset) == 5
    subset_times = [get_batch_time(batch) for batch in subset]
    assert len(set(subset_times)) == 5, (subset_times, set(subset_times))
    assert subset_times == original_times[5:]


@pytest.mark.parametrize(
    "n_times_total, time_buffer, start",
    [
        (12, 0, 0),
        (12, 2, 0),
        (12, 2, 3),
        (12, 2, 4),
    ],
)
def test_sliding_window_data_loader_subset_sequential(
    n_times_total, time_buffer, start
):
    loader = get_sliding_window_data_loader(0, n_times_total, time_buffer=time_buffer)
    assert len(loader) == n_times_total
    subset = loader.subset(start_batch=start)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert subset_times == list(range(start, n_times_total))


@pytest.mark.parametrize(
    "n_times_total, time_buffer, start",
    [
        (12, 0, 0),
        (12, 2, 0),
        (12, 2, 3),
        (12, 2, 4),
    ],
)
def test_sliding_window_data_loader_subset_random(n_times_total, time_buffer, start):
    loader = get_sliding_window_data_loader(
        0, n_times_total, time_buffer=time_buffer, shuffle=True
    )
    original_times = [get_batch_time(batch) for batch in loader]
    reproduced_times = [get_batch_time(batch) for batch in loader]
    assert original_times == reproduced_times, "times not reproducible"
    assert len(loader) == n_times_total
    subset = loader.subset(start_batch=start)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert len(subset_times) == n_times_total - start
    assert len(set(subset_times)) == n_times_total - start, (
        subset_times,
        set(subset_times),
    )
    assert subset_times == original_times[start:]
