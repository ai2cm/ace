import pytest
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.dataloader import SlidingWindowDataLoader, TorchDataLoader
from fme.ace.data_loading.getters import CollateFn
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.subset import SubsetDataset
from fme.core.dataset.testing import MockDataset
from fme.core.distributed import Distributed


def get_sample_tuples(start: int, end: int, times_per_batch: int):
    return [
        (
            {},
            xr.DataArray(
                data=[list(range(i, i + times_per_batch))],
                dims=["sample", "time"],
            ),
            None,
        )
        for i in range(start, end)
    ]


def get_batch_time(batch: BatchData):
    return MockDataset.time_to_int(batch.time.values[0, 0])


def get_data_loader(
    start: int,
    end: int,
    times_per_batch: int = 1,
    time_buffer: int = 0,
    shuffle: bool = False,
):
    inner_times_per_batch = times_per_batch + time_buffer
    n_skip = time_buffer + 1
    dataset: MockDataset | SubsetDataset = MockDataset.new(
        n_times=end - start,
        varnames=["var1"],
        sample_n_times=inner_times_per_batch,
    )
    dataset = SubsetDataset(dataset, indices=list(range(start, end, n_skip)))
    dist = Distributed.get_instance()
    sampler = dist.get_sampler(dataset, shuffle=shuffle)
    return TorchDataLoader(
        dataset=dataset,
        collate_fn=CollateFn(horizontal_dims=[]),
        batch_size=1,
        sampler=sampler,
    )


def get_sliding_window_data_loader(
    start: int,
    end: int,
    time_buffer: int,
    shuffle: bool = False,
    pool_size: int = 1,
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
        output_n_timesteps=IntSchedule.from_constant(1),
        time_buffer=time_buffer,
        shuffle=shuffle,
        pool_size=pool_size,
    )


def test_torch_data_loader_subset_sequential():
    loader = get_data_loader(0, 10, shuffle=False)
    assert len(loader) == 10
    subset = loader.subset(start_batch=2)
    assert len(subset) == 8
    times = [get_batch_time(batch) for batch in subset]
    assert times == [2, 3, 4, 5, 6, 7, 8, 9]


def test_torch_data_loader_subset_random():
    loader = get_data_loader(0, 10, shuffle=True)
    original_times = [get_batch_time(batch) for batch in loader]
    reproduced_times = [get_batch_time(batch) for batch in loader]
    assert original_times == reproduced_times, "times not reproducible"
    subset = loader.subset(start_batch=5)
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
    "n_times_total, time_buffer, stop",
    [
        (12, 0, 6),
        (12, 2, 6),
        (12, 2, 3),
        (12, 2, 4),
    ],
)
def test_sliding_window_data_loader_subset_sequential_stop(
    n_times_total, time_buffer, stop
):
    loader = get_sliding_window_data_loader(0, n_times_total, time_buffer=time_buffer)
    assert len(loader) == n_times_total
    subset = loader.subset(stop_batch=stop)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert subset_times == list(range(0, stop))


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


@pytest.mark.parametrize(
    "n_times_total, time_buffer, stop",
    [
        (12, 0, 6),
        (12, 2, 6),
        (12, 2, 3),
        (12, 2, 4),
    ],
)
def test_sliding_window_data_loader_subset_random_stop(
    n_times_total, time_buffer, stop
):
    loader = get_sliding_window_data_loader(
        0, n_times_total, time_buffer=time_buffer, shuffle=True
    )
    original_times = [get_batch_time(batch) for batch in loader]
    reproduced_times = [get_batch_time(batch) for batch in loader]
    assert original_times == reproduced_times, "times not reproducible"
    assert len(loader) == n_times_total
    subset = loader.subset(stop_batch=stop)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert len(subset_times) == stop
    assert len(set(subset_times)) == stop, (subset_times, set(subset_times))
    assert subset_times == original_times[:stop]


@pytest.mark.parametrize("pool_size", [2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
def test_sliding_window_pool_size_gt1_all_times_present(pool_size, shuffle):
    n_times = 12
    time_buffer = 2
    loader = get_sliding_window_data_loader(
        0, n_times, time_buffer=time_buffer, shuffle=shuffle, pool_size=pool_size
    )
    times = [get_batch_time(batch) for batch in loader]
    assert sorted(times) == list(range(n_times))


@pytest.mark.parametrize("pool_size", [2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("start", [0, 3, 4])
def test_sliding_window_pool_size_gt1_subset_start(pool_size, shuffle, start):
    n_times = 12
    time_buffer = 2
    loader = get_sliding_window_data_loader(
        0, n_times, time_buffer=time_buffer, shuffle=shuffle, pool_size=pool_size
    )
    original_times = [get_batch_time(batch) for batch in loader]
    subset = loader.subset(start_batch=start)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert len(subset_times) == n_times - start
    assert subset_times == original_times[start:]


@pytest.mark.parametrize("pool_size", [2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("stop", [3, 4, 6])
def test_sliding_window_pool_size_gt1_subset_stop(pool_size, shuffle, stop):
    n_times = 12
    time_buffer = 2
    loader = get_sliding_window_data_loader(
        0, n_times, time_buffer=time_buffer, shuffle=shuffle, pool_size=pool_size
    )
    original_times = [get_batch_time(batch) for batch in loader]
    subset = loader.subset(stop_batch=stop)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert len(subset_times) == stop
    assert subset_times == original_times[:stop]
