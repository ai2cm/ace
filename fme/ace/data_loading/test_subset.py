import numpy as np
import pytest
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.dataloader import (
    SlidingWindowDataLoader,
    TorchDataLoader,
    build_pool_schedule,
    get_skip_batches,
    get_stop_batches,
)
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
        output_n_timesteps=IntSchedule.from_constant(1),
        time_buffer=time_buffer,
        shuffle=shuffle,
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


@pytest.mark.parametrize(
    "sub_batches_per_contained_batch, start, "
    "expected_n_batches_to_skip, expected_n_sub_batches_to_skip",
    [
        pytest.param(3, 0, 0, 0, id="start_0"),
        pytest.param(3, 1, 0, 1, id="start_during_first_batch"),
        pytest.param(3, 4, 1, 1, id="start_during_second_batch"),
        pytest.param(3, 6, 2, 0, id="start_at_batch_boundary"),
    ],
)
def test_get_skip_batches(
    sub_batches_per_contained_batch,
    start,
    expected_n_batches_to_skip,
    expected_n_sub_batches_to_skip,
):
    n_batches_to_skip, n_sub_batches_to_skip = get_skip_batches(
        sub_batches_per_contained_batch, start
    )
    assert (
        n_batches_to_skip,
        n_sub_batches_to_skip,
    ) == (
        expected_n_batches_to_skip,
        expected_n_sub_batches_to_skip,
    )


@pytest.mark.parametrize(
    "sub_batches_per_contained_batch, stop, "
    "expected_n_batches_to_stop, expected_n_sub_batches_to_skip_last",
    [
        pytest.param(3, 2, 1, 1, id="stop_in_first_batch"),
        pytest.param(3, 4, 2, 2, id="stop_in_second_batch"),
        pytest.param(3, 5, 2, 1, id="stop_later_in_second_batch"),
        pytest.param(3, 6, 2, 0, id="stop_at_batch_boundary"),
    ],
)
def test_get_stop_batches(
    sub_batches_per_contained_batch,
    stop,
    expected_n_batches_to_stop,
    expected_n_sub_batches_to_skip_last,
):
    n_batches_to_stop, n_sub_batches_to_skip_last = get_stop_batches(
        sub_batches_per_contained_batch, stop
    )
    assert (
        n_batches_to_stop,
        n_sub_batches_to_skip_last,
    ) == (
        expected_n_batches_to_stop,
        expected_n_sub_batches_to_skip_last,
    )


# ---- pool_size > 1 tests ----


def get_pooled_sliding_window_data_loader(
    start: int,
    end: int,
    time_buffer: int,
    pool_size: int,
    shuffle: bool = False,
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


def test_build_pool_schedule_covers_all_samples():
    schedule = build_pool_schedule(
        n_input_batches=5, n_sub_batches=3, pool_size=2, shuffle=False, seed=0
    )
    assert schedule.shape == (15, 2)
    pairs = set(map(tuple, schedule.tolist()))
    expected = {(i, j) for i in range(5) for j in range(3)}
    assert pairs == expected


def test_build_pool_schedule_deterministic():
    a = build_pool_schedule(6, 3, pool_size=3, shuffle=True, seed=42)
    b = build_pool_schedule(6, 3, pool_size=3, shuffle=True, seed=42)
    np.testing.assert_array_equal(a, b)


def test_build_pool_schedule_pool_size_1_sequential():
    schedule = build_pool_schedule(4, 3, pool_size=1, shuffle=False, seed=0)
    expected = [(i, j) for i in range(4) for j in range(3)]
    assert list(map(tuple, schedule.tolist())) == expected


def test_build_pool_schedule_interleaves():
    schedule = build_pool_schedule(4, 3, pool_size=2, shuffle=False, seed=0)
    first_few_inputs = [int(schedule[i, 0]) for i in range(6)]
    assert len(set(first_few_inputs)) > 1


@pytest.mark.parametrize("pool_size", [2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
def test_pooled_determinism(pool_size, shuffle):
    loader = get_pooled_sliding_window_data_loader(
        0, 12, time_buffer=2, pool_size=pool_size, shuffle=shuffle
    )
    times_a = [get_batch_time(batch) for batch in loader]
    times_b = [get_batch_time(batch) for batch in loader]
    assert times_a == times_b


@pytest.mark.parametrize("pool_size", [2, 3])
@pytest.mark.parametrize(
    "n_times_total, time_buffer, start",
    [
        (12, 2, 0),
        (12, 2, 3),
        (12, 2, 6),
        (12, 2, 9),
    ],
)
def test_pooled_subset_start(pool_size, n_times_total, time_buffer, start):
    loader = get_pooled_sliding_window_data_loader(
        0, n_times_total, time_buffer=time_buffer, pool_size=pool_size, shuffle=False
    )
    full_times = [get_batch_time(batch) for batch in loader]
    subset = loader.subset(start_batch=start)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert len(subset_times) == n_times_total - start
    assert subset_times == full_times[start:]


@pytest.mark.parametrize("pool_size", [2, 3])
@pytest.mark.parametrize(
    "n_times_total, time_buffer, stop",
    [
        (12, 2, 3),
        (12, 2, 6),
        (12, 2, 9),
    ],
)
def test_pooled_subset_stop(pool_size, n_times_total, time_buffer, stop):
    loader = get_pooled_sliding_window_data_loader(
        0, n_times_total, time_buffer=time_buffer, pool_size=pool_size, shuffle=False
    )
    full_times = [get_batch_time(batch) for batch in loader]
    subset = loader.subset(stop_batch=stop)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert len(subset_times) == stop
    assert subset_times == full_times[:stop]


@pytest.mark.parametrize("pool_size", [2, 3])
@pytest.mark.parametrize(
    "n_times_total, time_buffer, start",
    [
        (12, 2, 0),
        (12, 2, 3),
        (12, 2, 6),
    ],
)
def test_pooled_subset_start_shuffled(pool_size, n_times_total, time_buffer, start):
    loader = get_pooled_sliding_window_data_loader(
        0, n_times_total, time_buffer=time_buffer, pool_size=pool_size, shuffle=True
    )
    full_times = [get_batch_time(batch) for batch in loader]
    reproduced = [get_batch_time(batch) for batch in loader]
    assert full_times == reproduced, "not deterministic"
    subset = loader.subset(start_batch=start)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert len(subset_times) == n_times_total - start
    assert subset_times == full_times[start:]


@pytest.mark.parametrize("pool_size", [2, 3])
@pytest.mark.parametrize(
    "n_times_total, time_buffer, stop",
    [
        (12, 2, 3),
        (12, 2, 6),
    ],
)
def test_pooled_subset_stop_shuffled(pool_size, n_times_total, time_buffer, stop):
    loader = get_pooled_sliding_window_data_loader(
        0, n_times_total, time_buffer=time_buffer, pool_size=pool_size, shuffle=True
    )
    full_times = [get_batch_time(batch) for batch in loader]
    reproduced = [get_batch_time(batch) for batch in loader]
    assert full_times == reproduced, "not deterministic"
    subset = loader.subset(stop_batch=stop)
    subset_times = [get_batch_time(batch) for batch in subset]
    assert len(subset_times) == stop
    assert subset_times == full_times[:stop]


def test_pooled_pool_size_exceeds_input_batches():
    loader = get_pooled_sliding_window_data_loader(
        0, 6, time_buffer=2, pool_size=10, shuffle=False
    )
    times = [get_batch_time(batch) for batch in loader]
    assert len(times) == 6
    assert len(set(times)) == 6


def test_pooled_all_samples_present():
    loader = get_pooled_sliding_window_data_loader(
        0, 12, time_buffer=2, pool_size=2, shuffle=True
    )
    times = [get_batch_time(batch) for batch in loader]
    assert sorted(times) == list(range(12))
