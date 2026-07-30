from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from fme.ace.data_loading.pool import PooledSequence, build_pool_schedule


class ListSource:
    """A simple Subsettable source backed by a list of containers.

    Each item represents a "fat batch" containing ``n_sub`` sub-items,
    mirroring how the real data loader yields batches with multiple
    timesteps that get sliced into smaller windows.
    """

    def __init__(self, items: list[list[int]]):
        self._items = items

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def subset(
        self, start_batch: int | None = None, stop_batch: int | None = None
    ) -> ListSource:
        return ListSource(self._items[start_batch:stop_batch])


def _extract(batch: list[int], offset: int) -> int:
    return batch[offset]


def make_pooled(
    n_items: int,
    n_sub: int,
    pool_size: int,
    shuffle: bool = False,
    seed: int = 0,
    prepare=None,
) -> PooledSequence[list[int], int]:
    source = ListSource(
        [list(range(i * n_sub, (i + 1) * n_sub)) for i in range(n_items)]
    )
    schedule = build_pool_schedule(
        n_items, n_sub, pool_size, shuffle=shuffle, seed=seed
    )
    return PooledSequence(
        source=source,
        schedule=schedule,
        pool_size=pool_size,
        extract=_extract,
        prepare=prepare,
    )


# --- build_pool_schedule tests ---


@pytest.mark.parametrize(
    "n_input,n_sub,pool_size",
    [
        (4, 3, 1),
        (4, 3, 2),
        (4, 3, 4),
        (5, 3, 2),
        (8, 5, 3),
        (10, 2, 4),
        (3, 6, 2),
        (1, 5, 1),
        (6, 1, 3),
    ],
)
@pytest.mark.parametrize("shuffle", [True, False])
def test_build_pool_schedule_covers_all_samples(n_input, n_sub, pool_size, shuffle):
    """Every (input_idx, offset) pair appears exactly once."""
    schedule = build_pool_schedule(n_input, n_sub, pool_size, shuffle=shuffle, seed=7)
    assert schedule.shape == (n_input * n_sub, 2)
    pairs = sorted(map(tuple, schedule.tolist()))
    expected = sorted((i, j) for i in range(n_input) for j in range(n_sub))
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


def test_build_pool_schedule_shuffled_interleaves():
    schedule = build_pool_schedule(4, 3, pool_size=4, shuffle=True, seed=0)
    source_indices = schedule[:, 0].tolist()
    runs = sum(
        1
        for i in range(1, len(source_indices))
        if source_indices[i] != source_indices[i - 1]
    )
    assert runs >= 4


# --- PooledSequence tests ---


def test_len():
    pooled = make_pooled(n_items=4, n_sub=3, pool_size=2)
    assert len(pooled) == 12


def test_sequential_iteration_pool_size_1():
    pooled = make_pooled(n_items=4, n_sub=3, pool_size=1, shuffle=False)
    results = list(pooled)
    assert results == list(range(12))


def test_all_sub_items_present():
    pooled = make_pooled(n_items=5, n_sub=3, pool_size=2, shuffle=True, seed=7)
    results = list(pooled)
    assert sorted(results) == list(range(15))
    assert len(results) == 15


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
def test_determinism(pool_size, shuffle):
    a = list(make_pooled(4, 3, pool_size, shuffle=shuffle, seed=42))
    b = list(make_pooled(4, 3, pool_size, shuffle=shuffle, seed=42))
    assert a == b


def test_repeated_iteration():
    pooled = make_pooled(n_items=3, n_sub=2, pool_size=2, shuffle=True, seed=0)
    first = list(pooled)
    second = list(pooled)
    assert first == second


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
def test_cache_eviction(pool_size, shuffle):
    pooled = make_pooled(n_items=6, n_sub=3, pool_size=pool_size, shuffle=shuffle)
    iter(pooled)
    max_pool_size = 0
    for _ in pooled:
        max_pool_size = max(max_pool_size, len(pooled._pool))
    assert max_pool_size <= pool_size
    assert len(pooled._pool) == 0


def test_pool_size_exceeds_source():
    pooled = make_pooled(n_items=2, n_sub=3, pool_size=10, shuffle=False)
    results = list(pooled)
    assert sorted(results) == list(range(6))


def test_prepare_is_called():
    prepared: list[list[int]] = []

    def track_prepare(batch: list[int]) -> list[int]:
        prepared.append(batch)
        return [v * 10 for v in batch]

    pooled = make_pooled(
        n_items=3, n_sub=2, pool_size=1, shuffle=False, prepare=track_prepare
    )
    results = list(pooled)
    assert prepared == [[0, 1], [2, 3], [4, 5]]
    assert results == [0, 10, 20, 30, 40, 50]


# --- subset tests ---


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("start", [0, 1, 2, 4, 5, 7])
def test_subset_start(pool_size, shuffle, start):
    pooled = make_pooled(4, 3, pool_size, shuffle=shuffle, seed=0)
    full = list(pooled)
    sub = pooled.subset(start=start)
    assert len(sub) == len(full) - start
    assert list(sub) == full[start:]


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("stop", [1, 2, 4, 5, 7])
def test_subset_stop(pool_size, shuffle, stop):
    pooled = make_pooled(4, 3, pool_size, shuffle=shuffle, seed=0)
    full = list(pooled)
    sub = pooled.subset(stop=stop)
    assert len(sub) == stop
    assert list(sub) == full[:stop]


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
def test_subset_start_and_stop(pool_size, shuffle):
    pooled = make_pooled(4, 3, pool_size, shuffle=shuffle, seed=0)
    full = list(pooled)
    sub = pooled.subset(start=2, stop=8)
    assert len(sub) == 6
    assert list(sub) == full[2:8]


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
def test_subset_of_subset(pool_size, shuffle):
    pooled = make_pooled(4, 3, pool_size, shuffle=shuffle, seed=0)
    full = list(pooled)
    outer = pooled.subset(start=1, stop=10)
    inner = outer.subset(start=2, stop=6)
    assert len(inner) == 4
    assert list(inner) == full[3:7]


def test_subset_empty():
    pooled = make_pooled(4, 3, pool_size=2, shuffle=False)
    sub = pooled.subset(start=5, stop=5)
    assert len(sub) == 0
    assert list(sub) == []


@pytest.mark.parametrize("pool_size", [2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("start", [0, 3, 5, 7])
def test_subset_pool_invariant(pool_size, shuffle, start):
    """Pool occupancy must never exceed pool_size when iterating a subset."""
    pooled = make_pooled(6, 3, pool_size, shuffle=shuffle, seed=0)
    sub = pooled.subset(start=start)
    iter(sub)
    max_pool = 0
    for _ in sub:
        max_pool = max(max_pool, len(sub._pool))
    assert max_pool <= pool_size


def test_subset_none_returns_self():
    pooled = make_pooled(4, 3, pool_size=2)
    assert pooled.subset() is pooled


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
def test_subset_stop_beyond_length(pool_size, shuffle):
    pooled = make_pooled(4, 3, pool_size, shuffle=shuffle, seed=0)
    full = list(pooled)
    sub = pooled.subset(stop=100)
    assert len(sub) == len(full)
    assert list(sub) == full


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("shuffle", [True, False])
def test_subset_of_subset_stop_beyond_length(pool_size, shuffle):
    pooled = make_pooled(4, 3, pool_size, shuffle=shuffle, seed=0)
    full = list(pooled)
    outer = pooled.subset(start=2, stop=8)
    inner = outer.subset(stop=100)
    assert len(inner) == 6
    assert list(inner) == full[2:8]
