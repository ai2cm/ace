import math

import pytest
import torch

from fme.core.dataset.merged import MergedXarrayDataset, TimePaddedMergedDataset
from fme.core.dataset.testing import MockDataset, assert_dataset_item_length


def test_merged_contains_all_data():
    datasets = [
        MockDataset.new(
            n_times=10, varnames=[f"var_{i}"], sample_n_times=3, initial_epoch=None
        )
        for i in range(3)
    ]
    merged_dataset = MergedXarrayDataset(datasets)
    assert len(merged_dataset) == len(datasets[0])
    item = merged_dataset[0]
    assert_dataset_item_length(item)
    for i in range(3):
        for var in datasets[i].data.keys():
            assert var in item[0]
            assert torch.equal(item[0][var], datasets[i].data[var][0:3])


def test_merged_propagates_metadata():
    datasets = [
        MockDataset.new(
            n_times=10,
            varnames=[f"var_{i}"],
            sample_n_times=3,
            labels={"label_a"} if i == 0 else {"label_b"},
            initial_epoch=5,
            missing_names=frozenset({f"missing_{i}"}),
        )
        for i in range(2)
    ]
    merged_dataset = MergedXarrayDataset(datasets)
    item = merged_dataset[0]
    assert_dataset_item_length(item)
    data, time, labels, epoch, missing_names = item
    assert set(data.keys()) == {"var_0", "var_1"}
    assert time.shape == (3,)
    assert labels == {"label_a", "label_b"}
    assert epoch == 5
    assert missing_names == frozenset({"missing_0", "missing_1"})


def test_merged_set_epoch():
    datasets = [
        MockDataset.new(
            n_times=10, varnames=[f"var_{i}"], sample_n_times=3, initial_epoch=None
        )
        for i in range(3)
    ]
    merged_dataset = MergedXarrayDataset(datasets)
    for dataset in datasets:
        assert dataset.epoch is None
    merged_dataset.set_epoch(5)
    for dataset in datasets:
        assert dataset.epoch == 5


def test_merged_raises_on_different_epochs():
    datasets = [
        MockDataset.new(
            n_times=10, varnames=[f"var_{i}"], sample_n_times=3, initial_epoch=i
        )
        for i in range(3)
    ]
    merged_dataset = MergedXarrayDataset(datasets)
    with pytest.raises(
        ValueError, match="All datasets in a merged dataset must have the same epoch."
    ):
        _ = merged_dataset[0]


def test_merged_raises_on_different_epochs_with_none():
    datasets = [
        MockDataset.new(
            n_times=10, varnames=["var_none"], sample_n_times=3, initial_epoch=None
        ),
        MockDataset.new(
            n_times=10, varnames=["var_0"], sample_n_times=3, initial_epoch=0
        ),
    ]
    merged_dataset = MergedXarrayDataset(datasets)
    with pytest.raises(
        ValueError, match="All datasets in a merged dataset must have the same epoch."
    ):
        _ = merged_dataset[0]


def _make_aligned_pair(short_n: int, long_n: int, n_starts: int = 4):
    """Create two MockDatasets whose sample_start_times match, but with
    differing sample_n_times. ``n_times = n_starts + sample_n_times - 1`` so
    that ``sample_start_times`` length is ``n_starts`` for both. Data is
    cast to float so NaN padding is representable.
    """
    short_ds = MockDataset.new(
        n_times=n_starts + short_n - 1,
        varnames=["short_var"],
        sample_n_times=short_n,
        initial_epoch=None,
    )
    long_ds = MockDataset.new(
        n_times=n_starts + long_n - 1,
        varnames=["long_var"],
        sample_n_times=long_n,
        initial_epoch=None,
    )
    short_ds.data = {k: v.float() for k, v in short_ds.data.items()}
    long_ds.data = {k: v.float() for k, v in long_ds.data.items()}
    return short_ds, long_ds


def test_time_padded_merged_pads_shorter_with_nan():
    short_ds, long_ds = _make_aligned_pair(short_n=3, long_n=5)
    merged = TimePaddedMergedDataset([short_ds, long_ds])
    assert merged.sample_n_times == 5
    item = merged[0]
    assert_dataset_item_length(item)
    tensors = item[0]
    assert set(tensors.keys()) == {"short_var", "long_var"}
    assert tensors["long_var"].shape[0] == 5
    assert tensors["short_var"].shape[0] == 5
    # first 3 entries match the underlying short dataset
    assert torch.equal(tensors["short_var"][:3], short_ds.data["short_var"][:3])
    # trailing entries are NaN
    for i in range(3, 5):
        assert math.isnan(float(tensors["short_var"][i]))
    # long var unchanged
    assert torch.equal(tensors["long_var"], long_ds.data["long_var"][:5])


def test_time_padded_merged_propagates_metadata():
    short_ds, long_ds = _make_aligned_pair(short_n=3, long_n=5)
    short_ds.labels = {"short"}
    long_ds.labels = {"long"}
    short_ds.epoch = 7
    long_ds.epoch = 7
    short_ds.missing_names = frozenset({"x"})
    long_ds.missing_names = frozenset({"y"})
    merged = TimePaddedMergedDataset([short_ds, long_ds])
    item = merged[0]
    assert_dataset_item_length(item)
    data, time, labels, epoch, missing_names = item
    assert time.shape == (5,)
    assert labels == {"long"}
    assert epoch == 7
    assert missing_names == frozenset({"x", "y"})


def test_time_padded_merged_canonical_picks_longest_when_first_is_short():
    short_ds, long_ds = _make_aligned_pair(short_n=3, long_n=5)
    # put the short one first; canonical should still be the long one
    merged = TimePaddedMergedDataset([short_ds, long_ds])
    _, time, _, _, _ = merged[0]
    assert time.shape == (5,)


def test_time_padded_merged_canonical_picks_longest_when_first_is_long():
    short_ds, long_ds = _make_aligned_pair(short_n=3, long_n=5)
    merged = TimePaddedMergedDataset([long_ds, short_ds])
    _, time, _, _, _ = merged[0]
    assert time.shape == (5,)


def test_time_padded_merged_no_padding_when_lengths_equal():
    a = MockDataset.new(
        n_times=10, varnames=["a"], sample_n_times=4, initial_epoch=None
    )
    b = MockDataset.new(
        n_times=10, varnames=["b"], sample_n_times=4, initial_epoch=None
    )
    a.data = {k: v.float() for k, v in a.data.items()}
    b.data = {k: v.float() for k, v in b.data.items()}
    merged = TimePaddedMergedDataset([a, b])
    assert merged.sample_n_times == 4
    item = merged[0]
    assert item[0]["a"].shape[0] == 4
    assert item[0]["b"].shape[0] == 4
    assert not torch.isnan(item[0]["a"]).any()
    assert not torch.isnan(item[0]["b"]).any()


def test_time_padded_merged_disjoint_names_required():
    a = MockDataset.new(
        n_times=10, varnames=["dup"], sample_n_times=3, initial_epoch=None
    )
    b = MockDataset.new(
        n_times=12, varnames=["dup"], sample_n_times=5, initial_epoch=None
    )
    with pytest.raises(ValueError, match="Variable names must be unique"):
        TimePaddedMergedDataset([a, b])


def test_time_padded_merged_set_epoch_propagates():
    short_ds, long_ds = _make_aligned_pair(short_n=3, long_n=5)
    merged = TimePaddedMergedDataset([short_ds, long_ds])
    assert short_ds.epoch is None
    assert long_ds.epoch is None
    merged.set_epoch(3)
    assert short_ds.epoch == 3
    assert long_ds.epoch == 3


def test_time_padded_merged_set_epoch_recomputes_canonical():
    # both datasets share the same long time axis so that changing
    # sample_n_times always yields compatible sample_start_times
    n_times = 20
    ds_a = MockDataset.new(
        n_times=n_times,
        varnames=["a"],
        sample_n_times=5,
        initial_epoch=None,
    )
    ds_b = MockDataset.new(
        n_times=n_times,
        varnames=["b"],
        sample_n_times=3,
        initial_epoch=None,
    )
    ds_a.data = {k: v.float() for k, v in ds_a.data.items()}
    ds_b.data = {k: v.float() for k, v in ds_b.data.items()}
    merged = TimePaddedMergedDataset([ds_a, ds_b])
    assert merged.sample_n_times == 5
    assert merged._canonical_idx == 0

    # simulate a schedule milestone shrinking ds_a below ds_b
    ds_a._sample_n_times = 2
    merged.set_epoch(1)
    assert merged.sample_n_times == 3
    assert merged._canonical_idx == 1

    # simulate ds_a growing back to be longest
    ds_a._sample_n_times = 7
    merged.set_epoch(2)
    assert merged.sample_n_times == 7
    assert merged._canonical_idx == 0


def test_time_padded_merged_validate_inference_length_propagates():
    short_ds, long_ds = _make_aligned_pair(short_n=3, long_n=5, n_starts=4)
    merged = TimePaddedMergedDataset([short_ds, long_ds])
    # within bounds for both
    merged.validate_inference_length(max_start_index=0, max_window_len=4)
    # long_ds has 8 timesteps; this should fail
    with pytest.raises(ValueError, match="Inference length exceeds"):
        merged.validate_inference_length(max_start_index=0, max_window_len=100)


def test_time_padded_merged_mismatched_start_times_raises():
    # Two datasets that start on different days -> the canonical's prefix is
    # not a prefix of the shorter one's start times.
    short_ds = MockDataset.new(
        n_times=8, varnames=["a"], sample_n_times=3, initial_epoch=None
    )
    long_ds = MockDataset.new(
        n_times=8, varnames=["b"], sample_n_times=5, initial_epoch=None
    )
    # shift the short dataset's times by one day so prefix won't match
    short_ds.time = short_ds.time + (long_ds.time[1] - long_ds.time[0])
    with pytest.raises(ValueError, match="prefix"):
        TimePaddedMergedDataset([short_ds, long_ds])


def test_time_padded_merged_short_dataset_can_have_more_starts():
    """The shorter sub-dataset may have more valid start times than the
    canonical (longer) sub-dataset, since fewer trailing timesteps need to
    fit into a sample.
    """
    short_ds = MockDataset.new(
        n_times=10, varnames=["short"], sample_n_times=3, initial_epoch=None
    )
    long_ds = MockDataset.new(
        n_times=10, varnames=["long"], sample_n_times=5, initial_epoch=None
    )
    short_ds.data = {k: v.float() for k, v in short_ds.data.items()}
    long_ds.data = {k: v.float() for k, v in long_ds.data.items()}
    # short_ds has 8 valid start positions, long_ds has 6
    assert len(short_ds.sample_start_times) == 8
    assert len(long_ds.sample_start_times) == 6
    merged = TimePaddedMergedDataset([short_ds, long_ds])
    # canonical is long_ds; merged length = canonical length
    assert len(merged) == 6
    item = merged[0]
    assert item[0]["short"].shape[0] == 5  # padded from 3
    assert item[0]["long"].shape[0] == 5


def test_time_padded_merged_get_sample_by_time_slice_no_padding():
    short_ds, long_ds = _make_aligned_pair(short_n=3, long_n=5)
    merged = TimePaddedMergedDataset([short_ds, long_ds])
    # Slice of length 2: each underlying dataset returns 2 timesteps; no padding
    tensors, time, _, _, _ = merged.get_sample_by_time_slice(slice(0, 2))
    assert tensors["short_var"].shape[0] == 2
    assert tensors["long_var"].shape[0] == 2
    assert time.shape == (2,)
    # No NaNs introduced
    assert not torch.isnan(tensors["short_var"]).any()
    assert not torch.isnan(tensors["long_var"]).any()


def test_time_padded_merged_raises_on_different_epochs():
    short_ds, long_ds = _make_aligned_pair(short_n=3, long_n=5)
    short_ds.epoch = 1
    long_ds.epoch = 2
    merged = TimePaddedMergedDataset([short_ds, long_ds])
    with pytest.raises(ValueError, match="must have the same"):
        _ = merged[0]


def test_time_padded_merged_empty_raises():
    with pytest.raises(ValueError, match="at least one dataset"):
        TimePaddedMergedDataset([])
