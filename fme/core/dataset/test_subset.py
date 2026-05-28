import torch

from fme.core.dataset.subset import SubsetDataset
from fme.core.dataset.testing import MockDataset, assert_dataset_item_length


def test_subset_dataset_first_last_time():
    dataset = MockDataset.new(n_times=20, varnames=["var1"], sample_n_times=2)

    subset = SubsetDataset(dataset, indices=[2, 3, 5, 7])

    assert subset.first_time == dataset.sample_start_times[2]
    assert subset.last_time == dataset.sample_start_times[7]


def test_subset_dataset_getitem():
    dataset = MockDataset.new(
        n_times=20,
        varnames=["var1"],
        sample_n_times=2,
        labels={"src_a"},
        initial_epoch=3,
        missing_names=frozenset({"var2"}),
    )

    subset = SubsetDataset(dataset, indices=[2, 3, 5, 7])

    for i, original_idx in enumerate([2, 3, 5, 7]):
        item_subset = subset[i]
        item_original = dataset[original_idx]
        assert_dataset_item_length(item_subset)
        data, time, labels, epoch, missing_names = item_subset
        orig_data, orig_time, orig_labels, orig_epoch, orig_missing = item_original
        torch.testing.assert_close(data, orig_data)
        assert time.equals(orig_time)
        assert labels == orig_labels
        assert epoch == orig_epoch
        assert missing_names == orig_missing


def test_subset_dataset_set_epoch():
    dataset = MockDataset.new(n_times=20, varnames=["var1"], sample_n_times=2)

    subset = SubsetDataset(dataset, indices=[2, 3, 5, 7])

    subset.set_epoch(1)
    # Just ensure no error is raised and the underlying dataset's epoch is set
    assert dataset.epoch == 1


def test_subset_dataset_len():
    dataset = MockDataset.new(n_times=20, varnames=["var1"], sample_n_times=2)

    subset = SubsetDataset(dataset, indices=[2, 3, 5, 7])

    assert len(subset) == 4
