import torch

from fme.core.dataset.subset import SubsetDataset
from fme.core.dataset.testing import TestingDataset


def test_subset_dataset_first_last_time():
    dataset = TestingDataset.new(n_times=20, varnames=["var1"], sample_n_times=2)

    subset = SubsetDataset(dataset, indices=[2, 3, 5, 7])

    assert subset.first_time == dataset.sample_start_times[2]
    assert subset.last_time == dataset.sample_start_times[7]


def test_subset_dataset_getitem():
    dataset = TestingDataset.new(n_times=20, varnames=["var1"], sample_n_times=2)

    subset = SubsetDataset(dataset, indices=[2, 3, 5, 7])

    for i, original_idx in enumerate([2, 3, 5, 7]):
        item_subset = subset[i]
        item_original = dataset[original_idx]
        torch.testing.assert_close(item_subset[0], item_original[0])


def test_subset_dataset_set_epoch():
    dataset = TestingDataset.new(n_times=20, varnames=["var1"], sample_n_times=2)

    subset = SubsetDataset(dataset, indices=[2, 3, 5, 7])

    subset.set_epoch(1)
    # Just ensure no error is raised and the underlying dataset's epoch is set
    assert dataset.epoch == 1


def test_subset_dataset_len():
    dataset = TestingDataset.new(n_times=20, varnames=["var1"], sample_n_times=2)

    subset = SubsetDataset(dataset, indices=[2, 3, 5, 7])

    assert len(subset) == 4
