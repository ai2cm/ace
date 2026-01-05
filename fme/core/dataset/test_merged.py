import pytest
import torch

from fme.core.dataset.merged import MergedXarrayDataset
from fme.core.dataset.testing import TestingDataset


def test_merged_contains_all_data():
    datasets = [
        TestingDataset.new(
            n_times=10, varnames=[f"var_{i}"], sample_n_times=3, initial_epoch=None
        )
        for i in range(3)
    ]
    merged_dataset = MergedXarrayDataset(datasets)
    assert len(merged_dataset) == len(datasets[0])
    item = merged_dataset[0]
    for i in range(3):
        for var in datasets[i].data.keys():
            assert var in item[0]
            assert torch.equal(item[0][var], datasets[i].data[var][0:3])


def test_merged_set_epoch():
    datasets = [
        TestingDataset.new(
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
        TestingDataset.new(
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
        TestingDataset.new(
            n_times=10, varnames=[f"var_none"], sample_n_times=3, initial_epoch=None
        ),
        TestingDataset.new(
            n_times=10, varnames=[f"var_0"], sample_n_times=3, initial_epoch=0
        ),
    ]
    merged_dataset = MergedXarrayDataset(datasets)
    with pytest.raises(
        ValueError, match="All datasets in a merged dataset must have the same epoch."
    ):
        _ = merged_dataset[0]
