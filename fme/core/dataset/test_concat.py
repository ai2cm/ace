from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.testing import TestDataset


def test_concat_set_epoch():
    datasets = [
        TestDataset.new(n_times=10, varnames=["var1"], sample_n_times=3)
        for _ in range(3)
    ]
    concat_dataset = XarrayConcat(datasets)
    concat_dataset.set_epoch(5)
    for dataset in datasets:
        assert dataset.epoch == 5


def test_concat_len():
    datasets = [
        TestDataset.new(n_times=10, varnames=["var1"], sample_n_times=3)
        for _ in range(3)
    ]
    concat_dataset = XarrayConcat(datasets)
    assert len(concat_dataset) == sum(len(ds) for ds in datasets)
