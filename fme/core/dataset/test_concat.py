import datetime

import pytest

from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.testing import TestingDataset


def test_concat_set_epoch():
    datasets = [
        TestingDataset.new(n_times=10, varnames=["var1"], sample_n_times=3)
        for _ in range(3)
    ]
    concat_dataset = XarrayConcat(datasets)
    concat_dataset.set_epoch(5)
    for dataset in datasets:
        assert dataset.epoch == 5


def test_concat_len():
    datasets = [
        TestingDataset.new(n_times=10, varnames=["var1"], sample_n_times=3)
        for _ in range(3)
    ]
    concat_dataset = XarrayConcat(datasets)
    assert len(concat_dataset) == sum(len(ds) for ds in datasets)


@pytest.mark.parametrize(
    "strict,context",
    [
        (True, pytest.raises(ValueError, match=r"Inconsistent timesteps.*")),
        (False, pytest.warns(UserWarning, match=r"Inconsistent timesteps.*")),
    ],
)
def test_concat_strict(strict, context):
    dataset = TestingDataset.new(n_times=10, varnames=["var1"], sample_n_times=3)
    new_properties = dataset.properties.copy()
    new_properties.timestep = datetime.timedelta(hours=6)
    new_dataset = TestingDataset.new(
        n_times=10, varnames=["var1"], sample_n_times=3, properties=new_properties
    )
    datasets = [dataset, new_dataset]
    with context:
        _ = XarrayConcat(datasets, strict=strict)
