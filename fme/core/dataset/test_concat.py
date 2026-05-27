import datetime

import pytest
import torch

from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.testing import MockDataset, assert_dataset_item_length


def test_concat_set_epoch():
    datasets = [
        MockDataset.new(n_times=10, varnames=["var1"], sample_n_times=3)
        for _ in range(3)
    ]
    concat_dataset = XarrayConcat(datasets)
    concat_dataset.set_epoch(5)
    for dataset in datasets:
        assert dataset.epoch == 5


def test_concat_len():
    datasets = [
        MockDataset.new(n_times=10, varnames=["var1"], sample_n_times=3)
        for _ in range(3)
    ]
    concat_dataset = XarrayConcat(datasets)
    assert len(concat_dataset) == sum(len(ds) for ds in datasets)


def test_concat_getitem_propagates_metadata():
    datasets = [
        MockDataset.new(
            n_times=10,
            varnames=["var1"],
            sample_n_times=3,
            labels={"src_a"},
            initial_epoch=2,
            missing_names=frozenset({"var2"}),
        )
        for _ in range(2)
    ]
    concat_dataset = XarrayConcat(datasets)
    item = concat_dataset[0]
    assert_dataset_item_length(item)
    data, time, labels, epoch, missing_names = item
    assert set(data.keys()) == {"var1"}
    torch.testing.assert_close(data["var1"], datasets[0].data["var1"][:3])
    assert time.shape == (3,)
    assert labels == {"src_a"}
    assert epoch == 2
    assert missing_names == frozenset({"var2"})


@pytest.mark.parametrize(
    "strict,context",
    [
        (True, pytest.raises(ValueError, match=r"Inconsistent timesteps.*")),
        (False, pytest.warns(UserWarning, match=r"Inconsistent timesteps.*")),
    ],
)
def test_concat_strict(strict, context):
    dataset = MockDataset.new(n_times=10, varnames=["var1"], sample_n_times=3)
    new_properties = dataset.properties.copy()
    new_properties.timestep = datetime.timedelta(hours=6)
    new_dataset = MockDataset.new(
        n_times=10, varnames=["var1"], sample_n_times=3, properties=new_properties
    )
    datasets = [dataset, new_dataset]
    with context:
        _ = XarrayConcat(datasets, strict=strict)
