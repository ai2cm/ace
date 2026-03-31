import torch

from fme.core.generics.dataset import GenericDataset


class _Dataset(GenericDataset[int]):
    def __init__(self, data: list[int]):
        self.data = data
        self.epoch = 0

    def __getitem__(self, index) -> int:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @property
    def first_time(self):
        return 0

    @property
    def last_time(self):
        return len(self.data) - 1


def test_generic_dataset_torch_dataset():
    data = list(range(10))
    dataset = _Dataset(data)

    # Verify torch_dataset property
    torch_ds = dataset.torch_dataset
    assert isinstance(torch_ds, torch.utils.data.Dataset)
    assert len(torch_ds) == 10
    for i in range(10):
        assert torch_ds[i] == i


def test_generic_dataset_subset_once():
    data = list(range(10))
    dataset = _Dataset(data)

    indices = [1, 3, 5, 7, 9]
    subset = dataset.subset(indices)

    assert len(subset) == 5

    torch_subset = subset.torch_dataset
    assert isinstance(torch_subset, torch.utils.data.Dataset)
    assert len(torch_subset) == 5

    # Check values
    expected_values = [1, 3, 5, 7, 9]
    for i, expected in enumerate(expected_values):
        assert torch_subset[i] == expected
        assert subset[i] == expected


def test_generic_dataset_subset_twice():
    data = list(range(10))
    dataset = _Dataset(data)

    # First subset: keep even numbers [0, 2, 4, 6, 8]
    indices1 = [0, 2, 4, 6, 8]
    subset1 = dataset.subset(indices1)

    # Second subset: keep indices 1 and 3 from the subset -> [2, 6]
    indices2 = [1, 3]
    subset2 = subset1.subset(indices2)

    assert len(subset2) == 2

    torch_subset2 = subset2.torch_dataset
    assert isinstance(torch_subset2, torch.utils.data.Dataset)

    expected_values = [2, 6]
    for i, expected in enumerate(expected_values):
        assert subset2[i] == expected
        assert torch_subset2[i] == expected
