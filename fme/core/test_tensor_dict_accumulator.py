import pytest
import torch

from fme.core.tensor_dict_accumulator import TensorDictAccumulator


def test_empty_state_returns_none():
    acc = TensorDictAccumulator()
    assert acc.get_sum() is None
    assert acc.get_mean() is None
    assert acc.count == 0


def test_single_add():
    acc = TensorDictAccumulator()
    acc.add({"a": torch.tensor([1.0, 2.0])})
    assert acc.count == 1
    total = acc.get_sum()
    mean = acc.get_mean()
    assert total is not None and mean is not None
    assert torch.equal(total["a"], torch.tensor([1.0, 2.0]))
    assert torch.equal(mean["a"], torch.tensor([1.0, 2.0]))


def test_multiple_adds_sum_and_mean():
    acc = TensorDictAccumulator()
    acc.add({"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([10.0])})
    acc.add({"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([20.0])})
    acc.add({"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([30.0])})
    assert acc.count == 3
    total = acc.get_sum()
    mean = acc.get_mean()
    assert total is not None and mean is not None
    assert torch.equal(total["a"], torch.tensor([9.0, 12.0]))
    assert torch.equal(total["b"], torch.tensor([60.0]))
    assert torch.equal(mean["a"], torch.tensor([3.0, 4.0]))
    assert torch.equal(mean["b"], torch.tensor([20.0]))


def test_key_mismatch_raises():
    acc = TensorDictAccumulator()
    acc.add({"a": torch.tensor(1.0), "b": torch.tensor(2.0)})
    with pytest.raises(ValueError, match="keys changed"):
        acc.add({"a": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="keys changed"):
        acc.add(
            {"a": torch.tensor(1.0), "b": torch.tensor(2.0), "c": torch.tensor(3.0)}
        )


def test_get_sum_does_not_alias_internal_state():
    acc = TensorDictAccumulator()
    acc.add({"a": torch.tensor([1.0])})
    returned = acc.get_sum()
    assert returned is not None
    returned["a"] = torch.tensor([999.0])
    after = acc.get_sum()
    assert after is not None
    assert torch.equal(after["a"], torch.tensor([1.0]))


def test_device_override():
    acc = TensorDictAccumulator(device=torch.device("cpu"))
    acc.add({"a": torch.tensor([1.0])})
    total = acc.get_sum()
    assert total is not None
    assert total["a"].device.type == "cpu"


def test_get_distributed_sum_non_distributed():
    acc = TensorDictAccumulator()
    acc.add({"b": torch.tensor([10.0]), "a": torch.tensor([1.0])})
    acc.add({"b": torch.tensor([20.0]), "a": torch.tensor([2.0])})
    result = acc.get_distributed_sum()
    assert list(result) == ["a", "b"]
    assert torch.equal(result["a"], torch.tensor([3.0]))
    assert torch.equal(result["b"], torch.tensor([30.0]))


def test_get_distributed_mean_non_distributed():
    acc = TensorDictAccumulator()
    acc.add({"a": torch.tensor([1.0]), "b": torch.tensor([10.0])})
    acc.add({"a": torch.tensor([3.0]), "b": torch.tensor([20.0])})
    result = acc.get_distributed_mean()
    assert torch.equal(result["a"], torch.tensor([2.0]))
    assert torch.equal(result["b"], torch.tensor([15.0]))


def test_get_distributed_raises_when_empty():
    acc = TensorDictAccumulator()
    with pytest.raises(ValueError, match="No values"):
        acc.get_distributed_sum()
    with pytest.raises(ValueError, match="No values"):
        acc.get_distributed_mean()
