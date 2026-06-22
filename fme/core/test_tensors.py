import pytest
import torch

from fme.core.tensors import (
    assert_dict_allclose,
    fold_sized_ensemble_dim,
    repeat_interleave_batch_dim,
    unfold_ensemble_dim,
)
from fme.core.typing_ import EnsembleTensorDict


@pytest.mark.parametrize(
    "a, b",
    [
        (
            {"a": torch.tensor(1)},
            {"a": torch.tensor(1)},
        ),
        (
            {"a": torch.tensor(1), "b": {"c": torch.tensor(2)}},
            {"a": torch.tensor(1), "b": {"c": torch.tensor(2)}},
        ),
    ],
)
def test_assert_dict_allclose_passes_for_equal_dicts(a, b):
    assert_dict_allclose(a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        (
            {"a": torch.tensor(1), "b": {"c": torch.tensor(2)}},
            {"a": torch.tensor(1), "b": {"c": torch.tensor(3)}},
        ),
        (
            {"a": torch.tensor(1)},
            {"b": torch.tensor(1)},
        ),
        (
            {"a": torch.tensor(1)},
            {"a": torch.tensor(2)},
        ),
    ],
)
def test_assert_allclose_raises_for_unequal_dicts(a, b):
    with pytest.raises(AssertionError):
        assert_dict_allclose(a, b)


@pytest.mark.parametrize("repeats", [1, 3])
def test_ensemble_dim_operations_give_correct_data_order(repeats: int):
    n_samples = 3
    data = {
        "a": torch.arange(n_samples)
        .reshape(n_samples, 1, 1)
        .broadcast_to(n_samples, 5, 5),
        "b": torch.arange(n_samples)
        .reshape(n_samples, 1, 1)
        .broadcast_to(n_samples, 5, 5),
    }
    intermediate = repeat_interleave_batch_dim(data, repeats)
    for k in data:
        assert intermediate[k].shape == (n_samples * repeats, 5, 5)
    result = unfold_ensemble_dim(intermediate, repeats)
    for k in data:
        assert result[k].shape == (n_samples, repeats, 5, 5)
        assert torch.allclose(result[k], data[k][:, None, :, :])


def test_fold_sized_ensemble_dim_broadcasts_singleton():
    a = torch.randn(1, 1, 1)
    d = EnsembleTensorDict({"a": a})
    folded = fold_sized_ensemble_dim(d, 2)
    assert set(folded.keys()) == {"a"}
    assert folded["a"].shape == (2, 1)
    assert (folded["a"] == a).all()  # allclose doesn't broadcast


def test_fold_sized_ensemble_dim_reduces_dimensionality():
    a = torch.randn(2, 3, 4, 8)
    d = EnsembleTensorDict({"a": a})
    folded = fold_sized_ensemble_dim(d, 3)
    assert set(folded.keys()) == {"a"}
    assert folded["a"].shape == (6, 4, 8)
    torch.testing.assert_close(folded["a"], a.reshape(6, 4, 8))


def test_fold_sized_ensemble_dim_raises_on_invalid_ensemble_counts():
    a = torch.randn(2, 3, 4, 8)
    b = torch.randn(2, 4, 4, 8)
    d = EnsembleTensorDict({"a": a, "b": b})
    with pytest.raises(ValueError) as excinfo:
        fold_sized_ensemble_dim(d, 3)
    assert "some values in d have invalid ensemble member counts" in str(excinfo.value)
    assert "a: 3" in str(excinfo.value)
    assert "b: 4" in str(excinfo.value)


def test_unfold_ensemble_dim_increases_dimensionality():
    a = torch.randn(6, 4, 8)
    d = EnsembleTensorDict({"a": a})
    unfolded = unfold_ensemble_dim(d, 3)
    assert set(unfolded.keys()) == {"a"}
    assert unfolded["a"].shape == (2, 3, 4, 8)
    torch.testing.assert_close(unfolded["a"], a.reshape(2, 3, 4, 8))
