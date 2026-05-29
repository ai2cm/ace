import torch

from .utils import captured_before, force_positive


def test_captured_before_returns_only_modified_variables():
    original = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([3.0, 4.0]),
    }
    # "a" is replaced by a new tensor; "b" keeps its original object.
    corrected = {"a": torch.tensor([5.0, 6.0]), "b": original["b"]}
    before = captured_before(original, corrected)
    assert set(before) == {"a"}
    assert before["a"] is original["a"]


def test_captured_before_empty_when_nothing_changed():
    original = {"a": torch.tensor([1.0])}
    corrected = dict(original)  # shallow copy: same tensor objects
    assert captured_before(original, corrected) == {}


def test_captured_before_ignores_variables_absent_from_original():
    original = {"a": torch.tensor([1.0])}
    corrected = {"a": original["a"], "new": torch.tensor([2.0])}
    # "new" did not exist before correction, so it has no pre-correction value.
    assert captured_before(original, corrected) == {}


def test_captured_before_matches_force_positive():
    original = {"a": torch.tensor([-1.0, 2.0]), "b": torch.tensor([3.0])}
    corrected = force_positive(original, ["a"])
    before = captured_before(original, corrected)
    assert set(before) == {"a"}
    torch.testing.assert_close(before["a"], original["a"])
    torch.testing.assert_close(corrected["a"], torch.tensor([0.0, 2.0]))
