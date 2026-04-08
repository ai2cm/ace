import pytest
import torch

from fme.core.typing_ import Slice, _shift_bound
from fme.core.validation import raise_if_any_variable_all_nan


@pytest.mark.parametrize(
    "start, stop, step, value, expected",
    [
        (0, 10, 1, 0, True),
        (0, 10, 1, 10, False),
        (0, 10, 2, 0, True),
        (0, 10, 2, 1, False),
        (None, None, None, 0, True),
        (None, None, None, 500, True),
        (0, 10, None, 4, True),
        (0, 10, None, 10, False),
        (0, None, 2, 0, True),
        (None, None, 2, 1, False),
    ],
)
def test_slice_contains(start, stop, step, value, expected):
    slice = Slice(start=start, stop=stop, step=step)
    assert slice.contains(value) == expected


def test__shift_bound():
    assert _shift_bound(5, 3, 0) == 2  # in bounds after shift
    assert _shift_bound(5, 10, 0) == 0  # out of bounds after shift
    assert _shift_bound(None, 3, 0) is None
    with pytest.raises(ValueError):
        _shift_bound(-1, 3, 0)  # invalid start bound


@pytest.mark.parametrize(
    "batch_values, batch_start_idx, expected_values",
    [
        ([0, 1, 2, 3, 4], 0, []),
        ([5, 6, 7, 8, 9], 5, [5, 6, 7, 8, 9]),
        (
            [10, 11, 12, 13, 14, 15, 16],
            10,
            [10, 11, 12, 13, 14],
        ),
        ([17, 18, 19, 20], 17, []),
    ],
)
def test_slice_shift_left(batch_values, batch_start_idx, expected_values):
    sl = Slice(5, 15, None)
    shifted = Slice.shift_left(sl, batch_start_idx)
    assert batch_values[shifted.slice] == expected_values


def test_raise_if_any_variable_all_nan_raises():
    data = {"good": torch.ones(2, 3), "bad": torch.full((2, 3), float("nan"))}
    with pytest.raises(ValueError, match="bad.*entirely NaN"):
        raise_if_any_variable_all_nan(data)


def test_raise_if_any_variable_all_nan_with_context():
    data = {"x": torch.full((1,), float("nan"))}
    with pytest.raises(ValueError, match="ocean prediction.*'x'.*entirely NaN"):
        raise_if_any_variable_all_nan(data, context="ocean prediction")


def test_raise_if_any_variable_all_nan_passes_for_valid_data():
    data = {"a": torch.ones(2, 3), "b": torch.zeros(4)}
    raise_if_any_variable_all_nan(data)


def test_raise_if_any_variable_all_nan_passes_for_partial_nan():
    tensor = torch.ones(2, 3)
    tensor[0, 0] = float("nan")
    data = {"partial": tensor}
    raise_if_any_variable_all_nan(data)


def test_raise_if_any_variable_all_nan_passes_for_empty_mapping():
    raise_if_any_variable_all_nan({})
