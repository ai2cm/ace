import pytest

from fme.core.typing_ import Slice


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
