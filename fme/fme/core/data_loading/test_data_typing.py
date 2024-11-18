import pytest
import torch

from fme.core.data_loading.data_typing import (
    HEALPixCoordinates,
    LatLonCoordinates,
    SigmaCoordinates,
)


@pytest.mark.parametrize(
    "first, second",
    [
        (
            SigmaCoordinates(ak=torch.tensor([1, 2, 3]), bk=torch.tensor([4, 5, 6])),
            SigmaCoordinates(ak=torch.tensor([1, 2, 3]), bk=torch.tensor([4, 5, 6])),
        ),
        (
            LatLonCoordinates(lat=torch.tensor([1, 2, 3]), lon=torch.tensor([4, 5, 6])),
            LatLonCoordinates(lat=torch.tensor([1, 2, 3]), lon=torch.tensor([4, 5, 6])),
        ),
        (
            HEALPixCoordinates(
                face=torch.tensor([1, 2, 3]),
                height=torch.tensor([4, 5, 6]),
                width=torch.tensor([7, 8, 9]),
            ),
            HEALPixCoordinates(
                face=torch.tensor([1, 2, 3]),
                height=torch.tensor([4, 5, 6]),
                width=torch.tensor([7, 8, 9]),
            ),
        ),
    ],
)
def test_equality(first, second):
    assert first == second


@pytest.mark.parametrize(
    "first, second",
    [
        (
            SigmaCoordinates(ak=torch.tensor([1, 2, 3]), bk=torch.tensor([4, 5, 6])),
            SigmaCoordinates(ak=torch.tensor([1, 2, 3]), bk=torch.tensor([5, 6, 7])),
        ),
        (
            LatLonCoordinates(lat=torch.tensor([1, 2, 3]), lon=torch.tensor([4, 5, 6])),
            LatLonCoordinates(lat=torch.tensor([1, 2, 3]), lon=torch.tensor([5, 6, 7])),
        ),
        (
            HEALPixCoordinates(
                face=torch.tensor([1, 2, 3]),
                height=torch.tensor([4, 5, 6]),
                width=torch.tensor([7, 8, 9]),
            ),
            HEALPixCoordinates(
                face=torch.tensor([1, 2, 3]),
                height=torch.tensor([4, 5, 6]),
                width=torch.tensor([8, 9, 10]),
            ),
        ),
        (
            LatLonCoordinates(lat=torch.tensor([1, 2, 3]), lon=torch.tensor([4, 5, 6])),
            HEALPixCoordinates(
                face=torch.tensor([1, 2, 3]),
                height=torch.tensor([4, 5, 6]),
                width=torch.tensor([7, 8, 9]),
            ),
        ),
    ],
)
def test_inequality(first, second):
    assert first != second
