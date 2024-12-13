import pytest
import torch

from fme.core.coordinates import (
    HEALPixCoordinates,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)


@pytest.mark.parametrize(
    "first, second",
    [
        (
            HybridSigmaPressureCoordinate(
                ak=torch.tensor([1, 2, 3]), bk=torch.tensor([4, 5, 6])
            ),
            HybridSigmaPressureCoordinate(
                ak=torch.tensor([1, 2, 3]), bk=torch.tensor([4, 5, 6])
            ),
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
            HybridSigmaPressureCoordinate(
                ak=torch.tensor([1, 2, 3]), bk=torch.tensor([4, 5, 6])
            ),
            HybridSigmaPressureCoordinate(
                ak=torch.tensor([1, 2, 3]), bk=torch.tensor([5, 6, 7])
            ),
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


def test_vertical_integral_shape():
    nlat, nlon, nz = 4, 8, 3
    water = torch.rand(nlat, nlon, nz)
    pressure = torch.rand(nlat, nlon)
    ak, bk = torch.arange(nz + 1), torch.arange(nz + 1)
    coords = HybridSigmaPressureCoordinate(ak, bk)
    water_path = coords.vertical_integral(water, pressure)
    assert water_path.shape == (nlat, nlon)


def test_vertical_coordinates_raises_value_error():
    ak, bk = torch.arange(3), torch.arange(4)
    with pytest.raises(ValueError):
        HybridSigmaPressureCoordinate(ak, bk)


def test_vertical_coordinates_len():
    ak, bk = torch.arange(3), torch.arange(3)
    coords = HybridSigmaPressureCoordinate(ak, bk)
    assert len(coords) == 3


def test_interface_pressure():
    ak = torch.tensor([2.0, 0.5, 0.0])
    bk = torch.tensor([0.0, 0.5, 1.0])
    psfc = torch.tensor([[1, 1], [2, 2]])
    coords = HybridSigmaPressureCoordinate(ak, bk)
    pinterface = coords.interface_pressure(psfc)
    assert pinterface.shape == (2, 2, 3)
    assert pinterface[0, 0, 0] == ak[0]
    assert pinterface[0, 0, -1] == bk[-1] * psfc[0, 0]
