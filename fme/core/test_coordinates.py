import pytest
import torch

from fme.core.coordinates import (
    DepthCoordinate,
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
        (
            DepthCoordinate(idepth=torch.tensor([1, 2, 3]), mask=torch.tensor([4, 5])),
            DepthCoordinate(idepth=torch.tensor([1, 2, 3]), mask=torch.tensor([4, 5])),
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
        (
            DepthCoordinate(idepth=torch.tensor([1, 2, 3]), mask=torch.tensor([4, 5])),
            DepthCoordinate(idepth=torch.tensor([2, 2, 3]), mask=torch.tensor([4, 5])),
        ),
        (
            DepthCoordinate(idepth=torch.tensor([1, 2, 3]), mask=torch.tensor([4, 5])),
            DepthCoordinate(idepth=torch.tensor([1, 2, 3]), mask=torch.tensor([5, 5])),
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
    with pytest.raises(ValueError, match="ak and bk must have the same length"):
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


@pytest.mark.parametrize(
    "idepth, mask, msg",
    [
        (torch.tensor([[1, 2, 3]]), torch.tensor([4, 5]), "1-dimensional tensor"),
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), "last dimension of mask"),
        (torch.tensor([1]), torch.tensor([]), "idepth must have at least two"),
    ],
    ids=["idepth 2D", "mask inconsistent with idepth", "idepth too short"],
)
def test_depth_coordinate_validation(idepth, mask, msg):
    with pytest.raises(ValueError, match=msg):
        DepthCoordinate(idepth, mask)


def test_depth_coordinate_integral_raises():
    idepth = torch.tensor([1, 2, 3])
    mask = torch.tensor([1, 1])
    data = torch.tensor([1, 1, 4])
    coords = DepthCoordinate(idepth, mask)
    with pytest.raises(ValueError, match="dimension of integrand must match"):
        coords.depth_integral(data)


@pytest.mark.parametrize(
    "idepth, mask, expected",
    [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 1]), torch.tensor(3)),
        (torch.tensor([1, 2, 3]), torch.tensor([1, 0]), torch.tensor(1)),
        (torch.tensor([1, 2, 3]), torch.tensor([0, 0]), torch.tensor(0)),
        (torch.tensor([1, 2, 4]), torch.tensor([1, 1]), torch.tensor(5)),
    ],
    ids=[
        "mask is all ones",
        "second layer is masked out",
        "all layers masked out",
        "varying depth",
    ],
)
def test_depth_integral_1d_data(idepth, mask, expected):
    data = torch.arange(1, len(idepth))
    result = DepthCoordinate(idepth, mask).depth_integral(data)
    torch.testing.assert_close(result, expected)


def test_depth_integral_3d_data():
    img_shape = 2, 4
    idepth = torch.tensor([1, 2, 4])
    mask = torch.tensor([1, 1])
    nz = len(mask)
    data = torch.arange(1, img_shape[0] * img_shape[1] * nz + 1).reshape(
        img_shape + (nz,)
    )
    depth_0 = idepth[1] - idepth[0]
    depth_1 = idepth[2] - idepth[1]
    expected = data[:, :, 0] * depth_0 + data[:, :, 1] * depth_1
    result = DepthCoordinate(idepth, mask).depth_integral(data)
    torch.testing.assert_close(result, expected)
