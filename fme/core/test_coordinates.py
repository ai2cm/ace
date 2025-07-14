import numpy as np
import pytest
import torch

from fme.ace.models.healpix.healpix_layers import HEALPixPadding
from fme.core.coordinates import (
    DepthCoordinate,
    HEALPixCoordinates,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
    e2ghpx,
)
from fme.core.mask_provider import MaskProvider


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
        (torch.tensor([1, 2, 3]), torch.tensor([1, 1]), torch.tensor(3.0)),
        (torch.tensor([1, 2, 3]), torch.tensor([1, 0]), torch.tensor(1.0)),
        (torch.tensor([1, 2, 3]), torch.tensor([0, 0]), torch.tensor(float("nan"))),
        (torch.tensor([1, 2, 4]), torch.tensor([1, 1]), torch.tensor(5.0)),
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
    torch.testing.assert_close(result, expected, equal_nan=True)


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
    expected = (data[:, :, 0] * depth_0 + data[:, :, 1] * depth_1).float()
    result = DepthCoordinate(idepth, mask).depth_integral(data)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "name, level",
    [
        ("sfc_level", 0),
        ("depth_0", 0),
        ("depth_3", 3),
    ],
)
def test_depth_get_mask_tensor_for(name, level):
    idepth = torch.arange(end=5)
    mask = torch.arange(end=4)
    coord = DepthCoordinate(idepth, mask)
    assert coord.get_mask_tensor_for(name) == level


def test_depth_returns_surface_mask_if_specified():
    idepth = torch.arange(end=5)
    mask = torch.arange(end=4)
    surface_mask = torch.tensor([4])
    coord_sfc_mask = DepthCoordinate(idepth, mask, surface_mask)
    coord_no_sfc_mask = DepthCoordinate(idepth, mask)
    assert coord_sfc_mask.get_mask_tensor_for("sfc_level") == surface_mask[0]
    assert coord_no_sfc_mask.get_mask_tensor_for("sfc_level") == mask[0]


def test_masked_lat_lon_ops_from_coords():
    lat = torch.tensor([0.0, 0.0, 0.0])
    lon = torch.tensor([0.0])
    mask = torch.tensor([[1], [0], [1]])
    coords = LatLonCoordinates(lat=lat, lon=lon)
    mask_provider = MaskProvider(masks={"mask_0": mask})
    gridded_ops = coords.get_gridded_operations(mask_provider=mask_provider)
    input_ = torch.tensor([[1.0], [-10.0], [3.0]])
    result = gridded_ops.area_weighted_mean(input_, name="T_0")
    torch.testing.assert_close(result, torch.tensor(2.0))


def test_healpix_ops_raises_value_error_with_mask():
    face = torch.arange(12)
    height = torch.arange(16)
    width = torch.arange(16)
    healpix_coords = HEALPixCoordinates(face=face, height=height, width=width)
    mask_provider = MaskProvider(masks={"mask_0": torch.tensor([1, 0, 1])})

    expected_msg = "HEALPixCoordinates does not support a mask"
    with pytest.raises(NotImplementedError, match=expected_msg):
        healpix_coords.get_gridded_operations(mask_provider=mask_provider)


@pytest.mark.skipif(e2ghpx is None, reason="earth2grid healpix not available")
@pytest.mark.parametrize("pad", [True, False])
def test_healpix_coordinates_xyz(pad: bool, very_fast_only: bool):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests and healpix (earth2grid) tests")

    face = torch.arange(12)
    height = torch.arange(16)
    width = torch.arange(16)
    healpix_coords = HEALPixCoordinates(face=face, height=height, width=width)

    x, y, z = healpix_coords.xyz

    # Calculate original distances between points along x and y axes
    distances_x = np.sqrt(
        (np.diff(x, axis=-1) ** 2)
        + (np.diff(y, axis=-1) ** 2)
        + (np.diff(z, axis=-1) ** 2)
    )
    distances_y = np.sqrt(
        (np.diff(x, axis=-2) ** 2)
        + (np.diff(y, axis=-2) ** 2)
        + (np.diff(z, axis=-2) ** 2)
    )

    # Apply HEALPix padding
    if pad:
        padding = 2
        healpix_padding = HEALPixPadding(padding=padding, enable_nhwc=False)
        padded_x = healpix_padding(torch.Tensor(x).unsqueeze(1)).squeeze(1)
        padded_y = healpix_padding(torch.Tensor(y).unsqueeze(1)).squeeze(1)
        padded_z = healpix_padding(torch.Tensor(z).unsqueeze(1)).squeeze(1)

        # Calculate distances between padding points along x and y axes
        distances_padded_x = np.sqrt(
            (np.diff(padded_x.numpy(), axis=-1) ** 2)
            + (np.diff(padded_y.numpy(), axis=-1) ** 2)
            + (np.diff(padded_z.numpy(), axis=-1) ** 2)
        )
        distances_padded_y = np.sqrt(
            (np.diff(padded_x.numpy(), axis=-2) ** 2)
            + (np.diff(padded_y.numpy(), axis=-2) ** 2)
            + (np.diff(padded_z.numpy(), axis=-2) ** 2)
        )

        max_distance_x = np.max(distances_x)
        max_distance_y = np.max(distances_y)

        # Assert distances are not too far from the expected diff (min = 0, max = 2dx)
        assert np.all(
            distances_padded_x <= 2 * max_distance_x
        ), "Some distances along x-axis exceed 2 * max_distance_x"
        assert np.all(
            distances_padded_y <= 2 * max_distance_y
        ), "Some distances along y-axis exceed 2 * max_distance_y"

    else:
        mean_distances_x = distances_x.mean()
        mean_distances_y = distances_y.mean()

        assert np.allclose(distances_x, mean_distances_x, atol=0.03)
        assert np.allclose(distances_y, mean_distances_y, atol=0.03)
