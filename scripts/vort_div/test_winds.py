import numpy as np
import pytest
import torch
from torch_harmonics import InverseRealSHT, RealSHT
from winds import u_v_to_vort_div, vort_div_to_u_v

np.random.seed(0)


def test_constant_u_only_gives_vorticity():
    """
    A field with only constant u-wind should be divergence-free.
    """
    # Define a test case
    n_lat = 10
    n_lon = 10
    u = np.ones([n_lat, n_lon])
    v = np.zeros_like(u)

    # Convert to vorticity and divergence
    vort, div = u_v_to_vort_div(u, v)

    # Check that the divergence is zero
    np.testing.assert_almost_equal(div, 0.0)

    # Check that the vorticity is non-zero
    assert not np.any(vort == 0)
    # latitude starts at -90
    vort_south = vort[: n_lat // 2, :]
    vort_north = vort[n_lat // 2 :, :]
    # Curl in the northern hemisphere should be positive
    assert np.all(vort_north > 0)
    # Curl in the southern hemisphere should be negative
    assert np.all(vort_south < 0)


def test_constant_v_only_gives_divergence():
    """
    A field with only constant v-wind should be curl-free.
    """
    # Define a test case
    n_lat = 10
    n_lon = 10
    u = np.zeros([n_lat, n_lon])
    v = np.ones_like(u)

    # Convert to vorticity and divergence
    vort, div = u_v_to_vort_div(u, v)

    # Check that the vorticity is zero
    np.testing.assert_almost_equal(vort, 0.0)

    # Check that the divergence is non-zero
    assert not np.any(div == 0)
    # latitude starts at -90
    div_south = div[: n_lat // 2, :]
    div_north = div[n_lat // 2 :, :]
    # Divergence in the northern hemisphere should be negative
    assert np.all(div_north < 0)
    # Divergence in the southern hemisphere should be positive
    assert np.all(div_south > 0)


def test_vorticity_gaussian():
    """
    Gaussian blob of vorticity at the equator.
    """
    # Define a test case
    n_lat = 10
    n_lon = 20
    lat_approx = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    lon, lat = np.meshgrid(lon, lat_approx)
    # gaussian blob of scale 10 degrees at lat=0, lon=0
    vort = np.exp(-((lat / 10) ** 2 + (lon / 10) ** 2))
    div = np.zeros_like(vort)
    u, v = vort_div_to_u_v(vort, div)
    # # Can use this code to inspect manually
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # im = ax[0].imshow(u)
    # fig.colorbar(im, ax=ax[0])
    # im = ax[1].imshow(v)
    # fig.colorbar(im, ax=ax[1])
    # ax[0].set_title("u")
    # ax[1].set_title("v")
    # plt.tight_layout()
    # plt.savefig("u_v_vort_blob.png")
    u_south = u[: n_lat // 2, n_lon // 4 : -n_lon // 4]
    u_north = u[n_lat // 2 :, n_lon // 4 : -n_lon // 4]
    v_west = v[:, : n_lon // 2]
    v_east = v[:, n_lon // 2 :]
    # u_south should be positive
    assert np.all(u_south > 0)
    # u_north should be negative
    assert np.all(u_north < 0)
    # v_west should be negative
    assert np.all(v_west < 0)
    # v_east should be positive
    assert np.all(v_east > 0)


def test_divergence_gaussian():
    """
    Gaussian blob of divergence at the equator.
    """
    # Define a test case
    n_lat = 10
    n_lon = 20
    lat_approx = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    lon, lat = np.meshgrid(lon, lat_approx)
    # gaussian blob of scale 10 degrees at lat=0, lon=0
    div = np.exp(-((lat / 10) ** 2 + (lon / 10) ** 2))
    vort = np.zeros_like(div)
    u, v = vort_div_to_u_v(vort, div)
    # # Can use this code to inspect manually
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # im = ax[0].imshow(u)
    # fig.colorbar(im, ax=ax[0])
    # im = ax[1].imshow(v)
    # fig.colorbar(im, ax=ax[1])
    # ax[0].set_title("u")
    # ax[1].set_title("v")
    # plt.tight_layout()
    # plt.savefig("u_v_div_blob.png")
    u_west = u[:, : n_lon // 2]
    u_east = u[:, n_lon // 2 :]
    v_south = v[: n_lat // 2, n_lon // 4 : -n_lon // 4]
    v_north = v[n_lat // 2 :, n_lon // 4 : -n_lon // 4]
    # u_west should be negative
    assert np.all(u_west < 0)
    # u_east should be positive
    assert np.all(u_east > 0)
    # v_south should be negative
    assert np.all(v_south < 0)
    # v_north should be positive
    assert np.all(v_north > 0)


@pytest.mark.parametrize(
    "vort, div",
    [
        pytest.param(
            np.random.randn(11, 20), np.random.randn(11, 20), id="random_vort_div"
        ),
    ],
)
def test_vort_div_roundtrip_from_vort_div(vort, div):
    sht = RealSHT(
        nlat=vort.shape[-2],
        nlon=vort.shape[-1],
        grid="legendre-gauss",
    ).float()
    inverse_sht = InverseRealSHT(
        nlat=vort.shape[-2],
        nlon=vort.shape[-1],
        grid="legendre-gauss",
    ).float()
    # First, round-trip the winds through SHT/ISHT to filter values that
    # aren't represented spectrally, we don't expect those to be kept.
    vort = (
        inverse_sht(sht(torch.as_tensor(vort, dtype=torch.float)))
        .cpu()
        .double()
        .numpy()
    )
    div = (
        inverse_sht(sht(torch.as_tensor(div, dtype=torch.float))).cpu().double().numpy()
    )

    u, v = vort_div_to_u_v(vort, div)
    # Then, do the round-trip we'll actually test
    vort, div = u_v_to_vort_div(u, v)
    u2, v2 = vort_div_to_u_v(vort, div)
    np.testing.assert_allclose(u, u2, atol=0.15)
    np.testing.assert_allclose(v, v2, atol=0.15)


@pytest.mark.parametrize(
    "u, v",
    [
        pytest.param(np.zeros([11, 20]), np.zeros([11, 20]), id="zero_wind"),
        pytest.param(np.ones([11, 20]), np.zeros([11, 20]), id="east_wind"),
        pytest.param(np.zeros([11, 20]), np.ones([11, 20]), id="north_wind"),
        pytest.param(
            np.random.randn(11, 20), np.random.randn(11, 20), id="random_wind"
        ),
    ],
)
def test_vort_div_roundtrip(u, v):
    # first, round-trip the winds through SHT/ISHT to filter values that
    # aren't represented spectrally, we don't expect those to be kept.
    vort, div = u_v_to_vort_div(u, v)
    u, v = vort_div_to_u_v(vort, div)
    # Then, do the round-trip we'll actually test
    vort, div = u_v_to_vort_div(u, v)
    u2, v2 = vort_div_to_u_v(vort, div)
    np.testing.assert_allclose(u, u2, atol=0.15)
    np.testing.assert_allclose(v, v2, atol=0.15)
