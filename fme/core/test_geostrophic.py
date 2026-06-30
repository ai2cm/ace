import numpy as np
import pytest
import torch
from torch_harmonics.quadrature import clenshaw_curtiss_weights

from fme.core.constants import EARTH_RADIUS, GRAVITY, OMEGA
from fme.core.geostrophic import (
    ageostrophic_wind_speed,
    coriolis_parameter,
    geostrophic_wind,
    horizontal_gradient,
)


def _grid_latitudes_deg(nlat: int) -> np.ndarray:
    """Latitudes (deg, N->S excluded) of the equiangular SHT quadrature grid.

    Mirrors the node construction in ``fme.sht_fix`` / torch_harmonics so test
    fields live exactly on the grid the transform integrates over.
    """
    cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
    cost = np.asarray(cost, dtype=np.float64)
    colatitude = np.flip(np.arccos(cost))  # 0 at North Pole, pi at South Pole
    latitude_rad = np.pi / 2 - colatitude
    return np.rad2deg(latitude_rad)


def _make_grid(nlat: int, nlon: int):
    lat_deg = _grid_latitudes_deg(nlat)
    lon_deg = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lat2d, lon2d = np.meshgrid(lat_deg, lon_deg, indexing="ij")
    return lat_deg, lon_deg, lat2d, lon2d


def test_horizontal_gradient_of_sin_latitude():
    # Phi = sin(lat) = z. On the unit sphere grad(Phi) = cos(lat) northward,
    # zero eastward; physical gradient divides by the Earth radius.
    nlat, nlon = 48, 96
    _, _, lat2d, _ = _make_grid(nlat, nlon)
    lat_rad = np.deg2rad(lat2d)
    phi = torch.tensor(np.sin(lat_rad), dtype=torch.float64)

    grad_east, grad_north = horizontal_gradient(phi)

    expected_north = torch.tensor(np.cos(lat_rad) / EARTH_RADIUS, dtype=torch.float64)
    # Exclude the polar rows where the equiangular quadrature is least accurate.
    interior = np.abs(lat2d) < 80.0
    mask = torch.tensor(interior)
    torch.testing.assert_close(
        grad_north[mask], expected_north[mask], rtol=1e-3, atol=1e-9
    )
    torch.testing.assert_close(
        grad_east[mask], torch.zeros_like(grad_east)[mask], rtol=0, atol=1e-9
    )


def test_horizontal_gradient_of_cos_lat_cos_lon():
    # Phi = cos(lat) cos(lon) = x. grad(x) on the unit sphere:
    #   east  component d/d(a cos(lat) lon) of x = -sin(lon)
    #   north component (1/a) d/d(lat) of x      = -sin(lat) cos(lon)
    nlat, nlon = 48, 96
    _, _, lat2d, lon2d = _make_grid(nlat, nlon)
    lat_rad = np.deg2rad(lat2d)
    lon_rad = np.deg2rad(lon2d)
    phi = torch.tensor(np.cos(lat_rad) * np.cos(lon_rad), dtype=torch.float64)

    grad_east, grad_north = horizontal_gradient(phi)

    expected_east = torch.tensor(-np.sin(lon_rad) / EARTH_RADIUS, dtype=torch.float64)
    expected_north = torch.tensor(
        -np.sin(lat_rad) * np.cos(lon_rad) / EARTH_RADIUS, dtype=torch.float64
    )
    interior = np.abs(lat2d) < 80.0
    mask = torch.tensor(interior)
    torch.testing.assert_close(
        grad_east[mask], expected_east[mask], rtol=2e-3, atol=1e-9
    )
    torch.testing.assert_close(
        grad_north[mask], expected_north[mask], rtol=2e-3, atol=1e-9
    )


def test_coriolis_parameter():
    lat = torch.tensor([-90.0, -30.0, 0.0, 30.0, 90.0], dtype=torch.float64)
    f = coriolis_parameter(lat)
    expected = 2.0 * OMEGA * torch.sin(torch.deg2rad(lat))
    torch.testing.assert_close(f, expected)
    assert f[2] == 0.0  # equator


def test_geostrophic_wind_zonal_jet():
    # Height depending only on latitude => purely zonal geostrophic wind
    # (v_g = 0), with u_g = -(g / (f a)) dh/dphi.
    nlat, nlon = 48, 96
    _, _, lat2d, _ = _make_grid(nlat, nlon)
    lat_rad = np.deg2rad(lat2d)
    # h = H0 + A sin(lat): dh/dphi = A cos(lat) (per radian).
    amplitude = 100.0
    height = torch.tensor(5000.0 + amplitude * np.sin(lat_rad), dtype=torch.float64)
    latitude = torch.tensor(lat2d, dtype=torch.float64)

    u_g, v_g = geostrophic_wind(height, latitude)

    f = 2.0 * OMEGA * np.sin(lat_rad)
    expected_u = -(GRAVITY / f) * (amplitude * np.cos(lat_rad) / EARTH_RADIUS)
    # Check away from the equator (f -> 0) and the poles, where the equiangular
    # quadrature has a small edge error (~1%) that is physically negligible (it
    # affects gen/target equally and carries tiny cos(lat) area weight).
    interior = (np.abs(lat2d) > 15.0) & (np.abs(lat2d) < 75.0)
    mask = torch.tensor(interior)
    torch.testing.assert_close(
        v_g[mask], torch.zeros_like(v_g)[mask], rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        u_g[mask],
        torch.tensor(expected_u, dtype=torch.float64)[mask],
        rtol=8e-3,
        atol=1e-3,
    )


def test_ageostrophic_wind_speed_masks_equator_and_matches_residual():
    nlat, nlon = 48, 96
    _, _, lat2d, _ = _make_grid(nlat, nlon)
    lat_rad = np.deg2rad(lat2d)
    height = torch.tensor(5000.0 + 100.0 * np.sin(lat_rad), dtype=torch.float64)
    latitude = torch.tensor(lat2d, dtype=torch.float64)
    u_g, v_g = geostrophic_wind(height, latitude)

    # Actual wind = geostrophic + a known ageostrophic perturbation.
    du = torch.full_like(u_g, 3.0)
    dv = torch.full_like(v_g, -4.0)
    speed = ageostrophic_wind_speed(u_g + du, v_g + dv, height, latitude)

    equator = np.abs(lat2d) < 10.0
    assert torch.isnan(speed[torch.tensor(equator)]).all()
    outside = (np.abs(lat2d) > 15.0) & (np.abs(lat2d) < 80.0)
    mask = torch.tensor(outside)
    torch.testing.assert_close(
        speed[mask], torch.full_like(speed, 5.0)[mask], rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_horizontal_gradient_batched_shapes(dtype):
    nlat, nlon = 48, 96
    field = torch.randn(2, 3, nlat, nlon, dtype=dtype)
    grad_east, grad_north = horizontal_gradient(field)
    assert grad_east.shape == field.shape
    assert grad_north.shape == field.shape
