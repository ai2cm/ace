import numpy as np
import pytest

from fme.core.winds import lon_lat_to_xyz, u_v_to_x_y_z_wind


def test_u_v_to_x_y_z_wind_energy_conservation():
    """
    Test that the energy is conserved when transforming winds.
    """
    # Define a test case
    u = np.random.randn(10, 10)
    v = np.random.randn(10, 10)
    lat = np.random.uniform(-90, 90, size=(10, 10))
    lon = np.random.uniform(-180, 180, size=(10, 10))

    # Convert to x, y, z
    x, y, z = u_v_to_x_y_z_wind(u, v, lat, lon)

    # Compute the energy conservation equation
    final_energy = x**2 + y**2 + z**2
    initial_energy = u**2 + v**2

    # Check that the energy is conserved
    assert np.allclose(final_energy, initial_energy)


def test_u_v_to_x_y_z_wind_is_horizontal():
    """
    Test that the transformed winds are perpendicular to the vertical vector.
    """
    # Define a test case
    u = np.random.randn(10, 10)
    v = np.random.randn(10, 10)
    lat = np.random.uniform(-90, 90, size=(10, 10))
    lon = np.random.uniform(-180, 180, size=(10, 10))

    # Convert to x, y, z
    wx, wy, wz = u_v_to_x_y_z_wind(u, v, lat, lon)
    x, y, z = lon_lat_to_xyz(lon, lat)

    # Compute the dot product
    dot_product = wx * x + wy * y + wz * z

    # Check that the dot product is zero
    assert np.allclose(dot_product, 0)


@pytest.mark.parametrize(
    "u, v, lat, lon, expected_x, expected_y, expected_z",
    [
        pytest.param(0, 1, 0, 0, 0, 0, 1, id="north_wind_at_equator"),
        pytest.param(1, 0, 0, 0, 0, 1, 0, id="east_wind_at_equator_prime_meridian"),
        pytest.param(
            1, 0, 0, 90, -1, 0, 0, id="east_wind_at_equator_90_degrees_east_meridian"
        ),
        pytest.param(0, -1, 0, 0, 0, 0, -1, id="south_wind_at_equator"),
    ],
)
def test_u_v_to_x_y_z_wind_expected_values(
    u, v, lat, lon, expected_x, expected_y, expected_z
):
    """
    Test that the expected values are returned.
    """
    # Convert to x, y, z
    x, y, z = u_v_to_x_y_z_wind(u, v, lat, lon)

    # Check that the expected values are returned
    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)
    assert np.allclose(z, expected_z)
