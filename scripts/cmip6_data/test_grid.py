"""Minimal tests for grid.py. Runnable standalone: ``python test_grid.py``
or via pytest from this directory.
"""

import numpy as np
from grid import GAUSSIAN_GRID_N, make_target_grid


def test_f22_5_shape():
    ds = make_target_grid("F22.5")
    assert ds["lat"].size == 45
    assert ds["lon"].size == 90
    assert ds["lat_b"].size == 46
    assert ds["lon_b"].size == 91


def test_f22_5_symmetric_about_equator():
    ds = make_target_grid("F22.5")
    lat = ds["lat"].values
    # Gaussian latitudes are symmetric in sin(lat); sorted south-to-north
    # means lat[i] == -lat[-(i+1)] to numerical precision.
    np.testing.assert_allclose(lat, -lat[::-1], atol=1e-12)


def test_f22_5_bounds_clipped_to_poles():
    ds = make_target_grid("F22.5")
    lat_b = ds["lat_b"].values
    assert lat_b[0] == -90.0
    assert lat_b[-1] == 90.0


def test_f22_5_lon_offset_half_step():
    ds = make_target_grid("F22.5")
    lon = ds["lon"].values
    # 90 cells -> dlon = 4 deg; first center at 2 deg, last at 358.
    assert lon[0] == 2.0
    assert lon[-1] == 358.0


def test_known_grid_names():
    assert set(GAUSSIAN_GRID_N) >= {"F22.5", "F90", "F360"}


def test_unknown_grid_name_raises():
    try:
        make_target_grid("nope")
    except ValueError as e:
        assert "Unknown Gaussian grid name" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all tests passed")
