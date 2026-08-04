import numpy as np

DENSITY_OF_AIR=1.0 # kg m^-3
# Powell and Reinhold integrate the surface wind field over a 1 m depth, which is
# what puts IKE in the conventional tens-to-hundreds-of-TJ range. Holding the
# depth fixed rather than tying it to the horizontal spacing also keeps IKE a
# property of the storm rather than of the grid, so runs at different resolutions
# stay comparable.
SURFACE_LAYER_DEPTH_M = 1.0


def integrated_kinetic_energy(
        wind_speed: np.ndarray,
        dx_km: float = 3.0,
) -> float:
    # c.f. Powell and Reinhold 2007
    # Only integrates over the lowest grid level, i.e. this is just the sum over the 2d domain
    # Assumes equal-area cells; on a lat/lon grid spanning a wide range of
    # latitudes, weight each cell by R^2 cos(lat) dlat dlon instead.
    dx_meters = dx_km * 1000.0
    cell_mass = DENSITY_OF_AIR * dx_meters**2 * SURFACE_LAYER_DEPTH_M
    return 0.5 * np.sum(cell_mass * wind_speed**2)

