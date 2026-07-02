"""Gauss-Legendre target grids for CMIP6 daily pilot regridding.

Copied (lightly adapted) from ``scripts/era5/pipeline/xr-beam-pipeline.py``
so this directory is self-contained; consolidate into a shared utility
module when the pilot graduates.

Naming follows the project convention ``F<N>`` where ``nlat = 2N`` and
``nlon = 4N``. For example, ``F22.5`` is a ~4° grid (45 x 90).
"""

import numpy as np
import xarray as xr

# Gaussian grid specs: name -> N (grid number; nlat=2N, nlon=4N)
GAUSSIAN_GRID_N: dict[str, float] = {
    "F22.5": 22.5,  # ~4 deg (45 x 90)
    "F90": 90,  # 1 deg (180 x 360)
    "F360": 360,  # 0.25 deg (720 x 1440)
}


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Compute cell boundaries from centers, clipping to [lo, hi]."""
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[lo], midpoints, [hi]])


def _gaussian_latitudes(n: float) -> np.ndarray:
    """Gaussian grid latitudes for grid number N (2N latitudes, degrees,
    sorted south-to-north). Roots of the Legendre polynomial P_{2N}.
    """
    from numpy.polynomial.legendre import leggauss

    nlat = round(2 * n)
    x, _ = leggauss(nlat)
    return np.sort(np.degrees(np.arcsin(x)))


def make_target_grid(name: str) -> xr.Dataset:
    """Create an xESMF-ready Gaussian target grid dataset.

    Returns an ``xr.Dataset`` with ``lat``, ``lon``, ``lat_b``, and
    ``lon_b`` — suitable as the target for ``xesmf.Regridder(..., method)``
    with either bilinear (ignores bounds) or conservative (uses bounds).

    Longitudes are offset by half a grid spacing so the first cell is
    centered at ``dlon / 2`` and the last at ``360 - dlon / 2``.
    """
    if name not in GAUSSIAN_GRID_N:
        raise ValueError(
            f"Unknown Gaussian grid name: {name!r}. "
            f"Known: {sorted(GAUSSIAN_GRID_N)}"
        )
    n = GAUSSIAN_GRID_N[name]
    lat = _gaussian_latitudes(n)
    nlon = round(4 * n)
    dlon = 360.0 / nlon
    lon = np.linspace(dlon / 2, 360 - dlon / 2, nlon)
    lat_b = _cell_bounds(lat, -90, 90)
    lon_b = _cell_bounds(lon, 0, 360)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], lat_b),
            "lon_b": (["lon_b"], lon_b),
        }
    )


__all__ = ["GAUSSIAN_GRID_N", "make_target_grid"]
