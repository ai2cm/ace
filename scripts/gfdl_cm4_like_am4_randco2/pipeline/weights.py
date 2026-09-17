"""Loader for the precomputed conservative regridding weight artifact.

Weights for a (native source grid) x (Gaussian target grid) pair are computed
once and published to GCS under a versioned prefix; pipeline workers load
them into a per-process cached regridder instead of recomputing weights at
startup (the era5 pipeline's regridder-cache pattern, extended with
precomputed weights since the tripolar source grid is far more expensive to
mesh than a regular lat-lon grid).

An artifact is a GCS prefix containing:

- ``source_grid.nc`` — tracer-cell centers, corners, and rotation angle
  extracted from the MOM6 supergrid; enough to reconstruct the regridder
  without reading the much larger ocean_hgrid.nc.
- ``weights.nc`` — xESMF-format sparse conservative weights.

The sources this pipeline reads sit on the 0.25-degree tripolar tracer grid
the published OM4 artifacts were built from, so no weight generation happens
here: the artifact prefix is named in the config and only read.

Precomputed weights are raw conservative weights: xESMF's ``skipna``
renormalization only exists at weight-computation time, so masking must be
applied explicitly at regrid time — see
:func:`pipeline.ocean_emulators_port.regrid_normalized`.
"""

import logging
import os
import tempfile

import fsspec
import xarray as xr
import xesmf as xe

from .grids import make_target_grid

logger = logging.getLogger(__name__)

REGRIDDER_KWARGS = dict(
    method="conservative",
    ignore_degenerate=True,
    periodic=True,
    unmapped_to_nan=True,
)
SOURCE_GRID_FILENAME = "source_grid.nc"
WEIGHTS_FILENAME = "weights.nc"


def _download(url: str, local_path: str) -> None:
    with fsspec.open(url, "rb") as src, open(local_path, "wb") as dst:
        dst.write(src.read())


# One regridder per (artifact, target grid) per worker process.
_REGRIDDER_CACHE: dict[tuple[str, str], xe.Regridder] = {}


def open_source_grid(weights_url: str) -> xr.Dataset:
    """Read the source-grid geometry (centers, corners, rotation angle)
    stored with the weight artifact at the ``weights_url`` prefix."""
    with fsspec.open(f"{weights_url.rstrip('/')}/{SOURCE_GRID_FILENAME}") as f:
        return xr.open_dataset(f).load().set_coords(["lon", "lat", "lon_b", "lat_b"])


def get_regridder(weights_url: str, target_grid_name: str) -> xe.Regridder:
    """Load the weight artifact at the ``weights_url`` prefix into a cached
    conservative regridder for the named Gaussian target grid."""
    key = (weights_url, target_grid_name)
    if key not in _REGRIDDER_CACHE:
        logger.info(f"Loading regridding weights from {weights_url}")
        source_grid = open_source_grid(weights_url)
        target_grid = make_target_grid(target_grid_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, WEIGHTS_FILENAME)
            _download(f"{weights_url.rstrip('/')}/{WEIGHTS_FILENAME}", weights_path)
            _REGRIDDER_CACHE[key] = xe.Regridder(
                source_grid,
                target_grid,
                weights=weights_path,
                reuse_weights=True,
                **REGRIDDER_KWARGS,
            )
    return _REGRIDDER_CACHE[key]
