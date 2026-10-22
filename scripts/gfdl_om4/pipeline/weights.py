"""Precomputed conservative regridding weights as a versioned GCS artifact.

Weights for a (native source grid) x (Gaussian target grid) pair are computed
once by the setup entry point in this module and written to GCS; pipeline
workers load them into a per-process cached regridder instead of recomputing
weights at startup (the era5 pipeline's regridder-cache pattern, extended
with precomputed weights since the tripolar source grid is far more expensive
to mesh than a regular lat-lon grid).

An artifact is a GCS prefix containing:

- ``source_grid.nc`` — tracer-cell centers, corners, and rotation angle
  extracted from the MOM6 supergrid; enough to reconstruct the regridder
  (and rotate vectors) without reading the much larger ocean_hgrid.nc.
- ``weights.nc`` — xESMF-format sparse conservative weights.
- ``areacello.nc`` — target cell areas ``A_J`` (native-area artifacts only;
  see :mod:`pipeline.native_area_weights`).

:func:`get_regridder` loads only native-area artifacts: it refuses a
``weights.nc`` lacking any attr of :data:`pipeline.native_area.ARTIFACT_ATTRS`.

Precomputed weights are raw conservative weights: xESMF's ``skipna``
renormalization only exists at weight-computation time, so masking must be
applied explicitly at regrid time — see
:func:`pipeline.ocean_emulators_port.regrid_normalized`.

Example setup invocation (see also the Makefile):

    python -m pipeline.weights \\
        --hgrid-url gs://vcm-ml-scratch/jamesd/2024-11-11-static-data/ocean_hgrid.nc \\
        --target-grid F90 \\
        --output-url gs://vcm-ml-scratch/jamesd/gfdl-om4-regridding-weights/v1/om4-tripolar-0.25deg-to-F90
"""

import argparse
import logging
import os
import tempfile

import fsspec
import numpy as np
import xarray as xr
import xesmf as xe

from .grids import make_target_grid
from .native_area import ARTIFACT_ATTRS, Triplets
from .ocean_emulators_port import convert_supergrid

logger = logging.getLogger(__name__)

REGRIDDER_KWARGS = dict(
    method="conservative",
    ignore_degenerate=True,
    periodic=True,
    unmapped_to_nan=True,
)
SOURCE_GRID_FILENAME = "source_grid.nc"
WEIGHTS_FILENAME = "weights.nc"
AREACELLO_FILENAME = "areacello.nc"


def _upload(local_path: str, url: str) -> None:
    with open(local_path, "rb") as src, fsspec.open(url, "wb") as dst:
        dst.write(src.read())


def _download(url: str, local_path: str) -> None:
    with fsspec.open(url, "rb") as src, open(local_path, "wb") as dst:
        dst.write(src.read())


def generate_weights(
    hgrid_url: str, target_grid_name: str, output_url: str, overwrite: bool = False
) -> None:
    """Compute conservative xESMF weights for the source grid described by
    the supergrid at ``hgrid_url`` onto the named Gaussian target grid, and
    write the weight artifact under the ``output_url`` prefix.

    Refuses to clobber an existing artifact unless ``overwrite`` is set:
    published artifacts are treated as immutable, so a changed weight
    computation belongs under a new version prefix."""
    fs, _ = fsspec.url_to_fs(output_url)
    existing = [
        url
        for filename in (SOURCE_GRID_FILENAME, WEIGHTS_FILENAME)
        if fs.exists(url := f"{output_url.rstrip('/')}/{filename}")
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Weight artifact already exists at {output_url} ({existing}). "
            "Published artifacts are immutable: bump the version in the "
            "output URL (e.g. WEIGHTS_VERSION in the Makefile), or pass "
            "--overwrite if you really mean to replace it."
        )
    logger.info(f"Reading supergrid from {hgrid_url}")
    with fsspec.open(hgrid_url) as f:
        hgrid = xr.open_dataset(f).load()
    source_grid = convert_supergrid(hgrid)
    source_grid.attrs["history"] = (
        f"Tracer-cell geometry extracted from {hgrid_url} by "
        "scripts/gfdl_om4/pipeline/weights.py."
    )
    target_grid = make_target_grid(target_grid_name)
    logger.info(
        f"Computing conservative weights: {dict(source_grid.sizes)} -> "
        f"{target_grid_name} {dict(target_grid.sizes)}"
    )
    regridder = xe.Regridder(source_grid, target_grid, **REGRIDDER_KWARGS)
    with tempfile.TemporaryDirectory() as tmpdir:
        source_grid_path = os.path.join(tmpdir, SOURCE_GRID_FILENAME)
        weights_path = os.path.join(tmpdir, WEIGHTS_FILENAME)
        source_grid.reset_coords().to_netcdf(source_grid_path)
        regridder.to_netcdf(weights_path)
        for filename, local_path in [
            (SOURCE_GRID_FILENAME, source_grid_path),
            (WEIGHTS_FILENAME, weights_path),
        ]:
            url = f"{output_url.rstrip('/')}/{filename}"
            logger.info(f"Uploading {url}")
            _upload(local_path, url)
    logger.info("Weight artifact complete")


# One regridder per (artifact, target grid) and one A_J per artifact, per
# worker process.
_REGRIDDER_CACHE: dict[tuple[str, str], xe.Regridder] = {}
_AREACELLO_CACHE: dict[str, xr.DataArray] = {}


def _artifact_url(weights_url: str, filename: str) -> str:
    return f"{weights_url.rstrip('/')}/{filename}"


def _assert_artifact_attrs(weights_path: str, url: str) -> None:
    """Raise unless ``weights.nc`` carries every native-area artifact attr."""
    with xr.open_dataset(weights_path) as ds:
        missing = [name for name in ARTIFACT_ATTRS if name not in ds.attrs]
    if missing:
        raise ValueError(
            f"{url} is not a native-area weight artifact: weights.nc lacks "
            f"attrs {missing}. Build one with pipeline.native_area_weights."
        )


def read_weights(weights_url: str) -> Triplets:
    """Raw xESMF triplets ``(S, row, col)`` of the artifact at the
    ``weights_url`` prefix, 0-based, NaN markers kept."""
    with fsspec.open(_artifact_url(weights_url, WEIGHTS_FILENAME)) as f:
        ds = xr.open_dataset(f).load()
    return ds.S.values, ds.row.values - 1, ds.col.values - 1


def open_areacello(weights_url: str) -> xr.DataArray:
    """Target cell areas ``A_J`` (m², float64, 0 on unmapped rows) stored
    with the native-area artifact at the ``weights_url`` prefix; cached."""
    if weights_url not in _AREACELLO_CACHE:
        with fsspec.open(_artifact_url(weights_url, AREACELLO_FILENAME)) as f:
            area = xr.open_dataset(f).load()["areacello"]
        _AREACELLO_CACHE[weights_url] = area.astype(np.float64)
    return _AREACELLO_CACHE[weights_url]


def open_source_grid(weights_url: str) -> xr.Dataset:
    """Read the source-grid geometry (centers, corners, rotation angle)
    stored with the weight artifact at the ``weights_url`` prefix."""
    with fsspec.open(f"{weights_url.rstrip('/')}/{SOURCE_GRID_FILENAME}") as f:
        return xr.open_dataset(f).load().set_coords(["lon", "lat", "lon_b", "lat_b"])


def get_regridder(weights_url: str, target_grid_name: str) -> xe.Regridder:
    """Load the native-area weight artifact at the ``weights_url`` prefix
    into a cached conservative regridder for the named Gaussian target grid.

    Raises ``ValueError`` on an artifact lacking any native-area attr."""
    key = (weights_url, target_grid_name)
    if key not in _REGRIDDER_CACHE:
        logger.info(f"Loading regridding weights from {weights_url}")
        source_grid = open_source_grid(weights_url)
        target_grid = make_target_grid(target_grid_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, WEIGHTS_FILENAME)
            _download(_artifact_url(weights_url, WEIGHTS_FILENAME), weights_path)
            _assert_artifact_attrs(weights_path, weights_url)
            _REGRIDDER_CACHE[key] = xe.Regridder(
                source_grid,
                target_grid,
                weights=weights_path,
                reuse_weights=True,
                **REGRIDDER_KWARGS,
            )
    return _REGRIDDER_CACHE[key]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute conservative regridding weights and write "
        "them to GCS as a versioned artifact."
    )
    parser.add_argument(
        "--hgrid-url", required=True, help="URL of the MOM6 supergrid netCDF"
    )
    parser.add_argument(
        "--target-grid", required=True, help="Target Gaussian grid name, e.g. F90"
    )
    parser.add_argument(
        "--output-url", required=True, help="GCS prefix for the weight artifact"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing weight artifact at the output URL "
        "(default: refuse; bump the artifact version instead)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_weights(args.hgrid_url, args.target_grid, args.output_url, args.overwrite)


if __name__ == "__main__":
    main()
