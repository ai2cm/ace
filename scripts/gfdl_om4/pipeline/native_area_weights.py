"""Build a native-area weight artifact from an ESMF weight artifact.

Rebuilds the conservative weights of the artifact at ``source_weights_url``
in native cell areas ``a_i`` (the ``areacello`` of the ``a_source_url``
statics store), per :func:`pipeline.native_area.native_area_weights`, and
writes a new artifact prefix holding

- ``weights.nc`` — xESMF format, ``S = ŵ_Ji`` in float64, no entries on rows
  ESMF left unmapped; root attrs :data:`pipeline.native_area.ARTIFACT_ATTRS`;
- ``areacello.nc`` — ``areacello`` = ``A_J`` (m², float64, 0 on unmapped rows);
- ``source_grid.nc`` — the source artifact's grid.

``weights.nc`` also carries ``regrid_path = "direct"`` and
``regrid_path_description`` (read by the pipelines, :mod:`pipeline.weights`),
``ace_commit`` and a CF ``history`` line holding the build command.

Published artifacts are immutable: an existing prefix raises
``FileExistsError``, and a superseding artifact takes a new version prefix
(``NATIVE_AREA_WEIGHTS_VERSION`` in the Makefile).

    python -m pipeline.native_area_weights \\
        --source-weights-url <WEIGHTS_URL_ROOT>/om4-tripolar-0.25deg-to-F90 \\
        --target-grid F90 \\
        --a-source-url <statics store with areacello> \\
        --ace-commit <ace commit> \\
        --output-url <WEIGHTS_URL_ROOT>/<version>/om4-tripolar-0.25deg-to-F90
"""

import argparse
import datetime
import logging
import os
import shlex
import sys
import tempfile

import fsspec
import numpy as np
import xarray as xr

from .grids import MEAN_EARTH_RADIUS_M, make_target_grid
from .native_area import (
    ARTIFACT_ATTRS,
    ESMPY_AREA_RTOL,
    NativeAreaWeights,
    esmf_cell_areas,
    native_area_weights,
)
from .weights import (
    AREACELLO_FILENAME,
    REGRID_PATH_ATTR,
    REGRID_PATH_DESCRIPTION_ATTR,
    REGRID_PATH_DIRECT,
    SOURCE_GRID_FILENAME,
    WEIGHTS_FILENAME,
    _upload,
    open_source_grid,
    read_weights,
)
from .zarr_io import make_zarr_store

logger = logging.getLogger(__name__)

ARTIFACT_FILENAMES = (SOURCE_GRID_FILENAME, AREACELLO_FILENAME, WEIGHTS_FILENAME)


def regrid_path_description(target_grid_name: str) -> str:
    return f"OM4 0.25° tripolar -> {target_grid_name} (direct)"


def _assert_prefix_absent(output_url: str) -> None:
    fs, _ = fsspec.url_to_fs(output_url)
    existing = [
        url
        for filename in ARTIFACT_FILENAMES
        if fs.exists(url := f"{output_url.rstrip('/')}/{filename}")
    ]
    if existing:
        raise FileExistsError(
            f"Weight artifact already exists at {output_url} ({existing}). "
            "Published artifacts are immutable: bump "
            "NATIVE_AREA_WEIGHTS_VERSION in the Makefile."
        )


def _target_area(area: np.ndarray, target_grid: xr.Dataset) -> xr.DataArray:
    """``A_J`` on the target grid, in the dims and cell order of regridder output."""
    lat, lon = target_grid["lat"], target_grid["lon"]
    if lat.ndim == 1:
        dims: tuple = ("lat", "lon")
        shape: tuple = (lat.size, lon.size)
    else:
        dims, shape = lat.dims, lat.shape
    return xr.DataArray(
        area.reshape(shape),
        dims=dims,
        coords={"lat": lat, "lon": lon},
        attrs={"long_name": "target cell area covered by native cells", "units": "m2"},
        name="areacello",
    )


def write_native_area_artifact(
    result: NativeAreaWeights,
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    attrs: dict,
    output_url: str,
) -> None:
    """Write ``result`` as a weight artifact under the ``output_url`` prefix.

    Raises ``FileExistsError`` on an existing prefix and ``ValueError`` when
    ``attrs`` lacks an artifact attr, before any write."""
    _assert_prefix_absent(output_url)
    missing = [name for name in ARTIFACT_ATTRS if name not in attrs]
    if missing:
        raise ValueError(f"artifact attrs missing {missing}")
    w_hat = result.w_hat.tocoo()
    weights = xr.Dataset(
        {
            "S": ("n_s", w_hat.data.astype(np.float64)),
            "col": ("n_s", w_hat.col + 1),  # xESMF files are 1-based
            "row": ("n_s", w_hat.row + 1),
        },
        attrs=attrs,
    )
    area = _target_area(result.area.astype(np.float64), target_grid)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {name: os.path.join(tmpdir, name) for name in ARTIFACT_FILENAMES}
        source_grid.reset_coords().to_netcdf(paths[SOURCE_GRID_FILENAME])
        area.to_dataset().to_netcdf(paths[AREACELLO_FILENAME])
        weights.to_netcdf(paths[WEIGHTS_FILENAME])
        # weights.nc last: a prefix without it is never loadable.
        for name in ARTIFACT_FILENAMES:
            url = f"{output_url.rstrip('/')}/{name}"
            logger.info(f"Uploading {url}")
            _upload(paths[name], url)


def build_native_area_artifact(
    source_weights_url: str,
    target_grid_name: str,
    a_source_url: str,
    ace_commit: str,
    output_url: str,
    argv: list[str],
) -> None:
    """Rebuild the artifact at ``source_weights_url`` in the native areas of
    ``a_source_url`` and write it under ``output_url``.

    ``argv`` is the command's arguments, recorded in ``history``. Every
    artifact-build assert raises before any write."""
    _assert_prefix_absent(output_url)
    target_grid = make_target_grid(target_grid_name)
    source_grid = open_source_grid(source_weights_url)
    w = read_weights(source_weights_url)
    statics = xr.open_zarr(
        make_zarr_store(a_source_url), chunks=None, decode_timedelta=False
    )
    a = statics["areacello"].transpose(*source_grid["lon"].dims).values
    logger.info(
        f"Rebuilding {source_weights_url} in native areas: "
        f"{dict(source_grid.sizes)} -> {dict(target_grid.sizes)}"
    )
    result = native_area_weights(
        w,
        esmf_cell_areas(target_grid),
        esmf_cell_areas(source_grid),
        a,
        radius=MEAN_EARTH_RADIUS_M,
    )
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    attrs = {
        "source_weights_url": source_weights_url,
        "a_source_url": a_source_url,
        "esmpy_area_rtol": ESMPY_AREA_RTOL,
        "esmpy_max_rel_residual": result.max_rel_residual,
        "a_over_a_tilde_spread": result.a_over_a_tilde_spread,
        "ace_commit": ace_commit,
        "history": (
            f"{timestamp}: python -m pipeline.native_area_weights {shlex.join(argv)}"
        ),
    }
    # Last: only an artifact that passed every assert selects the path.
    attrs[REGRID_PATH_ATTR] = REGRID_PATH_DIRECT
    attrs[REGRID_PATH_DESCRIPTION_ATTR] = regrid_path_description(target_grid_name)
    logger.info(
        f"esmpy_max_rel_residual={result.max_rel_residual}, "
        f"a_over_a_tilde_spread={result.a_over_a_tilde_spread}"
    )
    write_native_area_artifact(result, source_grid, target_grid, attrs, output_url)
    logger.info("Native-area weight artifact complete")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description="Rebuild an ESMF weight artifact in native cell areas."
    )
    parser.add_argument(
        "--source-weights-url",
        required=True,
        help="GCS prefix of the ESMF weight artifact",
    )
    parser.add_argument(
        "--target-grid", required=True, help="Target Gaussian grid name, e.g. F90"
    )
    parser.add_argument(
        "--a-source-url",
        required=True,
        help="statics zarr store whose areacello is the native cell area",
    )
    parser.add_argument("--ace-commit", required=True)
    parser.add_argument(
        "--output-url", required=True, help="GCS prefix for the new artifact"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_native_area_artifact(
        args.source_weights_url,
        args.target_grid,
        args.a_source_url,
        args.ace_commit,
        args.output_url,
        argv,
    )


if __name__ == "__main__":
    main()
