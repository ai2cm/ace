"""Build the C96-path weight artifact: OM4 -> C96 (CM4 exchange grid) ->
target (``fregrid`` conserve_order1), in native cell areas.

The artifact has the native-area format of
:mod:`pipeline.native_area_weights` (``source_grid.nc``, ``areacello.nc``,
``weights.nc``), so :func:`pipeline.weights.get_regridder` loads it
unchanged. ŵ and ``areacello`` are
:func:`pipeline.native_area.cm4_chain_native_area_weights`. ``weights.nc``
also carries ``wet_columns_only = 1`` (ŵ has no columns on dry native cells)
and, written last after every assert and gate passed, ``regrid_path = "c96"``.

Optional gate, run when ``--am4-land-static-url`` is given (J: target cell):

    max_J |cm4_ocean_fraction_J − (1 − land_fraction_J)| ≤ LF_A_TOL

    python -m pipeline.c96_weights \\
        --cm4-xgrid-tar-url <CM4_XGRID_TAR_URL> \\
        --fregrid-remap-url <FREGRID_REMAP_URL_F90> \\
        --target-grid F90 \\
        --a-source-url <statics store with areacello, wet> \\
        --source-grid-weights-url <direct artifact holding source_grid.nc> \\
        [--am4-land-static-url <land_static.zarr on the target grid>] \\
        --ace-commit <sha> \\
        --output-url <WEIGHTS_URL_ROOT>/<version>/<artifact name>
"""

import argparse
import datetime
import logging
import shlex
import sys

import fsspec
import numpy as np
import xarray as xr

from .cm4_xgrid import read_cm4_xgrid, read_fregrid_remap
from .grids import make_target_grid
from .native_area import (
    F64_RTOL,
    assert_close,
    cm4_chain_native_area_weights,
    cm4_ocean_fraction,
)
from .native_area_weights import ARTIFACT_FILENAMES, write_native_area_artifact
from .weights import (
    REGRID_PATH_ATTR,
    REGRID_PATH_C96,
    REGRID_PATH_DESCRIPTION_ATTR,
    WET_COLUMNS_ONLY_ATTR,
    open_source_grid,
)
from .zarr_io import make_zarr_store

logger = logging.getLogger(__name__)

# Max |ocean_fraction - (1 - land_fraction)| per target cell, where land_fraction is a
# reference atmosphere land fraction on the target grid. The reference is stored as
# float32, so on cells with no ocean 1 - land_fraction is rounding residue, not zero.
# The tolerance is a round value above the largest such residue in the reference. An
# ocean fraction built from the exchange grid and remap file the reference was made
# with stays within it on every cell; one built through a different regridding chain
# exceeds it on many cells.
LF_A_TOL = 1e-4

NO_LAND_STATIC = "none provided"
Q6_OPTION = "a"
LAND_FRACTION = "land_fraction"


def regrid_path_description(target_grid_name: str) -> str:
    return (
        f"OM4 0.25° tripolar -> C96 (CM4 xgrid) -> {target_grid_name} "
        "(fregrid conserve_order1)"
    )


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
            "Published artifacts are immutable: bump C96_PATH_WEIGHTS_VERSION "
            "in the Makefile."
        )


def _assert_remap_covers_target(gauss_area: np.ndarray, remap_url: str, name: str):
    n_empty = int((gauss_area <= 0).sum())
    if n_empty:
        raise ValueError(
            f"{remap_url} leaves {n_empty} of {gauss_area.size} cells of target "
            f"grid {name} uncovered: its Gaussian grid is not {name}"
        )


def _land_fraction_on_target(
    land_fraction: xr.DataArray, target_grid: xr.Dataset, url: str
) -> np.ndarray:
    """``land_fraction`` flattened (lat, lon), its lat/lon checked against the
    target grid's to ``F64_RTOL · 360`` (as ``weights._on_target_grid``)."""
    for name in ("lat", "lon"):
        stored, ref = land_fraction[name], target_grid[name]
        if stored.dims != ref.dims or stored.shape != ref.shape:
            raise ValueError(
                f"{url}: {name} {dict(stored.sizes)} is not the target grid's "
                f"{dict(ref.sizes)}"
            )
        assert_close(
            stored.values,
            ref.values,
            rtol=F64_RTOL,
            scale=360.0,
            context=f"{url}: {name} vs target grid",
        )
    return land_fraction.transpose("lat", "lon").values.astype(np.float64).ravel()


def land_fraction_gate(
    ocean_fraction: np.ndarray,
    land_fraction: np.ndarray,
    urls: tuple[str, str, str],
) -> dict[str, float]:
    """Gate ``|cm4_ocean_fraction_J − (1 − land_fraction_J)| ≤ LF_A_TOL``;
    returns its max and rms. ``urls``: xgrid tar, remap file, land static."""
    diff = ocean_fraction - (1.0 - land_fraction)
    xgrid_url, remap_url, land_url = urls
    assert_close(
        ocean_fraction,
        1.0 - land_fraction,
        rtol=LF_A_TOL,
        scale=1.0,
        context=(
            f"cm4_ocean_fraction_J (xgrid {xgrid_url}, fregrid {remap_url}) vs "
            f"1 − land_fraction_J ({land_url}), LF_A_TOL"
        ),
    )
    return {
        "max_abs_cm4_ocean_fraction_minus_one_minus_lf_A": float(np.abs(diff).max()),
        "rms_cm4_ocean_fraction_minus_one_minus_lf_A": float(np.sqrt(np.mean(diff**2))),
    }


def build_c96_path_artifact(
    cm4_xgrid_tar_url: str,
    fregrid_remap_url: str,
    target_grid_name: str,
    a_source_url: str,
    source_grid_weights_url: str,
    am4_land_static_url: str | None,
    ace_commit: str,
    output_url: str,
    argv: list[str],
) -> None:
    """Build the C96-path artifact and write it under ``output_url``.

    ``argv`` is the command's arguments, recorded in ``history``. Every build
    assert and the optional land-fraction gate raise before any write."""
    _assert_prefix_absent(output_url)
    target_grid = make_target_grid(target_grid_name)
    xgrid = read_cm4_xgrid(cm4_xgrid_tar_url)
    fregrid_overlap = read_fregrid_remap(
        fregrid_remap_url, target_grid_name, xgrid.c96_shape
    )
    _assert_remap_covers_target(
        np.asarray(fregrid_overlap.sum(axis=1)).ravel(),
        fregrid_remap_url,
        target_grid_name,
    )
    source_grid = open_source_grid(source_grid_weights_url)
    if tuple(source_grid["lon"].shape) != xgrid.om4_shape:
        raise ValueError(
            f"{source_grid_weights_url} source_grid.nc {source_grid['lon'].shape} "
            f"is not the xgrid's OM4 {xgrid.om4_shape}"
        )
    dims = source_grid["lon"].dims
    statics = xr.open_zarr(
        make_zarr_store(a_source_url), chunks=None, decode_timedelta=False
    )
    native_area = statics["areacello"].transpose(*dims).values
    wet = statics["wet"].transpose(*dims).values
    logger.info(
        f"Composing {cm4_xgrid_tar_url} and {fregrid_remap_url} in the native "
        f"areas of {a_source_url}"
    )
    result = cm4_chain_native_area_weights(
        xgrid.xgrid_ocean_area,
        xgrid.xgrid_land_area,
        fregrid_overlap,
        native_area,
        wet,
        source_urls={
            "xgrid": cm4_xgrid_tar_url,
            "fregrid": fregrid_remap_url,
            "native": a_source_url,
        },
    )

    gate = {
        "max_abs_cm4_ocean_fraction_minus_one_minus_lf_A": float("nan"),
        "rms_cm4_ocean_fraction_minus_one_minus_lf_A": float("nan"),
    }
    if am4_land_static_url:
        land = xr.open_zarr(
            make_zarr_store(am4_land_static_url), chunks=None, decode_timedelta=False
        )[LAND_FRACTION]
        gate = land_fraction_gate(
            cm4_ocean_fraction(
                xgrid.xgrid_ocean_area, xgrid.xgrid_land_area, fregrid_overlap
            ),
            _land_fraction_on_target(land, target_grid, am4_land_static_url),
            (cm4_xgrid_tar_url, fregrid_remap_url, am4_land_static_url),
        )
        logger.info(f"land_fraction gate passed: {gate}")

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    attrs = {
        "source_weights_url": cm4_xgrid_tar_url,
        "a_source_url": a_source_url,
        "esmpy_area_rtol": float("nan"),
        "esmpy_max_rel_residual": result.max_rel_residual,
        "a_over_a_tilde_spread": result.a_over_a_tilde_spread,
        "fregrid_remap_url": fregrid_remap_url,
        WET_COLUMNS_ONLY_ATTR: 1,
        "am4_land_static_url": am4_land_static_url or NO_LAND_STATIC,
        "q6_option": Q6_OPTION,
        **gate,
        "ace_commit": ace_commit,
        "source_grid_url": source_grid_weights_url,
        "history": f"{timestamp}: python -m pipeline.c96_weights {shlex.join(argv)}",
    }
    # Last: only an artifact that passed every assert and gate selects the path.
    attrs[REGRID_PATH_ATTR] = REGRID_PATH_C96
    attrs[REGRID_PATH_DESCRIPTION_ATTR] = regrid_path_description(target_grid_name)
    logger.info(f"a_over_a_tilde_spread={result.a_over_a_tilde_spread}")
    write_native_area_artifact(result, source_grid, target_grid, attrs, output_url)
    logger.info("C96-path weight artifact complete")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description="Build the OM4 -> C96 -> target native-area weight artifact."
    )
    parser.add_argument("--cm4-xgrid-tar-url", required=True)
    parser.add_argument("--fregrid-remap-url", required=True)
    parser.add_argument(
        "--target-grid", required=True, help="Target Gaussian grid name, e.g. F90"
    )
    parser.add_argument(
        "--a-source-url",
        required=True,
        help="statics zarr store whose areacello, wet are the native area, mask",
    )
    parser.add_argument(
        "--source-grid-weights-url",
        required=True,
        help="direct native-area artifact whose source_grid.nc is copied",
    )
    parser.add_argument(
        "--am4-land-static-url",
        default=None,
        help="land_static zarr on the target grid; gates the ocean fraction",
    )
    parser.add_argument("--ace-commit", required=True)
    parser.add_argument(
        "--output-url", required=True, help="GCS prefix for the new artifact"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_c96_path_artifact(
        args.cm4_xgrid_tar_url,
        args.fregrid_remap_url,
        args.target_grid,
        args.a_source_url,
        args.source_grid_weights_url,
        args.am4_land_static_url,
        args.ace_commit,
        args.output_url,
        argv,
    )


if __name__ == "__main__":
    main()
