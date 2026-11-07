"""Checks of the C96 path, written to JSON.

Subcommand ``operators``: the readers of :mod:`pipeline.cm4_xgrid` on one
target grid. c C96 cell, i OM4 cell, J Gaussian cell; symbols as in
:mod:`pipeline.cm4_xgrid`, plus

    native_area_i, wet_i      ``areacello``, ``wet`` of the native statics (m², 1)
    earth_radius_xgrid        radius of the FRE-NCtools xgrid (m)
    center_distance(p, q)     great-circle distance of cell centers (degrees)
    center_radius_max(grid)   max over cells of center_distance(center, vertex)
    band_area_J             = earth_radius_xgrid² Δlon_J
                              · (sin lat_b_north − sin lat_b_south)

Gates (raise):

    center_distance(c, i) ≤ XGRID_INDEX_MAX_DEG   every xgrid_ocean_area_ci entry
    center_distance(c, J) ≤ XGRID_INDEX_MAX_DEG   every fregrid_overlap_Jc entry
    |fregrid_gauss_area_J − band_area_J| ≤ AREA_RTOL band_area_J
    fregrid_c96_area_c > 0
    {i : xgrid_om4_area_i > 0} = {i : wet_i = 1}

with XGRID_INDEX_MAX_DEG = center_radius_max(C96)
+ max(center_radius_max(OM4), center_radius_max(Gaussian)). Reports: area
ratio quantiles and the Gaussian sphere fraction.

    python -m pipeline.c96_checks operators --target-grid F90 \\
        --cm4-xgrid-tar-url <CM4_XGRID_TAR_URL> \\
        --fregrid-remap-url <FREGRID_REMAP_URL_F90> \\
        --a-source-url <NATIVE_AREA_A_SOURCE_URL> \\
        --source-grid-weights-url <weight artifact holding source_grid.nc> \\
        --output-json <path or URL>/01-operators-F90.json
"""

import argparse
import json
import logging

import fsspec
import numpy as np
import scipy.sparse
import xarray as xr

from .cm4_xgrid import CM4Xgrid, read_cm4_xgrid, read_fregrid_remap
from .grids import make_target_grid
from .native_area import AREA_RTOL, F64_RTOL, assert_bound, assert_close
from .weights import open_source_grid
from .zarr_io import make_zarr_store

logger = logging.getLogger(__name__)

# FRE-NCtools RADIUS:
# analysis/2026-10-16-0950-om4-am4-regrid-path-fields/01g-xgrid-area.json
# ::radius.R_x_m
EARTH_RADIUS_XGRID_M = 6371.0e3
QUANTILES = {
    "q0": 0.0,
    "q0p1": 0.001,
    "q1": 0.01,
    "q50": 0.5,
    "q99": 0.99,
    "q99p9": 0.999,
    "q100": 1.0,
}


def _unit_vectors(lon, lat) -> np.ndarray:
    lon, lat = np.deg2rad(np.asarray(lon)), np.deg2rad(np.asarray(lat))
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def center_distance(lon_p, lat_p, lon_q, lat_q) -> np.ndarray:
    """Great-circle distance (degrees) between points p and q."""
    p, q = _unit_vectors(lon_p, lat_p), _unit_vectors(lon_q, lat_q)
    cross = np.linalg.norm(np.cross(p, q), axis=-1)
    return np.rad2deg(np.arctan2(cross, np.sum(p * q, axis=-1)))


def center_radius_max(lon, lat, corner_lon, corner_lat) -> float:
    """Max over cells of the distance from the center (``(..., ny, nx)``)
    to its four vertices (``(..., ny + 1, nx + 1)``)."""
    vertices = [
        (slice(None, -1), slice(None, -1)),
        (slice(None, -1), slice(1, None)),
        (slice(1, None), slice(None, -1)),
        (slice(1, None), slice(1, None)),
    ]
    return max(
        float(
            np.max(
                center_distance(lon, lat, corner_lon[..., y, x], corner_lat[..., y, x])
            )
        )
        for y, x in vertices
    )


def _gauss_corners(target: xr.Dataset) -> tuple[np.ndarray, ...]:
    lon, lat = np.meshgrid(target["lon"].values, target["lat"].values)
    corner_lon, corner_lat = np.meshgrid(target["lon_b"].values, target["lat_b"].values)
    return lon, lat, corner_lon, corner_lat


def band_area(target: xr.Dataset) -> np.ndarray:
    """``band_area_J`` (m²), flattened ``(lat, lon)``."""
    lat_b, lon_b = (
        np.deg2rad(target["lat_b"].values),
        np.deg2rad(target["lon_b"].values),
    )
    area = EARTH_RADIUS_XGRID_M**2 * np.outer(np.diff(np.sin(lat_b)), np.diff(lon_b))
    return area.ravel()


def _entries(m: scipy.sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    coo = m.tocoo()
    return coo.row, coo.col


def xgrid_index_max_deg(
    xgrid: CM4Xgrid, source_grid: xr.Dataset, target: xr.Dataset
) -> dict[str, float]:
    """``XGRID_INDEX_MAX_DEG`` and its terms."""
    c96 = center_radius_max(
        xgrid.c96_lon.reshape(xgrid.c96_shape),
        xgrid.c96_lat.reshape(xgrid.c96_shape),
        xgrid.c96_corner_lon,
        xgrid.c96_corner_lat,
    )
    om4 = center_radius_max(
        *(source_grid[n].values for n in ("lon", "lat", "lon_b", "lat_b"))
    )
    gauss = center_radius_max(*_gauss_corners(target))
    return {
        "center_radius_max_c96_deg": c96,
        "center_radius_max_om4_deg": om4,
        "center_radius_max_gauss_deg": gauss,
        "XGRID_INDEX_MAX_DEG": c96 + max(om4, gauss),
    }


def check_index_maps(
    xgrid: CM4Xgrid,
    fregrid_overlap: scipy.sparse.csr_matrix,
    source_grid: xr.Dataset,
    target: xr.Dataset,
    max_deg: float,
) -> dict[str, float]:
    """Gate every xgrid and ``fregrid`` entry's center distance by ``max_deg``."""
    c, i = _entries(xgrid.xgrid_ocean_area)
    om4_lon, om4_lat = (source_grid[n].values.ravel() for n in ("lon", "lat"))
    d_ci = center_distance(xgrid.c96_lon[c], xgrid.c96_lat[c], om4_lon[i], om4_lat[i])
    assert_bound(
        d_ci,
        upper=max_deg,
        rtol=F64_RTOL,
        scale=max_deg,
        context="xgrid index map: center_distance(c, i) ≤ XGRID_INDEX_MAX_DEG",
    )
    J, c = _entries(fregrid_overlap)
    lon, lat, _, _ = _gauss_corners(target)
    d_cJ = center_distance(
        xgrid.c96_lon[c], xgrid.c96_lat[c], lon.ravel()[J], lat.ravel()[J]
    )
    assert_bound(
        d_cJ,
        upper=max_deg,
        rtol=F64_RTOL,
        scale=max_deg,
        context="fregrid index map: center_distance(c, J) ≤ XGRID_INDEX_MAX_DEG",
    )
    return {
        "max_center_distance_ci_deg": float(d_ci.max()),
        "max_center_distance_cJ_deg": float(d_cJ.max()),
    }


def check_band_area(
    fregrid_overlap: scipy.sparse.csr_matrix, target: xr.Dataset
) -> float:
    """Gate ``fregrid_gauss_area_J`` against ``band_area_J``; returns the max
    relative difference."""
    gauss_area = np.asarray(fregrid_overlap.sum(axis=1)).ravel()
    expected = band_area(target)
    assert_close(
        gauss_area,
        expected,
        rtol=AREA_RTOL,
        scale=expected,
        context="fregrid_gauss_area_J vs make_target_grid band_area_J",
    )
    return float(np.max(np.abs(gauss_area / expected - 1)))


def check_fregrid_c96_area_positive(fregrid_overlap: scipy.sparse.csr_matrix) -> None:
    positive = (np.asarray(fregrid_overlap.sum(axis=0)).ravel() > 0).astype(np.float64)
    assert_bound(
        positive, lower=1.0, rtol=F64_RTOL, scale=1.0, context="fregrid_c96_area_c > 0"
    )


def check_wet_support(
    xgrid: CM4Xgrid, wet: np.ndarray, tar_url: str, a_source_url: str
) -> None:
    """Gate ``{i : xgrid_om4_area_i > 0} = {i : wet_i = 1}``."""
    covered = (xgrid.xgrid_om4_area > 0).astype(np.float64)
    assert_close(
        covered,
        np.asarray(wet, dtype=np.float64).ravel(),
        rtol=F64_RTOL,
        scale=1.0,
        context=(
            f"{{i : xgrid_om4_area_i > 0}} (CM4_XGRID_TAR_URL {tar_url}) vs "
            f"{{i : wet_i = 1}} (NATIVE_AREA_A_SOURCE_URL {a_source_url}): "
            "differing OM4 cells"
        ),
    )


def _quantiles(x: np.ndarray) -> dict[str, float]:
    return {
        k: float(v) for k, v in zip(QUANTILES, np.quantile(x, list(QUANTILES.values())))
    }


def operator_checks(
    xgrid: CM4Xgrid,
    fregrid_overlap: scipy.sparse.csr_matrix,
    source_grid: xr.Dataset,
    target: xr.Dataset,
    native_area: np.ndarray,
    wet: np.ndarray,
    tar_url: str,
    a_source_url: str,
) -> dict:
    """Every gate and report of ``operators``; raises on a failed gate."""
    if tuple(source_grid["lon"].shape) != xgrid.om4_shape:
        raise ValueError(
            f"source_grid.nc {source_grid['lon'].shape} is not the xgrid's OM4 "
            f"{xgrid.om4_shape}"
        )
    bound = xgrid_index_max_deg(xgrid, source_grid, target)
    index = check_index_maps(
        xgrid, fregrid_overlap, source_grid, target, bound["XGRID_INDEX_MAX_DEG"]
    )
    band = check_band_area(fregrid_overlap, target)
    check_fregrid_c96_area_positive(fregrid_overlap)
    check_wet_support(xgrid, wet, tar_url, a_source_url)

    fregrid_c96_area = np.asarray(fregrid_overlap.sum(axis=0)).ravel()
    fregrid_gauss_area = np.asarray(fregrid_overlap.sum(axis=1)).ravel()
    xgrid_vs_supergrid = (xgrid.xgrid_c96_area / xgrid.supergrid_c96_area - 1).reshape(
        xgrid.c96_shape[0], -1
    )
    wet_cells = np.asarray(wet).ravel() == 1
    return {
        "gates": {
            "index_map": {**bound, **index},
            "band_area": {"AREA_RTOL": AREA_RTOL, "max_rel_diff": band},
            "fregrid_c96_area_positive": True,
            "xgrid_om4_support_equals_wet": True,
        },
        "reports": {
            "fregrid_c96_area_over_xgrid_c96_area_minus_one": _quantiles(
                fregrid_c96_area / xgrid.xgrid_c96_area - 1
            ),
            "xgrid_c96_area_over_supergrid_c96_area_minus_one_per_tile": {
                f"tile{t + 1}": _quantiles(x) for t, x in enumerate(xgrid_vs_supergrid)
            },
            "earth_radius_xgrid_m": EARTH_RADIUS_XGRID_M,
            "sum_fregrid_gauss_area_over_sphere": float(
                fregrid_gauss_area.sum() / (4 * np.pi * EARTH_RADIUS_XGRID_M**2)
            ),
            "xgrid_om4_area_over_native_area_wet": _quantiles(
                xgrid.xgrid_om4_area[wet_cells]
                / np.asarray(native_area, dtype=np.float64).ravel()[wet_cells]
            ),
        },
    }


def run_operators(args: argparse.Namespace) -> dict:
    target = make_target_grid(args.target_grid)
    xgrid = read_cm4_xgrid(args.cm4_xgrid_tar_url)
    fregrid_overlap = read_fregrid_remap(
        args.fregrid_remap_url, args.target_grid, xgrid.c96_shape
    )
    source_grid = open_source_grid(args.source_grid_weights_url)
    dims = source_grid["lon"].dims
    statics = xr.open_zarr(
        make_zarr_store(args.a_source_url), chunks=None, decode_timedelta=False
    )
    native_area = statics["areacello"].transpose(*dims).values
    wet = statics["wet"].transpose(*dims).values
    result = operator_checks(
        xgrid,
        fregrid_overlap,
        source_grid,
        target,
        native_area,
        wet,
        args.cm4_xgrid_tar_url,
        args.a_source_url,
    )
    return {
        "target_grid": args.target_grid,
        "inputs": {
            "cm4_xgrid_tar_url": args.cm4_xgrid_tar_url,
            "fregrid_remap_url": args.fregrid_remap_url,
            "a_source_url": args.a_source_url,
            "source_grid_weights_url": args.source_grid_weights_url,
        },
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checks of the C96 path, written to JSON."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ops = sub.add_parser("operators", help="CM4 xgrid and fregrid reader checks")
    ops.add_argument("--target-grid", required=True, help="F90 or F22.5")
    ops.add_argument("--cm4-xgrid-tar-url", required=True)
    ops.add_argument("--fregrid-remap-url", required=True)
    ops.add_argument(
        "--a-source-url", required=True, help="statics zarr store with areacello, wet"
    )
    ops.add_argument(
        "--source-grid-weights-url",
        required=True,
        help="weight artifact prefix holding source_grid.nc (OM4 centers, corners)",
    )
    ops.add_argument("--output-json", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_operators(args)
    with fsspec.open(args.output_json, "w") as f:
        json.dump(result, f, indent=1)
    logger.info(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
