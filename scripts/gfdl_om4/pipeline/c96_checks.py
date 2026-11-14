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

Subcommand ``statics``: the C96-path artifact at ``--weights-url``, loaded by
:func:`pipeline.weights.get_regridder`, through
:func:`pipeline.native_area.native_area_statics` and
:func:`pipeline.native_area.check_static_identities` (with the artifact's
``wet_columns_only``) on the native statics of ``--a-source-url``: one
level, wetmask ``wet``, Δz = ``deptho``, ``sftof``.

    python -m pipeline.c96_checks statics --target-grid F90 \\
        --weights-url <C96-path artifact> \\
        --a-source-url <NATIVE_AREA_A_SOURCE_URL> \\
        --output-json <path or URL>/02-statics-F90.json

Subcommand ``stores``: a C96-path store against its direct-path twin, same
config and timesteps. p path in {direct, c96path}; per path p and timestep,
sums over the J where the summand is finite:

    C1:  Σ_J areacello_p,J sea_surface_fraction_p,J = Σ_i native_area_i wet_i0
         scale Σ_i native_area_i wet_i0
    C2:  Σ_J areacello_c96path,J sea_surface_fraction_c96path,J field_c96path,J
       = Σ_J areacello_direct,J sea_surface_fraction_direct,J field_direct,J
         scale max_p Σ_J areacello_p,J sea_surface_fraction_p,J |field_p,J|
         (per-ocean-area fields, ``--c2-fields``)
    C3:  Σ_J areacello_c96path,J field_c96path,J
       = Σ_J areacello_direct,J field_direct,J
         scale max_p Σ_J areacello_p,J |field_p,J|
         (full-cell fields, ``--c3-fields``)

each within ``AREA_RTOL · scale``. ``native_area_i`` and ``wet_i0`` (the
surface of the run's wetmask) come from the C96 config through the pipeline's
own readers. Report: the share of the C96 store's ocean area on

    gained_band = {J : sea_surface_fraction_direct,J = 0,
                       sea_surface_fraction_c96path,J > 0}

The JSON is written before a failed item raises.

    python -m pipeline.c96_checks stores --pipeline om4 \\
        --c96-config <C96 test config> \\
        --c96-store <C96 store> --direct-store <direct store> \\
        --c2-fields tos hfds --c3-fields hfds_total_area \\
        --output-json <path or URL>/04-stores-<config basename>.json
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
from .native_area import (
    AREA_RTOL,
    F64_RTOL,
    LEVEL_DIM,
    assert_bound,
    assert_close,
    check_static_identities,
    level_thickness,
    native_area_statics,
)
from .ocean_emulators_port import regrid_normalized
from .weights import (
    REGRID_PATH_ATTR,
    REGRID_PATH_C96,
    REGRID_PATH_DIRECT,
    get_regridder,
    open_areacello,
    open_source_grid,
    read_artifact_attrs,
    read_wet_columns_only,
)
from .zarr_io import TIME_DIM, make_zarr_store

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


def statics_checks(
    regridder,
    area: xr.DataArray,
    statics: xr.Dataset,
    *,
    wet_columns_only: bool,
    mask_source_urls: tuple[str, str],
) -> dict:
    """``native_area_statics`` and ``check_static_identities`` on one-level
    native ``statics`` (``areacello``, ``wet``, ``deptho``, ``sftof``);
    raises on a failed identity."""
    dims = regridder.in_horiz_dims
    wet = statics["wet"].transpose(*dims) == 1
    deptho = statics["deptho"].transpose(*dims).where(wet)
    interfaces = np.array([0.0, float(deptho.max())])
    wetmask = wet.expand_dims({LEVEL_DIM: [0.5 * interfaces[1]]})
    dz = level_thickness(deptho, interfaces)
    result = native_area_statics(
        regridder, area, wetmask, dz, statics["sftof"].transpose(*dims)
    )
    deptho_J = regrid_normalized(deptho.to_dataset(name="deptho"), regridder, wet)[0][
        "deptho"
    ]
    areacello = statics["areacello"].transpose(*dims).astype(np.float64)
    check_static_identities(
        result,
        regridder,
        wetmask,
        areacello,
        deptho,
        deptho_J,
        wet_columns_only=wet_columns_only,
        mask_source_urls=mask_source_urls,
    )
    stored = result.stored
    native_total = float(areacello.where(wet, 0.0).sum())
    ocean_area = result.ocean_area
    return {
        "gates": {"check_static_identities": True},
        "reports": {
            "sum_ocean_area_over_native_wet_area_minus_one": float(
                ocean_area.sum() / native_total - 1
            ),
            "n_target_cells": int(area.size),
            "n_target_cells_with_ocean": int((ocean_area > 0).sum()),
            "sea_surface_fraction_min": float(stored["sea_surface_fraction"].min()),
            "sea_surface_fraction_max": float(stored["sea_surface_fraction"].max()),
            "areacello_min_m2": float(stored["areacello"].min()),
        },
    }


def run_statics(args: argparse.Namespace) -> dict:
    regridder = get_regridder(args.weights_url, args.target_grid)
    area = open_areacello(args.weights_url, args.target_grid)
    wet_columns_only = read_wet_columns_only(args.weights_url)
    artifact_a_source_url = read_artifact_attrs(args.weights_url)["a_source_url"]
    statics = xr.open_zarr(
        make_zarr_store(args.a_source_url), chunks=None, decode_timedelta=False
    )[["areacello", "wet", "deptho", "sftof"]].load()
    result = statics_checks(
        regridder,
        area,
        statics,
        wet_columns_only=wet_columns_only,
        mask_source_urls=(artifact_a_source_url, args.a_source_url),
    )
    return {
        "target_grid": args.target_grid,
        "inputs": {
            "weights_url": args.weights_url,
            "a_source_url": args.a_source_url,
            "artifact_a_source_url": artifact_a_source_url,
        },
        "wet_columns_only": wet_columns_only,
        **result,
    }


def _ocean_sum(area: xr.DataArray, field: xr.DataArray) -> xr.DataArray:
    """Σ_J area_J field_J over the J where the summand is finite, per time."""
    summand = (area * field).astype(np.float64)
    spatial = [d for d in summand.dims if d != TIME_DIM]
    return summand.where(np.isfinite(summand), 0.0).sum(spatial)


def _pair_item(lhs, rhs, scale) -> dict:
    lhs, rhs, scale = (
        np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in (lhs, rhs, scale)
    )
    rel = np.abs(lhs - rhs) / scale
    return {
        "lhs": lhs.tolist(),
        "rhs": rhs.tolist(),
        "scale": scale.tolist(),
        "max_abs_diff_over_scale": float(rel.max()),
        "AREA_RTOL": AREA_RTOL,
        "pass": bool(np.all(rel <= AREA_RTOL)),
    }


def stores_checks(
    c96: xr.Dataset,
    direct: xr.Dataset,
    native_wet_area: float,
    c2_fields: list[str],
    c3_fields: list[str],
) -> dict:
    """C1–C3 between a C96-path and a direct-path store, and the
    ``gained_band`` report; ``pass`` is the conjunction of every item."""
    for ds, path in ((c96, REGRID_PATH_C96), (direct, REGRID_PATH_DIRECT)):
        if ds.attrs.get(REGRID_PATH_ATTR) != path:
            raise ValueError(
                f"store root {REGRID_PATH_ATTR} is "
                f"{ds.attrs.get(REGRID_PATH_ATTR)!r}; expected {path!r}"
            )
    xr.align(c96[TIME_DIM], direct[TIME_DIM], join="exact")
    stores = {"direct": direct, "c96path": c96}
    ocean_area = {
        p: ds["areacello"].astype(np.float64) * ds["sea_surface_fraction"]
        for p, ds in stores.items()
    }
    c1 = {
        p: _pair_item(
            _ocean_sum(ocean_area[p], xr.ones_like(ocean_area[p])),
            native_wet_area,
            native_wet_area,
        )
        for p in stores
    }

    def field_item(weight: dict, name: str) -> dict:
        totals = {p: _ocean_sum(weight[p], stores[p][name]) for p in stores}
        scale = np.maximum(
            *(_ocean_sum(weight[p], abs(stores[p][name])).values for p in stores)
        )
        return _pair_item(totals["c96path"].values, totals["direct"].values, scale)

    cell_area = {p: ds["areacello"].astype(np.float64) for p, ds in stores.items()}
    c2 = {name: field_item(ocean_area, name) for name in c2_fields}
    c3 = {name: field_item(cell_area, name) for name in c3_fields}
    gained = (direct["sea_surface_fraction"] == 0) & (c96["sea_surface_fraction"] > 0)
    c96_ocean_area = ocean_area["c96path"]
    items = [*c1.values(), *c2.values(), *c3.values()]
    return {
        "pass": all(item["pass"] for item in items),
        "time": [str(t) for t in c96[TIME_DIM].values],
        "native_wet_area_m2": native_wet_area,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "reports": {
            "gained_band_n_cells": int(gained.sum()),
            "gained_band_ocean_area_share": float(
                c96_ocean_area.where(gained, 0.0).sum() / c96_ocean_area.sum()
            ),
        },
    }


def _om4_native_wet_area(config_path: str) -> float:
    from .om4 import config, run

    cfg = config.load_config(config_path)
    wetmask = run.load_wetmask(cfg)
    native = run.load_native_statics(cfg, wetmask)
    return float(native["areacello"].where(wetmask.isel({LEVEL_DIM: 0}), 0.0).sum())


def _randco2_native_wet_area(config_path: str, member: str | None) -> float:
    from .randco2 import config, run

    cfg = config.load_config(config_path, member)
    wet0 = run.load_wetmask(cfg)
    native = run.load_native_statics(cfg, wet0)
    return float(native["areacello"].where(wet0, 0.0).sum())


def native_wet_area(pipeline: str, config_path: str, member: str | None) -> float:
    """``Σ_i native_area_i wet_i0`` through the pipeline's own readers."""
    if pipeline == "om4":
        return _om4_native_wet_area(config_path)
    return _randco2_native_wet_area(config_path, member)


def _open_store(url: str, names: list[str]) -> xr.Dataset:
    ds = xr.open_zarr(make_zarr_store(url), decode_timedelta=False)
    return ds[names].load()


def run_stores(args: argparse.Namespace) -> dict:
    names = ["areacello", "sea_surface_fraction", *args.c2_fields, *args.c3_fields]
    c96 = _open_store(args.c96_store, names)
    direct = _open_store(args.direct_store, names)
    result = stores_checks(
        c96,
        direct,
        native_wet_area(args.pipeline, args.c96_config, args.member),
        args.c2_fields,
        args.c3_fields,
    )
    return {
        "inputs": {
            "pipeline": args.pipeline,
            "c96_config": args.c96_config,
            "member": args.member,
            "c96_store": args.c96_store,
            "direct_store": args.direct_store,
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
    sta = sub.add_parser("statics", help="static identities of a C96-path artifact")
    sta.add_argument("--target-grid", required=True, help="F90 or F22.5")
    sta.add_argument("--weights-url", required=True, help="C96-path artifact prefix")
    sta.add_argument(
        "--a-source-url",
        required=True,
        help="statics zarr store with areacello, wet, deptho, sftof",
    )
    sta.add_argument("--output-json", required=True)
    sto = sub.add_parser("stores", help="C1–C3 between a C96 and a direct store")
    sto.add_argument("--pipeline", required=True, choices=["om4", "randco2"])
    sto.add_argument(
        "--c96-config", required=True, help="config of the C96 run (native statics)"
    )
    sto.add_argument("--member", default=None, help="randco2 ensemble member")
    sto.add_argument("--c96-store", required=True)
    sto.add_argument("--direct-store", required=True)
    sto.add_argument("--c2-fields", nargs="*", default=[], help="per-ocean-area")
    sto.add_argument("--c3-fields", nargs="*", default=[], help="full-cell")
    sto.add_argument("--output-json", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = {
        "operators": run_operators,
        "statics": run_statics,
        "stores": run_stores,
    }[args.command](args)
    with fsspec.open(args.output_json, "w") as f:
        json.dump(result, f, indent=1)
    logger.info(f"Wrote {args.output_json}")
    if result.get("pass") is False:
        raise AssertionError(f"failed items; see {args.output_json}")


if __name__ == "__main__":
    main()
