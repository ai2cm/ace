"""Readers for the two operator sources of the all-CM4 C96 path.

c C96 cell, flattened ``(tile, y, x)``; i OM4 cell, flattened ``(yh, xh)``;
J Gaussian target cell, flattened ``(lat, lon)``; all 0-based here:

    xgrid_ocean_area_ci   CM4 xgrid area of C96 cell c ∩ OM4 cell i (m²)
    xgrid_land_area_c     CM4 xgrid land area in C96 cell c (m²)
    xgrid_c96_area_c    = Σ_i xgrid_ocean_area_ci + xgrid_land_area_c
    fregrid_overlap_Jc    ``fregrid`` conserve_order1 xgrid area of J ∩ c (m²)

Index maps, from the files' 1-based ``(col, row)`` pairs:

    c = (t − 1)·ny·nx + (row − 1)·nx + (col − 1)    t: C96 tile
    i = (row − 1)·n_xh + (col − 1)
    J = (row − 1)·n_lon + (col − 1)                 col = lon, row = lat

Every file and grid size comes from the files read: the xgrid files through
``grid_spec.nc`` ``aXo_file``, ``aXl_file``; tile numbers through each
file's ``contact`` and the atmosphere mosaic's ``gridtiles``; C96 and OM4
sizes from the mosaics' grid files; Gaussian sizes from
:func:`pipeline.grids.make_target_grid`.

Both sources are read at artifact build only; pipeline workers never read them.
"""

import dataclasses
import os
import tarfile
import tempfile

import numpy as np
import scipy.sparse
import xarray as xr

from .grids import make_target_grid
from .weights import _download

GRID_SPEC_FILENAME = "grid_spec.nc"


@dataclasses.dataclass
class CM4Xgrid:
    xgrid_ocean_area: scipy.sparse.csr_matrix  # (c, i), m²
    xgrid_land_area: np.ndarray  # (c,), m²
    supergrid_c96_area: np.ndarray  # (c,), m², 2 × 2 sum of the supergrid `area`
    c96_lon: np.ndarray  # (c,), degrees
    c96_lat: np.ndarray  # (c,), degrees
    c96_corner_lon: np.ndarray  # (tile, ny + 1, nx + 1), degrees
    c96_corner_lat: np.ndarray  # (tile, ny + 1, nx + 1), degrees
    om4_shape: tuple[int, int]  # (yh, xh)

    @property
    def c96_shape(self) -> tuple[int, int, int]:
        n_tiles, nyp, nxp = self.c96_corner_lon.shape
        return n_tiles, nyp - 1, nxp - 1

    @property
    def xgrid_c96_area(self) -> np.ndarray:
        return (
            np.asarray(self.xgrid_ocean_area.sum(axis=1)).ravel() + self.xgrid_land_area
        )

    @property
    def xgrid_om4_area(self) -> np.ndarray:
        return np.asarray(self.xgrid_ocean_area.sum(axis=0)).ravel()


def _text(value) -> str:
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()


def _string(da: xr.DataArray) -> str:
    return _text(np.asarray(da.values).item())


def _strings(da: xr.DataArray) -> list[str]:
    return [_text(v) for v in np.atleast_1d(da.values)]


def _mosaic_tiles(directory: str, mosaic_file: str) -> tuple[str, list[str], list[str]]:
    """(mosaic name, tile names, grid file paths) of a solo mosaic file."""
    with xr.open_dataset(os.path.join(directory, mosaic_file)) as m:
        files = [os.path.join(directory, f) for f in _strings(m["gridfiles"])]
        return _string(m["mosaic"]), _strings(m["gridtiles"]), files


def _contact_tile(contact: str, mosaic: str, tiles: list[str], path: str) -> int:
    """1-based tile of ``mosaic`` on the first side of an xgrid ``contact``."""
    name, tile = contact.split("::")[0].split(":")
    if name != mosaic or tile not in tiles:
        raise ValueError(
            f"{path}: contact {contact!r} does not name a tile of {mosaic} {tiles}"
        )
    return tiles.index(tile) + 1


def _check_range(index: np.ndarray, size: int, what: str, path: str) -> None:
    bad = (index < 1) | (index > size)
    if bad.any():
        raise ValueError(
            f"{path}: {int(bad.sum())} {what} outside 1..{size} "
            f"(range {index.min()}..{index.max()})"
        )


def _check_area(area: np.ndarray, path: str) -> None:
    if (area < 0).any():
        raise ValueError(f"{path}: {int((area < 0).sum())} xgrid_area < 0")


def _cells(
    cell: np.ndarray, shape: tuple[int, int], what: str, path: str
) -> np.ndarray:
    """Flat 0-based ``(row, col)`` positions of 1-based ``(col, row)`` pairs."""
    ny, nx = shape
    _check_range(cell[:, 0], nx, f"{what} col", path)
    _check_range(cell[:, 1], ny, f"{what} row", path)
    return (cell[:, 1] - 1) * nx + (cell[:, 0] - 1)


def _supergrid_shape(path: str) -> tuple[int, int]:
    """Cell shape ``(ny, nx)`` of the FMS supergrid at ``path``."""
    with xr.open_dataset(path) as g:
        ny2, nx2 = g.sizes["ny"], g.sizes["nx"]
    return ny2 // 2, nx2 // 2


def _read_c96_tiles(files: list[str]) -> dict:
    lon, lat, corner_lon, corner_lat, area = [], [], [], [], []
    for path in files:
        with xr.open_dataset(path) as g:
            x, y, a = (g[name].values.astype(np.float64) for name in ("x", "y", "area"))
        lon.append(x[1::2, 1::2].ravel())
        lat.append(y[1::2, 1::2].ravel())
        corner_lon.append(x[0::2, 0::2])
        corner_lat.append(y[0::2, 0::2])
        area.append(
            (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]).ravel()
        )
    shapes = {c.shape for c in corner_lon}
    if len(shapes) != 1:
        raise ValueError(f"C96 tiles differ in shape: {shapes}")
    return dict(
        c96_lon=np.concatenate(lon),
        c96_lat=np.concatenate(lat),
        c96_corner_lon=np.stack(corner_lon),
        c96_corner_lat=np.stack(corner_lat),
        supergrid_c96_area=np.concatenate(area),
    )


def read_cm4_xgrid_dir(directory: str) -> CM4Xgrid:
    """:func:`read_cm4_xgrid` on an extracted grid directory."""
    with xr.open_dataset(os.path.join(directory, GRID_SPEC_FILENAME)) as spec:
        atm_mosaic_file = _string(spec["atm_mosaic_file"])
        ocn_mosaic_file = _string(spec["ocn_mosaic_file"])
        aXo_files = _strings(spec["aXo_file"])
        aXl_files = _strings(spec["aXl_file"])
    atm_mosaic, atm_tiles, atm_files = _mosaic_tiles(directory, atm_mosaic_file)
    _, _, ocn_files = _mosaic_tiles(directory, ocn_mosaic_file)
    if len(ocn_files) != 1:
        raise ValueError(f"{ocn_mosaic_file}: {len(ocn_files)} ocean tiles, expected 1")
    c96 = _read_c96_tiles(atm_files)
    n_tiles, nyp, nxp = c96["c96_corner_lon"].shape
    tile_shape = (nyp - 1, nxp - 1)
    n_c = n_tiles * tile_shape[0] * tile_shape[1]
    om4_shape = _supergrid_shape(ocn_files[0])

    def xgrid_file(name: str) -> tuple[np.ndarray, xr.Dataset, str]:
        path = os.path.join(directory, name)
        ds = xr.open_dataset(path).load()
        t = _contact_tile(_string(ds["contact"]), atm_mosaic, atm_tiles, path)
        _check_range(np.asarray([t]), n_tiles, "tile", path)
        _check_area(ds["xgrid_area"].values, path)
        cell = _cells(ds["tile1_cell"].values, tile_shape, "C96 cell", path)
        return (t - 1) * tile_shape[0] * tile_shape[1] + cell, ds, path

    rows, cols, vals = [], [], []
    for name in aXo_files:
        c, ds, path = xgrid_file(name)
        rows.append(c)
        cols.append(_cells(ds["tile2_cell"].values, om4_shape, "OM4 cell", path))
        vals.append(ds["xgrid_area"].values.astype(np.float64))
    ocean = scipy.sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_c, om4_shape[0] * om4_shape[1]),
    ).tocsr()
    land = np.zeros(n_c)
    for name in aXl_files:
        c, ds, _ = xgrid_file(name)
        np.add.at(land, c, ds["xgrid_area"].values.astype(np.float64))
    return CM4Xgrid(
        xgrid_ocean_area=ocean, xgrid_land_area=land, om4_shape=om4_shape, **c96
    )


def read_cm4_xgrid(tar_url: str) -> CM4Xgrid:
    """CM4's exchange grid from the grid tar at ``tar_url``.

    Raises ``ValueError`` on a tile outside the atmosphere mosaic, a cell
    index outside its grid, or ``xgrid_area < 0``."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "grid.tar")
        _download(tar_url, tar_path)
        directory = os.path.join(tmpdir, "grid")
        with tarfile.open(tar_path) as tar:
            tar.extractall(directory, filter="data")
        os.remove(tar_path)
        return read_cm4_xgrid_dir(directory)


def read_fregrid_remap(
    remap_url: str, target_grid_name: str, c96_shape: tuple[int, int, int]
) -> scipy.sparse.csr_matrix:
    """``fregrid_overlap`` (J, c), m², of the ``fregrid`` remap file at
    ``remap_url``, shaped by ``make_target_grid(target_grid_name)`` and
    ``c96_shape`` ``(tile, ny, nx)`` (:attr:`CM4Xgrid.c96_shape`).

    Raises ``ValueError`` on a tile outside ``1..tile``, a cell index outside
    its grid, or ``xgrid_area < 0``."""
    target = make_target_grid(target_grid_name)
    gauss_shape = (target.sizes["lat"], target.sizes["lon"])
    n_tiles, ny, nx = c96_shape
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "remap.nc")
        _download(remap_url, path)
        with xr.open_dataset(path) as r:
            tile, c96_cell, gauss_cell, area = (
                r[name].values
                for name in ("tile1", "tile1_cell", "tile2_cell", "xgrid_area")
            )
    _check_range(tile, n_tiles, "tile", remap_url)
    _check_area(area, remap_url)
    c = (tile - 1) * ny * nx + _cells(c96_cell, (ny, nx), "C96 cell", remap_url)
    J = _cells(gauss_cell, gauss_shape, "Gaussian cell", remap_url)
    return scipy.sparse.coo_matrix(
        (area.astype(np.float64), (J, c)),
        shape=(gauss_shape[0] * gauss_shape[1], n_tiles * ny * nx),
    ).tocsr()
