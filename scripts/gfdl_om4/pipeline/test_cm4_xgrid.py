"""CM4 xgrid and ``fregrid`` readers and the ``operators`` gates, on toy files.

Toy atmosphere mosaic: two lat-lon tiles (lon 0–180°, 180–360°) of 15° cells.
Toy ocean: one lat-lon tile of 7.5° cells, so each ocean cell lies in one
atmosphere cell. Target: the real F22.5 grid. Overlaps are exact
lat-lon rectangle areas, so ``fregrid_gauss_area_J = band_area_J``.
"""

import os
import tarfile

import numpy as np
import pytest
import xarray as xr

for _name in ("xesmf", "esmpy", "apache_beam", "xarray_beam", "obstore"):
    pytest.importorskip(_name)

from .c96_checks import (
    EARTH_RADIUS_XGRID_M,
    check_band_area,
    check_fregrid_c96_area_positive,
    check_index_maps,
    check_wet_support,
    operator_checks,
    xgrid_index_max_deg,
)
from .cm4_xgrid import read_cm4_xgrid, read_fregrid_remap
from .grids import make_target_grid

TARGET = "F22.5"
ATM_TILE_LON = [(0.0, 180.0), (180.0, 360.0)]
ATM_N = 12  # cells per tile side
OCN_SHAPE = (24, 48)  # (lat, lon)
TAR_URL = "gs://toy/cm4-grid.tar"
A_SOURCE_URL = "gs://toy/ocean_static.zarr"


def _rect_area(lon0, lon1, lat0, lat1):
    """Area (m²) of lat-lon rectangles; 0 where they are empty."""
    dlon = np.clip(lon1 - lon0, 0, None)
    dsin = np.clip(np.sin(np.deg2rad(lat1)) - np.sin(np.deg2rad(lat0)), 0, None)
    return EARTH_RADIUS_XGRID_M**2 * np.deg2rad(dlon) * dsin


def _overlap(a, b):
    """Overlap areas of rectangle sets a, b (each ``(lon0, lon1, lat0, lat1)``
    arrays broadcast against each other)."""
    return _rect_area(
        np.maximum(a[0], b[0]),
        np.minimum(a[1], b[1]),
        np.maximum(a[2], b[2]),
        np.minimum(a[3], b[3]),
    )


def _cells(lon_b, lat_b):
    """Flat ``(row, col)`` cell rectangles and their 1-based ``(col, row)``."""
    row, col = np.meshgrid(
        np.arange(lat_b.size - 1), np.arange(lon_b.size - 1), indexing="ij"
    )
    row, col = row.ravel(), col.ravel()
    rect = (lon_b[col], lon_b[col + 1], lat_b[row], lat_b[row + 1])
    return rect, np.stack([col + 1, row + 1], axis=1).astype(np.int32)


def _supergrid(lon_b, lat_b) -> xr.Dataset:
    x, y = np.meshgrid(
        np.linspace(lon_b[0], lon_b[-1], 2 * (lon_b.size - 1) + 1),
        np.linspace(lat_b[0], lat_b[-1], 2 * (lat_b.size - 1) + 1),
    )
    area = _rect_area(x[:-1, :-1], x[:-1, 1:], y[:-1, :-1], y[1:, :-1])
    return xr.Dataset(
        {
            "x": (("nyp", "nxp"), x),
            "y": (("nyp", "nxp"), y),
            "area": (("ny", "nx"), area),
        }
    )


def _xgrid_file(contact, tile1_cell, tile2_cell, area) -> xr.Dataset:
    return xr.Dataset(
        {
            "contact": np.bytes_(contact),
            "tile1_cell": (("ncells", "two"), tile1_cell),
            "tile2_cell": (("ncells", "two"), tile2_cell),
            "xgrid_area": ("ncells", area),
        }
    )


def _mosaic(name, tiles, files) -> xr.Dataset:
    return xr.Dataset(
        {
            "mosaic": np.bytes_(name),
            "gridtiles": ("ntiles", np.array(tiles, dtype="S")),
            "gridfiles": ("ntiles", np.array(files, dtype="S")),
        }
    )


class Toy:
    """Toy grids and their exact operators."""

    def __init__(self):
        self.ocn_lon_b = np.linspace(0, 360, OCN_SHAPE[1] + 1)
        self.ocn_lat_b = np.linspace(-90, 90, OCN_SHAPE[0] + 1)
        ocn_rect, self.ocn_cell = _cells(self.ocn_lon_b, self.ocn_lat_b)
        lat_c = 0.5 * (ocn_rect[2] + ocn_rect[3])
        lon_c = 0.5 * (ocn_rect[0] + ocn_rect[1])
        self.wet = ((lat_c > -60) & ~((lon_c > 90) & (lon_c < 135))).astype(np.float64)
        self.native_area = _rect_area(*ocn_rect) * 1.001
        self.atm_lat_b = np.linspace(-90, 90, ATM_N + 1)
        self.atm_lon_b = [np.linspace(lo, hi, ATM_N + 1) for lo, hi in ATM_TILE_LON]
        n_c = len(ATM_TILE_LON) * ATM_N**2
        self.ocean = np.zeros((n_c, self.wet.size))
        self.land = np.zeros(n_c)
        target = make_target_grid(TARGET)
        g_rect, self.gauss_cell = _cells(target["lon_b"].values, target["lat_b"].values)
        self.fregrid = np.zeros((g_rect[0].size, n_c))
        self.atm_tiles = []
        for t, lon_b in enumerate(self.atm_lon_b):
            rect, cell = _cells(lon_b, self.atm_lat_b)
            c = t * ATM_N**2 + np.arange(cell.shape[0])
            ov = _overlap([r[:, None] for r in rect], ocn_rect) * self.wet
            self.ocean[c] = ov
            self.land[c] = _rect_area(*rect) - ov.sum(axis=1)
            self.fregrid[:, c] = _overlap(
                [r[None, :] for r in rect], [r[:, None] for r in g_rect]
            )
            self.atm_tiles.append(cell)

    def source_grid(self) -> xr.Dataset:
        lon_b, lat_b = np.meshgrid(self.ocn_lon_b, self.ocn_lat_b)
        lon = 0.5 * (lon_b[:-1, :-1] + lon_b[:-1, 1:])
        lat = 0.5 * (lat_b[:-1, :-1] + lat_b[1:, :-1])
        return xr.Dataset(
            {
                "lon": (("yh", "xh"), lon),
                "lat": (("yh", "xh"), lat),
                "lon_b": (("yh_b", "xh_b"), lon_b),
                "lat_b": (("yh_b", "xh_b"), lat_b),
            }
        )

    def xgrid_files(self) -> dict[str, xr.Dataset]:
        files = {}
        for t, cell in enumerate(self.atm_tiles):
            c = t * ATM_N**2 + np.arange(cell.shape[0])
            ci, i = np.nonzero(self.ocean[c])
            files[f"A_mosaic_tile{t + 1}Xocean_mosaic_tile1.nc"] = _xgrid_file(
                f"A_mosaic:tile{t + 1}::ocean_mosaic:tile1",
                cell[ci],
                self.ocn_cell[i],
                self.ocean[c][ci, i],
            )
            land = np.nonzero(self.land[c] > 0)[0]
            files[f"A_mosaic_tile{t + 1}XA_mosaic_tile{t + 1}.nc"] = _xgrid_file(
                f"A_mosaic:tile{t + 1}::A_mosaic:tile{t + 1}",
                cell[land],
                cell[land],
                self.land[c][land],
            )
        return files

    def write_tar(self, tmp_path, xgrid_files=None) -> str:
        d = tmp_path / "grid"
        d.mkdir()
        xgrid_files = xgrid_files or self.xgrid_files()
        names = list(xgrid_files)
        xr.Dataset(
            {
                "atm_mosaic_file": np.bytes_("A_mosaic.nc"),
                "ocn_mosaic_file": np.bytes_("ocean_mosaic.nc"),
                "aXo_file": (
                    "nfile_aXo",
                    np.array([n for n in names if "Xocean" in n], dtype="S"),
                ),
                "aXl_file": (
                    "nfile_aXl",
                    np.array([n for n in names if "XA_" in n], dtype="S"),
                ),
            }
        ).to_netcdf(d / "grid_spec.nc")
        tiles = [f"tile{t + 1}" for t in range(len(self.atm_lon_b))]
        _mosaic("A_mosaic", tiles, [f"A_grid.{t}.nc" for t in tiles]).to_netcdf(
            d / "A_mosaic.nc"
        )
        for t, lon_b in zip(tiles, self.atm_lon_b):
            _supergrid(lon_b, self.atm_lat_b).to_netcdf(d / f"A_grid.{t}.nc")
        _mosaic("ocean_mosaic", ["tile1"], ["ocean_hgrid.nc"]).to_netcdf(
            d / "ocean_mosaic.nc"
        )
        _supergrid(self.ocn_lon_b, self.ocn_lat_b).to_netcdf(d / "ocean_hgrid.nc")
        for name, ds in xgrid_files.items():
            ds.to_netcdf(d / name)
        path = str(tmp_path / "grid.tar")
        with tarfile.open(path, "w") as tar:
            for name in os.listdir(d):
                tar.add(d / name, arcname=name)
        return path

    def remap(self) -> xr.Dataset:
        J, c = np.nonzero(self.fregrid)
        tile, cell = np.divmod(c, ATM_N**2)
        return xr.Dataset(
            {
                "tile1": ("ncells", (tile + 1).astype(np.int32)),
                "tile1_cell": (("ncells", "two"), self.atm_tiles[0][cell]),
                "tile2_cell": (("ncells", "two"), self.gauss_cell[J]),
                "xgrid_area": ("ncells", self.fregrid[J, c]),
            }
        )

    def write_remap(self, tmp_path, remap=None) -> str:
        path = str(tmp_path / "remap.nc")
        (self.remap() if remap is None else remap).to_netcdf(path)
        return path


def _read(tmp_path, toy=None):
    toy = toy or Toy()
    xgrid = read_cm4_xgrid(toy.write_tar(tmp_path))
    overlap = read_fregrid_remap(toy.write_remap(tmp_path), TARGET, xgrid.c96_shape)
    return toy, xgrid, overlap


def _max_deg(toy, xgrid):
    return xgrid_index_max_deg(xgrid, toy.source_grid(), make_target_grid(TARGET))[
        "XGRID_INDEX_MAX_DEG"
    ]


def test_readers_match_toy_operators(tmp_path):
    toy, xgrid, overlap = _read(tmp_path)
    assert xgrid.c96_shape == (2, ATM_N, ATM_N)
    assert xgrid.om4_shape == OCN_SHAPE
    np.testing.assert_array_equal(xgrid.xgrid_ocean_area.toarray(), toy.ocean)
    np.testing.assert_allclose(xgrid.xgrid_land_area, toy.land, rtol=1e-12, atol=1e-3)
    np.testing.assert_allclose(
        xgrid.xgrid_c96_area, xgrid.supergrid_c96_area, rtol=1e-12
    )
    np.testing.assert_array_equal(overlap.toarray(), toy.fregrid)
    np.testing.assert_allclose(xgrid.c96_lon[:ATM_N], np.arange(ATM_N) * 15 + 7.5)
    np.testing.assert_allclose(
        xgrid.c96_lat[::ATM_N][:ATM_N], np.arange(ATM_N) * 15 - 82.5
    )


def test_operator_checks_pass_on_toy(tmp_path):
    toy, xgrid, overlap = _read(tmp_path)
    result = operator_checks(
        xgrid,
        overlap,
        toy.source_grid(),
        make_target_grid(TARGET),
        toy.native_area,
        toy.wet,
        TAR_URL,
        A_SOURCE_URL,
    )
    assert result["reports"]["sum_fregrid_gauss_area_over_sphere"] == pytest.approx(
        1, rel=1e-12
    )
    ratio = result["reports"]["xgrid_om4_area_over_native_area_wet"]
    assert ratio["q0"] == pytest.approx(1 / 1.001, rel=1e-12)


def _corrupt_xgrid(toy, tmp_path, name, **changes):
    files = toy.xgrid_files()
    ds = files[name]
    for var, f in changes.items():
        ds[var] = (ds[var].dims, f(ds[var].values.copy()))
    return toy.write_tar(tmp_path, files)


OCEAN_FILE = "A_mosaic_tile1Xocean_mosaic_tile1.nc"


def _set_first(value, column=None):
    def f(x):
        if column is None:
            x[0] = value
        else:
            x[0, column] = value
        return x

    return f


@pytest.mark.parametrize(
    "changes, match",
    [
        (
            {"contact": lambda _: np.bytes_("A_mosaic:tile7::ocean_mosaic:tile1")},
            "does not name a tile",
        ),
        ({"tile1_cell": _set_first(ATM_N + 1, 0)}, "C96 cell col outside"),
        ({"tile2_cell": _set_first(0, 1)}, "OM4 cell row outside"),
        ({"xgrid_area": _set_first(-1.0)}, "xgrid_area < 0"),
    ],
)
def test_read_cm4_xgrid_raises(tmp_path, changes, match):
    toy = Toy()
    with pytest.raises(ValueError, match=match):
        read_cm4_xgrid(_corrupt_xgrid(toy, tmp_path, OCEAN_FILE, **changes))


@pytest.mark.parametrize(
    "var, f, match",
    [
        ("tile1", _set_first(3), "tile outside 1..2"),
        ("tile1_cell", _set_first(0, 1), "C96 cell row outside"),
        ("tile2_cell", _set_first(91, 0), "Gaussian cell col outside"),
        ("xgrid_area", _set_first(-1.0), "xgrid_area < 0"),
    ],
)
def test_read_fregrid_remap_raises(tmp_path, var, f, match):
    toy = Toy()
    remap = toy.remap()
    remap[var] = (remap[var].dims, f(remap[var].values.copy()))
    with pytest.raises(ValueError, match=match):
        read_fregrid_remap(toy.write_remap(tmp_path, remap), TARGET, (2, ATM_N, ATM_N))


def test_xgrid_index_gate_fails_on_shifted_ocean_rows(tmp_path):
    toy = Toy()
    shift = OCN_SHAPE[0] // 2
    path = _corrupt_xgrid(
        toy,
        tmp_path,
        OCEAN_FILE,
        tile2_cell=lambda x: np.stack(
            [x[:, 0], (x[:, 1] - 1 + shift) % OCN_SHAPE[0] + 1], 1
        ),
    )
    xgrid = read_cm4_xgrid(path)
    overlap = read_fregrid_remap(toy.write_remap(tmp_path), TARGET, xgrid.c96_shape)
    with pytest.raises(AssertionError, match="xgrid index map"):
        check_index_maps(
            xgrid,
            overlap,
            toy.source_grid(),
            make_target_grid(TARGET),
            _max_deg(toy, xgrid),
        )


def test_fregrid_index_gate_fails_on_flipped_lat(tmp_path):
    toy = Toy()
    remap = toy.remap()
    n_lat = make_target_grid(TARGET).sizes["lat"]
    cell = remap["tile2_cell"].values.copy()
    cell[:, 1] = n_lat + 1 - cell[:, 1]
    remap["tile2_cell"] = (remap["tile2_cell"].dims, cell)
    xgrid = read_cm4_xgrid(toy.write_tar(tmp_path))
    overlap = read_fregrid_remap(
        toy.write_remap(tmp_path, remap), TARGET, xgrid.c96_shape
    )
    with pytest.raises(AssertionError, match="fregrid index map"):
        check_index_maps(
            xgrid,
            overlap,
            toy.source_grid(),
            make_target_grid(TARGET),
            _max_deg(toy, xgrid),
        )


def test_band_area_gate_fails_on_scaled_entry(tmp_path):
    _, _, overlap = _read(tmp_path)
    overlap.data[0] *= 1.01
    with pytest.raises(AssertionError, match="band_area_J"):
        check_band_area(overlap, make_target_grid(TARGET))


def test_fregrid_c96_area_gate_fails_on_empty_cell(tmp_path):
    _, _, overlap = _read(tmp_path)
    overlap = overlap.tolil()
    overlap[:, 5] = 0
    with pytest.raises(AssertionError, match="fregrid_c96_area_c > 0"):
        check_fregrid_c96_area_positive(overlap.tocsr())


def test_wet_support_gate_names_sources_and_count(tmp_path):
    toy, xgrid, _ = _read(tmp_path)
    wet = toy.wet.copy()
    wet[np.flatnonzero(wet)[:3]] = 0
    with pytest.raises(AssertionError, match="3 of") as err:
        check_wet_support(xgrid, wet, TAR_URL, A_SOURCE_URL)
    for text in (
        "CM4_XGRID_TAR_URL",
        TAR_URL,
        "NATIVE_AREA_A_SOURCE_URL",
        A_SOURCE_URL,
    ):
        assert text in str(err.value)
