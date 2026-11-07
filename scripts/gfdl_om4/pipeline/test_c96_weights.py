"""C96-path artifact build and load, on a toy of lat-lon rectangles.

O: 3 × 6 cells of 10° on lon 0–60, lat 0–30, dry on lon ≥ 30. C: two "tiles"
(lon 0–30, 30–60) of 2 × 2 cells of 15°; tile 2 is all land. G: 2 × 3 cells
of 20° × 15°; the J on lon 40–60 are land-only, those on lon 20–40 coastal.
Areas are exact spherical rectangles on the unit sphere.
"""

import dataclasses
import os
import re

import numpy as np
import pytest
import scipy.sparse
import xarray as xr

for _name in ("xesmf", "esmpy", "apache_beam", "xarray_beam", "obstore"):
    pytest.importorskip(_name)

import xesmf as xe

from . import c96_weights, weights
from .c96_checks import statics_checks
from .c96_weights import LF_A_TOL, NO_LAND_STATIC, land_fraction_gate
from .cm4_xgrid import CM4Xgrid
from .native_area import (
    F64_RTOL,
    LEVEL_DIM,
    NativeAreaStatics,
    assert_c96_path_weights,
    assert_close,
    c96_to_gauss_share,
    check_extensive_identities,
    check_static_identities,
    cm4_chain_native_area_weights,
    cm4_ocean_fraction,
    level_thickness,
    native_area_statics,
    regrid_full_cell,
)
from .ocean_emulators_port import regrid_normalized
from .om4.postprocess import sea_ice
from .postprocess import ChunkContext
from .weights import (
    REGRID_PATH_ATTR,
    REGRID_PATH_DESCRIPTION_ATTR,
    SOURCE_GRID_FILENAME,
    WEIGHTS_FILENAME,
)

O_LON_B, O_LAT_B = np.arange(0.0, 61.0, 10.0), np.arange(0.0, 31.0, 10.0)
C_TILE_LON_B = [np.array([0.0, 15.0, 30.0]), np.array([30.0, 45.0, 60.0])]
C_LAT_B = np.array([0.0, 15.0, 30.0])
G_LON_B, G_LAT_B = np.array([0.0, 20.0, 40.0, 60.0]), np.array([0.0, 15.0, 30.0])
COAST_LON = 30.0
TARGET = "toy"
URLS = {"xgrid": "gs://toy/xgrid.tar", "fregrid": "gs://toy/remap.nc"}
DEPTH = 100.0


def _cells(lon_b, lat_b) -> np.ndarray:
    """(lon0, lon1, lat0, lat1) per cell, flattened (lat, lon)."""
    return np.array(
        [
            (lon_b[x], lon_b[x + 1], lat_b[y], lat_b[y + 1])
            for y in range(len(lat_b) - 1)
            for x in range(len(lon_b) - 1)
        ]
    )


def _area(lon0, lon1, lat0, lat1):
    return np.deg2rad(lon1 - lon0) * (
        np.sin(np.deg2rad(lat1)) - np.sin(np.deg2rad(lat0))
    )


def _overlap(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    lon0 = np.maximum(rows[:, None, 0], cols[None, :, 0])
    lon1 = np.minimum(rows[:, None, 1], cols[None, :, 1])
    lat0 = np.maximum(rows[:, None, 2], cols[None, :, 2])
    lat1 = np.minimum(rows[:, None, 3], cols[None, :, 3])
    return np.where((lon1 > lon0) & (lat1 > lat0), _area(lon0, lon1, lat0, lat1), 0.0)


@dataclasses.dataclass
class Toy:
    X: np.ndarray  # xgrid_ocean_area (c, i)
    land: np.ndarray  # xgrid_land_area (c,)
    Y: np.ndarray  # fregrid_overlap (J, c)
    native_area: np.ndarray  # (i,)
    wet: np.ndarray  # (i,)

    def weights(self, **source_urls):
        return cm4_chain_native_area_weights(
            scipy.sparse.csr_matrix(self.X),
            self.land,
            scipy.sparse.csr_matrix(self.Y),
            self.native_area,
            self.wet,
            source_urls=URLS | {"native": "gs://toy/ocean_static.zarr"} | source_urls,
        )

    def fraction(self) -> np.ndarray:
        return cm4_ocean_fraction(
            scipy.sparse.csr_matrix(self.X), self.land, scipy.sparse.csr_matrix(self.Y)
        )

    def overlap(self) -> np.ndarray:
        """O_Ji by the § Task formula, dense."""
        F = self.Y / self.Y.sum(axis=0)
        a_tilde = self.X.sum(axis=0)
        ratio = np.divide(
            self.native_area, a_tilde, out=np.zeros_like(a_tilde), where=self.wet == 1
        )
        return F @ (self.X * ratio[None, :])


def _toy() -> Toy:
    o = _cells(O_LON_B, O_LAT_B)
    c = np.concatenate([_cells(lon_b, C_LAT_B) for lon_b in C_TILE_LON_B])
    g = _cells(G_LON_B, G_LAT_B)
    wet = (0.5 * (o[:, 0] + o[:, 1]) < COAST_LON).astype(np.float64)
    X = _overlap(c, o) * wet[None, :]
    # Exact 0 on the all-ocean tile, so f_J = 1 there exactly.
    land = np.where(c[:, 0] >= COAST_LON, _area(*c.T), 0.0)
    native_area = _area(*o.T) * (1.0 + 0.01 * (np.arange(len(o)) % 5))
    return Toy(X, land, _overlap(g, c), native_area, wet)


def _source_grid() -> xr.Dataset:
    return xe.util.grid_2d(0, 60, 10, 0, 30, 10)


def _target_grid() -> xr.Dataset:
    def centers(b):
        return 0.5 * (b[:-1] + b[1:])

    return xr.Dataset(
        coords={
            "lat": ("lat", centers(G_LAT_B)),
            "lon": ("lon", centers(G_LON_B)),
            "lat_b": ("lat_b", G_LAT_B),
            "lon_b": ("lon_b", G_LON_B),
        }
    )


def _on_source(values, name=None) -> xr.DataArray:
    shape = (len(O_LAT_B) - 1, len(O_LON_B) - 1)
    return xr.DataArray(
        np.asarray(values, dtype=np.float64).reshape(shape), dims=("y", "x"), name=name
    )


def _on_target(values) -> np.ndarray:
    return np.asarray(values).reshape(len(G_LAT_B) - 1, len(G_LON_B) - 1)


def _native_statics(toy: Toy) -> xr.Dataset:
    wet = toy.wet == 1
    return xr.Dataset(
        {
            "areacello": _on_source(toy.native_area),
            "wet": _on_source(toy.wet).astype(np.float32),
            "deptho": _on_source(np.where(wet, DEPTH, np.nan)),
            "sftof": _on_source(np.where(wet, 100.0, np.nan)),
        }
    )


def _cm4_xgrid(toy: Toy) -> CM4Xgrid:
    n_c = toy.X.shape[0]
    corners = np.zeros((2, 3, 3))
    return CM4Xgrid(
        xgrid_ocean_area=scipy.sparse.csr_matrix(toy.X),
        xgrid_land_area=toy.land,
        supergrid_c96_area=np.zeros(n_c),
        c96_lon=np.zeros(n_c),
        c96_lat=np.zeros(n_c),
        c96_corner_lon=corners,
        c96_corner_lat=corners,
        om4_shape=(len(O_LAT_B) - 1, len(O_LON_B) - 1),
    )


@dataclasses.dataclass
class Inputs:
    toy: Toy
    a_source_url: str
    source_grid_url: str
    land_static_url: str
    output_url: str


def _inputs(tmp_path, monkeypatch, toy=None, land_fraction=None) -> Inputs:
    """Write the toy's statics, source grid and land static; patch the
    readers and target grid. ``land_fraction`` defaults to 1 − f_J."""
    toy = _toy() if toy is None else toy
    target = _target_grid()
    a_source_url = str(tmp_path / "ocean_static.zarr")
    _native_statics(toy).to_zarr(a_source_url)
    source_grid_url = tmp_path / "direct-artifact"
    source_grid_url.mkdir()
    _source_grid().reset_coords().to_netcdf(source_grid_url / SOURCE_GRID_FILENAME)
    lf = 1.0 - toy.fraction() if land_fraction is None else land_fraction
    land_static_url = str(tmp_path / "land_static.zarr")
    xr.Dataset(
        {"land_fraction": (("lat", "lon"), _on_target(lf))},
        coords={"lat": target["lat"], "lon": target["lon"]},
    ).to_zarr(land_static_url)
    monkeypatch.setattr(c96_weights, "read_cm4_xgrid", lambda url: _cm4_xgrid(toy))
    monkeypatch.setattr(
        c96_weights,
        "read_fregrid_remap",
        lambda url, name, shape: scipy.sparse.csr_matrix(toy.Y),
    )
    monkeypatch.setattr(c96_weights, "make_target_grid", lambda name: target)
    _fresh_caches(monkeypatch, target)
    return Inputs(
        toy,
        a_source_url,
        str(source_grid_url),
        land_static_url,
        str(tmp_path / "c96-path-v1" / "artifact"),
    )


def _fresh_caches(monkeypatch, target):
    monkeypatch.setattr(weights, "make_target_grid", lambda name: target)
    monkeypatch.setattr(weights, "_REGRIDDER_CACHE", {})
    monkeypatch.setattr(weights, "_AREACELLO_CACHE", {})


def _argv(inputs: Inputs, land_static: bool = True) -> list[str]:
    argv = [
        "--cm4-xgrid-tar-url",
        URLS["xgrid"],
        "--fregrid-remap-url",
        URLS["fregrid"],
        "--target-grid",
        TARGET,
        "--a-source-url",
        inputs.a_source_url,
        "--source-grid-weights-url",
        inputs.source_grid_url,
        "--ace-commit",
        "0123abcd",
        "--output-url",
        inputs.output_url,
    ]
    if land_static:
        argv += ["--am4-land-static-url", inputs.land_static_url]
    return argv


def _built(tmp_path, monkeypatch, land_static: bool = True) -> Inputs:
    inputs = _inputs(tmp_path, monkeypatch)
    c96_weights.main(_argv(inputs, land_static))
    return inputs


def _attrs(url: str) -> dict:
    with xr.open_dataset(os.path.join(url, WEIGHTS_FILENAME)) as ds:
        return dict(ds.attrs)


def _surface(wet) -> xr.DataArray:
    return (_on_source(wet) == 1).expand_dims({LEVEL_DIM: [0.5 * DEPTH]})


def _toy_statics(regridder, url, toy: Toy) -> tuple[NativeAreaStatics, xr.DataArray]:
    native = _native_statics(toy)
    wetmask = _surface(toy.wet)
    dz = level_thickness(native["deptho"], np.array([0.0, DEPTH]))
    statics = native_area_statics(
        regridder, weights.open_areacello(url, TARGET), wetmask, dz, native["sftof"]
    )
    return statics, wetmask


# ---------------------------------------------------------------------------
# cm4_chain_native_area_weights
# ---------------------------------------------------------------------------


def test_c96_to_gauss_share_by_hand():
    share = c96_to_gauss_share(scipy.sparse.csr_matrix([[1.0, 3.0], [2.0, 1.0]]))
    expected = np.array([[1 / 3, 3 / 4], [2 / 3, 1 / 4]])
    assert_close(share.toarray(), expected, rtol=F64_RTOL, scale=1.0, context="F")


def test_toy_weights_match_formula():
    toy = _toy()
    result = toy.weights()
    overlap = toy.overlap()
    fraction = toy.fraction()
    np.testing.assert_allclose(_on_target(fraction), [[1, 0.5, 0], [1, 0.5, 0]])
    ocean_area = overlap.sum(axis=1)
    gauss_area = toy.Y.sum(axis=1)
    area = np.where(
        fraction > 0, ocean_area / np.where(fraction > 0, fraction, 1), gauss_area
    )
    scale = area.max()
    assert_close(result.area, area, rtol=F64_RTOL, scale=scale, context="A_J")
    assert_close(
        result.w_hat.toarray(),
        overlap / area[:, None],
        rtol=F64_RTOL,
        scale=1.0,
        context="ŵ = O / A",
    )
    assert_close(
        overlap.sum(axis=0),
        toy.native_area * toy.wet,
        rtol=F64_RTOL,
        scale=toy.native_area,
        context="Σ_J O_Ji = a_i m_i",
    )
    assert np.isnan(result.max_rel_residual)
    mapped = ocean_area > 0
    xgrid_wet_area = toy.Y / toy.Y.sum(axis=0) @ (toy.X @ toy.wet)
    rho = ocean_area[mapped] / xgrid_wet_area[mapped]
    rho_bar = (toy.native_area * toy.wet).sum() / toy.X.sum()
    assert result.a_over_a_tilde_spread == pytest.approx(
        np.max(np.abs(rho / rho_bar - 1)), rel=1e-12
    )
    assert result.a_over_a_tilde_spread > 0


def _with(toy: Toy, **changes) -> Toy:
    return dataclasses.replace(toy, **{k: np.array(v) for k, v in changes.items()})


def _empty_c96_cell(toy: Toy) -> Toy:
    """C96 cell 0 (all-ocean tile) with no xgrid area; OM4 cells it held
    alone become dry, so only f_J = 1 on land-free J can fail."""
    X = toy.X.copy()
    X[0] = 0.0
    return _with(toy, X=X, wet=(X.sum(axis=0) > 0).astype(np.float64))


def _corrupt(toy: Toy, name: str) -> Toy:
    if name == "wet_support":
        wet = toy.wet.copy()
        wet[np.flatnonzero(toy.wet == 0)[0]] = 1.0
        return _with(toy, wet=wet)
    if name == "share_sum":
        Y = toy.Y.copy()
        Y[:, 0] = 0.0
        return _with(toy, Y=Y)
    if name == "overlap_sum":
        a = toy.native_area.copy()
        a[np.flatnonzero(toy.wet == 1)[0]] = np.nan
        return _with(toy, native_area=a)
    if name == "fraction_bound":
        land = toy.land.copy()
        land[0] = -0.5 * toy.X[0].sum()
        return _with(toy, land=land)
    if name == "mask_equality":
        return _with(toy, native_area=np.zeros_like(toy.native_area))
    if name == "areacello_positive":
        Y = toy.Y.copy()
        Y[1] = 0.0  # a coastal J; its C96 cells keep other J
        return _with(toy, Y=Y)
    if name == "land_free_fraction":
        return _empty_c96_cell(toy)
    if name == "w_hat_negative":
        a = toy.native_area.copy()
        a[0] = -1e-6 * a[0]
        return _with(toy, native_area=a)
    raise KeyError(name)


@pytest.mark.parametrize(
    "name, match, urls",
    [
        ("wet_support", "xgrid_om4_area_i > 0", ("xgrid", "native")),
        ("share_sum", "c96_to_gauss_share", ("fregrid",)),
        ("overlap_sum", "Σ_J overlap_c96path_Ji = native_area_i wet_i", ()),
        ("fraction_bound", "upper bound", ("xgrid", "fregrid")),
        ("mask_equality", "ocean_area_c96path_J0 > 0", ("xgrid", "fregrid", "native")),
        ("areacello_positive", "areacello_c96path_J > 0", ("fregrid",)),
        ("land_free_fraction", "cm4_ocean_fraction_J = 1", ("xgrid", "fregrid")),
        ("w_hat_negative", "entries of ŵ are < 0", ()),
    ],
)
def test_build_asserts_raise(name, match, urls):
    toy = _toy()
    toy.weights()
    with pytest.raises(AssertionError, match=re.escape(match)) as info:
        _corrupt(toy, name).weights()
    all_urls = URLS | {"native": "gs://toy/ocean_static.zarr"}
    for key in urls:
        assert all_urls[key] in str(info.value), key


def test_w_hat_support_assert_raises():
    toy = _toy()
    result = toy.weights()
    overlap = scipy.sparse.csr_matrix(toy.overlap())
    assert_c96_path_weights(result.w_hat, overlap)
    w_hat = result.w_hat.tolil()
    w_hat[0, w_hat[0].nonzero()[1][0]] = 0.0
    w_hat = w_hat.tocsr()
    w_hat.eliminate_zeros()
    with pytest.raises(AssertionError, match="support"):
        assert_c96_path_weights(w_hat, overlap)


def test_weights_raise_on_shape_mismatch():
    toy = _toy()
    with pytest.raises(ValueError, match="OM4 cells"):
        _with(toy, wet=toy.wet[:-1]).weights()
    with pytest.raises(ValueError, match="C96 cells"):
        _with(toy, Y=toy.Y[:, :-1]).weights()


# ---------------------------------------------------------------------------
# c96_weights: build, gate, attrs
# ---------------------------------------------------------------------------


def test_built_artifact_attrs(tmp_path, monkeypatch):
    inputs = _built(tmp_path, monkeypatch)
    attrs = _attrs(inputs.output_url)
    assert attrs[REGRID_PATH_ATTR] == "c96"
    assert attrs[REGRID_PATH_DESCRIPTION_ATTR] == (
        f"OM4 0.25° tripolar -> C96 (CM4 xgrid) -> {TARGET} (fregrid conserve_order1)"
    )
    assert attrs["source_weights_url"] == URLS["xgrid"]
    assert attrs["fregrid_remap_url"] == URLS["fregrid"]
    assert attrs["a_source_url"] == inputs.a_source_url
    assert np.isnan(attrs["esmpy_area_rtol"])
    assert np.isnan(attrs["esmpy_max_rel_residual"])
    assert attrs["a_over_a_tilde_spread"] == inputs.toy.weights().a_over_a_tilde_spread
    assert attrs["wet_columns_only"] == 1
    assert attrs["am4_land_static_url"] == inputs.land_static_url
    assert attrs["q6_option"] == "a"
    assert attrs["max_abs_cm4_ocean_fraction_minus_one_minus_lf_A"] == 0.0
    assert attrs["rms_cm4_ocean_fraction_minus_one_minus_lf_A"] == 0.0
    assert attrs["ace_commit"] == "0123abcd"
    assert attrs["source_grid_url"] == inputs.source_grid_url
    assert "build_time_utc" not in attrs
    history = attrs["history"]
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00: "
        r"python -m pipeline\.c96_weights ",
        history,
    ), history
    for arg in _argv(inputs):
        assert arg in history, arg
    xr.testing.assert_identical(
        weights.open_source_grid(inputs.output_url),
        weights.open_source_grid(inputs.source_grid_url),
    )


def test_build_without_land_static(tmp_path, monkeypatch):
    inputs = _built(tmp_path, monkeypatch, land_static=False)
    attrs = _attrs(inputs.output_url)
    assert attrs["am4_land_static_url"] == NO_LAND_STATIC
    assert np.isnan(attrs["max_abs_cm4_ocean_fraction_minus_one_minus_lf_A"])
    assert np.isnan(attrs["rms_cm4_ocean_fraction_minus_one_minus_lf_A"])
    assert attrs[REGRID_PATH_ATTR] == "c96"


def test_land_fraction_gate_raises(tmp_path, monkeypatch):
    toy = _toy()
    lf = 1.0 - toy.fraction()
    lf[1] += 2 * LF_A_TOL
    inputs = _inputs(tmp_path, monkeypatch, toy, land_fraction=lf)
    with pytest.raises(AssertionError, match="LF_A_TOL") as info:
        c96_weights.main(_argv(inputs))
    for url in (URLS["xgrid"], URLS["fregrid"], inputs.land_static_url):
        assert url in str(info.value)
    assert not os.path.exists(inputs.output_url)


def test_land_fraction_gate_within_tol():
    fraction = np.array([1.0, 0.5, 0.0])
    gate = land_fraction_gate(
        fraction, 1.0 - fraction + 0.5 * LF_A_TOL, ("x", "f", "l")
    )
    assert gate["max_abs_cm4_ocean_fraction_minus_one_minus_lf_A"] == pytest.approx(
        0.5 * LF_A_TOL
    )


def test_build_raises_on_land_static_off_grid(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, monkeypatch)
    target = _target_grid().isel(lat=slice(0, 1))
    monkeypatch.setattr(c96_weights, "make_target_grid", lambda name: target)
    monkeypatch.setattr(
        c96_weights,
        "read_fregrid_remap",
        lambda url, name, shape: scipy.sparse.csr_matrix(inputs.toy.Y),
    )
    with pytest.raises(ValueError, match="lat"):
        c96_weights._land_fraction_on_target(
            xr.open_zarr(inputs.land_static_url)["land_fraction"],
            target,
            inputs.land_static_url,
        )


def test_build_raises_on_remap_not_covering_target(tmp_path, monkeypatch):
    toy = _toy()
    inputs = _inputs(tmp_path, monkeypatch, toy)
    Y = toy.Y.copy()
    Y[-1] = 0.0
    monkeypatch.setattr(
        c96_weights,
        "read_fregrid_remap",
        lambda url, name, shape: scipy.sparse.csr_matrix(Y),
    )
    with pytest.raises(ValueError, match="uncovered"):
        c96_weights.main(_argv(inputs))


def test_build_raises_on_source_grid_shape(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, monkeypatch)
    xgrid = dataclasses.replace(_cm4_xgrid(inputs.toy), om4_shape=(6, 3))
    monkeypatch.setattr(c96_weights, "read_cm4_xgrid", lambda url: xgrid)
    with pytest.raises(ValueError, match="source_grid.nc"):
        c96_weights.main(_argv(inputs))


def test_build_refuses_existing_prefix(tmp_path, monkeypatch):
    inputs = _built(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError, match="C96_PATH_WEIGHTS_VERSION"):
        c96_weights.main(_argv(inputs))


# ---------------------------------------------------------------------------
# Loaded artifact: round trip, statics, sea ice
# ---------------------------------------------------------------------------


def test_round_trip(tmp_path, monkeypatch):
    inputs = _built(tmp_path, monkeypatch)
    result = inputs.toy.weights()
    regridder = weights.get_regridder(inputs.output_url, TARGET)
    area = weights.open_areacello(inputs.output_url, TARGET)
    assert weights.read_wet_columns_only(inputs.output_url) is True
    assert_close(
        area.values.ravel(),
        result.area,
        rtol=F64_RTOL,
        scale=result.area,
        context="A_J",
    )
    loaded = regridder.weights.data.to_scipy_sparse().toarray()
    assert_close(
        np.nan_to_num(loaded),
        result.w_hat.toarray(),
        rtol=F64_RTOL,
        scale=1.0,
        context="loaded ŵ",
    )


def test_read_wet_columns_only_absent(tmp_path):
    url = tmp_path / "direct"
    url.mkdir()
    xr.Dataset({"S": ("n_s", [1.0])}).to_netcdf(url / WEIGHTS_FILENAME)
    assert weights.read_wet_columns_only(str(url)) is False


def test_statics_on_land_only_cells(tmp_path, monkeypatch):
    inputs = _built(tmp_path, monkeypatch)
    toy = inputs.toy
    regridder = weights.get_regridder(inputs.output_url, TARGET)
    statics, _ = _toy_statics(regridder, inputs.output_url, toy)
    stored = statics.stored
    for name in stored.data_vars:
        assert np.isfinite(stored[name]).all(), name
    fraction = toy.fraction()
    land_only = fraction == 0
    assert land_only.any()
    assert (stored["areacello"].values.ravel()[land_only] > 0).all()
    field = regridder(_on_source(np.ones(toy.wet.size)))
    assert np.isnan(field.values.ravel()[land_only]).all()
    assert np.isfinite(field.values.ravel()[~land_only]).all()

    ssf = stored["sea_surface_fraction"].values.ravel()
    assert_close(ssf, fraction, rtol=F64_RTOL, scale=1.0, context="ssf = f_J")
    ocean_area = toy.overlap().sum(axis=1)
    assert_close(
        stored["areacello"].values.ravel() * ssf,
        ocean_area,
        rtol=F64_RTOL,
        scale=ocean_area.max(),
        context="areacello · ssf = ocean_area_c96path_J0",
    )


def test_static_identities_wet_columns_only(tmp_path, monkeypatch):
    inputs = _built(tmp_path, monkeypatch)
    toy = inputs.toy
    url = inputs.output_url
    regridder = weights.get_regridder(url, TARGET)
    area = weights.open_areacello(url, TARGET)
    urls = (inputs.a_source_url, "gs://toy/wetmask.zarr")
    result = statics_checks(
        regridder,
        area,
        _native_statics(toy),
        wet_columns_only=True,
        mask_source_urls=urls,
    )
    assert result["gates"]["check_static_identities"]

    statics, wetmask = _toy_statics(regridder, url, toy)
    native_area = _on_source(toy.native_area)
    with pytest.raises(AssertionError):
        check_static_identities(statics, regridder, wetmask, native_area)

    run_wet = toy.wet.copy()
    run_wet[np.flatnonzero(toy.wet == 0)[0]] = 1.0
    with pytest.raises(AssertionError, match="on 1 native cells") as info:
        check_static_identities(
            statics,
            regridder,
            _surface(run_wet),
            native_area,
            wet_columns_only=True,
            mask_source_urls=urls,
        )
    for u in urls:
        assert u in str(info.value)

    with pytest.raises(ValueError, match="mask_source_urls"):
        check_static_identities(
            statics, regridder, wetmask, native_area, wet_columns_only=True
        )


def test_sea_ice_volume_is_native(tmp_path, monkeypatch):
    inputs = _built(tmp_path, monkeypatch)
    toy = inputs.toy
    regridder = weights.get_regridder(inputs.output_url, TARGET)
    area = weights.open_areacello(inputs.output_url, TARGET)
    wet = _on_source(toy.wet) == 1
    n = toy.wet.size
    thickness = _on_source(np.where(toy.wet == 1, 1.0 + np.arange(n) / n, np.nan))
    concentration = _on_source(
        np.where(toy.wet == 1, 0.2 + 0.6 * (np.arange(n) % 3) / 2, np.nan)
    )
    surface = xr.Dataset({"sea_ice_fraction": concentration, "HI": thickness})
    per_ocean_area, ocean_fraction = regrid_normalized(surface, regridder, wet)
    product = (thickness * concentration).fillna(0.0)
    output = xr.Dataset(
        {
            "sea_ice_fraction": regrid_full_cell(
                regridder, concentration, wet, ocean_fraction
            ),
            "ocean_sea_ice_fraction": per_ocean_area["sea_ice_fraction"],
            "HI": per_ocean_area["HI"],
            "UI": 0.0 * per_ocean_area["HI"],
            "VI": 0.0 * per_ocean_area["HI"],
        }
    )
    context = ChunkContext(
        ocean_fraction=ocean_fraction,
        store="toy",
        areacello=area,
        ice_volume_per_cell_area=regrid_full_cell(
            regridder, product, wet, ocean_fraction
        ),
    )
    out = sea_ice(
        output,
        context,
        sea_ice_fraction="sea_ice_fraction",
        ocean_sea_ice_fraction="ocean_sea_ice_fraction",
    )
    volume = out["sea_ice_volume"].values.ravel()
    expected = toy.overlap() @ product.values.ravel()
    ocean = toy.fraction() > 0
    assert_close(
        volume[ocean],
        expected[ocean],
        rtol=F64_RTOL,
        scale=expected[ocean],
        context="sea_ice_volume_J = Σ_i O_Ji HI_i c_i",
    )
    assert np.isnan(volume[~ocean]).all()
    check_extensive_identities(
        out,
        {"sea_ice_volume": product},
        wet,
        area,
        ocean_fraction,
        _on_source(toy.native_area),
        "toy",
    )
