"""om4 on the C96-path toy of ``pipeline/test_c96_weights.py`` with the coast
at lon 20°, and the store root ``regrid_path`` of C96 and direct artifacts.

i: toy O cell; J: toy G cell; k: level. With the coast at lon 20° the G cells
on lon 20–40 hold no wet O cell, so their direct-path ocean area is 0, but
the C cell on lon 15–30 carries ocean into them: they are ``gained_band``.
The G cells on lon 40–60 are land-only. Levels: interfaces 0, 50, 100 m;
every wet cell is 100 m deep but one, 40 m.
"""

import dataclasses
import os
import sys

import numpy as np
import pytest
import xarray as xr
import yaml

for _name in ("xesmf", "esmpy", "apache_beam", "xarray_beam", "obstore"):
    pytest.importorskip(_name)

import xarray_beam as xbeam

from .. import c96_weights
from ..config_io import OutputConfig, StaticsConfig, WetmaskConfig
from ..native_area import AREA_RTOL, LEVEL_DIM, assert_close
from ..native_area_weights import regrid_path_description as direct_description
from ..postprocess import DERIVATION_ATTR
from ..test_c96_weights import (
    G_LAT_B,
    G_LON_B,
    O_DIMS,
    O_LAT_B,
    O_LON_B,
    TARGET,
    Toy,
    _argv,
    _cells,
    _inputs,
    _on_source,
    _overlap,
    _toy,
)
from ..test_native_area import _toy_grid, toy_artifact_built
from ..weights import (
    C96_PATH_AREACELLO_DERIVATION,
    C96_PATH_AREACELLO_LONG_NAME,
    C96_PATH_SFTOF_DERIVATION,
    REGRID_PATH_ATTR,
    REGRID_PATH_C96,
    REGRID_PATH_DESCRIPTION_ATTR,
    REGRID_PATH_DIRECT,
    WEIGHTS_FILENAME,
)
from ..zarr_io import TIME_DIM
from . import run
from .config import PipelineConfig, StreamConfig
from .test_native_area_run import DIRECT_FLAGS, _toy_run_config

COAST_LON = 20.0
LEVEL_CENTERS = np.array([25.0, 75.0])
INTERFACES = np.array([0.0, 50.0, 100.0])
SHALLOW_CELL = (2, 0)  # (yh, xh): O cell on lat 20–30, lon 0–10
O_SHAPE = (len(O_LAT_B) - 1, len(O_LON_B) - 1)
G_SHAPE = (len(G_LAT_B) - 1, len(G_LON_B) - 1)


def _deptho(toy: Toy) -> np.ndarray:
    H = np.where(toy.wet == 1, 100.0, np.nan).reshape(O_SHAPE)
    H[SHALLOW_CELL] = 40.0
    return H.ravel()


def _wet_ik(toy: Toy) -> np.ndarray:
    """wet_ik, (k, i)."""
    return np.nan_to_num(_deptho(toy))[None, :] > INTERFACES[:-1, None]


def _wetmask(toy: Toy) -> xr.DataArray:
    return xr.DataArray(
        _wet_ik(toy).reshape(len(LEVEL_CENTERS), *O_SHAPE),
        dims=(LEVEL_DIM, *O_DIMS),
        coords={LEVEL_DIM: LEVEL_CENTERS},
    )


@dataclasses.dataclass
class Expected:
    """By the formulas of issue 02 § Task, from the toy's geometry."""

    area: np.ndarray  # areacello_c96path_J
    ocean_area: np.ndarray  # ocean_area_c96path_J0
    ocean_volume: np.ndarray  # ocean_volume_c96path_Jk, (J, k)
    gained_band: np.ndarray  # bool (J,)
    land_only: np.ndarray  # bool (J,)


def _expected(toy: Toy) -> Expected:
    overlap = toy.overlap()
    wet = _wet_ik(toy)
    dz = (
        np.clip(
            np.nan_to_num(_deptho(toy))[None, :],
            INTERFACES[:-1, None],
            INTERFACES[1:, None],
        )
        - INTERFACES[:-1, None]
    )
    ocean_area = overlap @ wet[0]
    # Direct path: exact conservative overlap of the same O and G rectangles.
    ocean_area_direct = (
        _overlap(_cells(G_LON_B, G_LAT_B), _cells(O_LON_B, O_LAT_B)) @ wet[0]
    )
    return Expected(
        area=toy.weights().area,
        ocean_area=ocean_area,
        ocean_volume=overlap @ (wet * dz).T,
        gained_band=(ocean_area_direct == 0) & (ocean_area > 0),
        land_only=ocean_area == 0,
    )


@dataclasses.dataclass
class C96Run:
    toy: Toy
    config: PipelineConfig
    a_source_url: str


def _c96_run(tmp_path, monkeypatch, description: bool = True) -> C96Run:
    """The toy C96 artifact built by ``python -m pipeline.c96_weights``, a
    run statics store and a config naming them; no stream yet."""
    toy = _toy(COAST_LON)
    inputs = _inputs(tmp_path, monkeypatch, toy)
    c96_weights.main(_argv(inputs))
    if not description:
        path = os.path.join(inputs.output_url, WEIGHTS_FILENAME)
        ds = xr.open_dataset(path).load()
        ds.close()
        del ds.attrs[REGRID_PATH_DESCRIPTION_ATTR]
        ds.to_netcdf(path)
    statics_store = str(tmp_path / "run_static.zarr")
    xr.Dataset(
        {
            "areacello": _on_source(toy.native_area),
            "sftof": _on_source(np.where(toy.wet == 1, 100.0, np.nan)),
            "deptho": _on_source(_deptho(toy)),
        }
    ).to_zarr(statics_store)
    config = PipelineConfig(
        streams=[],
        statics=StaticsConfig(store=statics_store, variables=["deptho"]),
        wetmask=WetmaskConfig(store="unused", variable="thetao"),
        target_grid=TARGET,
        weights_url=inputs.output_url,
        output=OutputConfig(path=str(tmp_path / "out.zarr")),
        expected_level_count=len(LEVEL_CENTERS),
    )
    return C96Run(toy, config, inputs.a_source_url)


def _on_g(da: xr.DataArray) -> np.ndarray:
    """Values of one G field, flattened (lat, lon)."""
    return np.asarray(da, dtype=np.float64).reshape(-1, int(np.prod(G_SHAPE)))[-1]


def _assert_g(da, expected, name):
    lhs = _on_g(da)
    assert_close(
        lhs, expected, rtol=AREA_RTOL, scale=np.abs(expected).max(), context=name
    )


def test_gained_band_and_land_only_exist():
    e = _expected(_toy(COAST_LON))
    assert e.gained_band.any()
    assert e.land_only.any()
    assert not (e.gained_band & e.land_only).any()


def test_build_statics_c96_path(tmp_path, monkeypatch):
    """areacello = areacello_c96path, sea_surface_fraction = W'_J0 / A_J,
    mask_k = 1[V'_Jk > 0]; check_static_identities runs inside."""
    c96 = _c96_run(tmp_path, monkeypatch)
    e = _expected(c96.toy)
    statics = run.build_statics(c96.config, _wetmask(c96.toy))

    _assert_g(statics["areacello"], e.area, "areacello = areacello_c96path")
    assert (_on_g(statics["areacello"]) > 0).all()
    _assert_g(
        statics["sea_surface_fraction"],
        e.ocean_area / e.area,
        "sea_surface_fraction = ocean_area_c96path_J0 / areacello_c96path_J",
    )
    _assert_g(statics["sftof"], 100 * e.ocean_area / e.area, "sftof")
    for k in range(len(LEVEL_CENTERS)):
        np.testing.assert_array_equal(
            _on_g(statics[f"mask_{k}"]), (e.ocean_volume[:, k] > 0).astype(float)
        )
    assert (e.ocean_volume[:, 1] < e.ocean_volume[:, 0]).any(), "shallow cell"
    for name, da in statics.data_vars.items():
        if da.ndim:
            assert np.isfinite(_on_g(da)[e.gained_band]).all(), name


def test_build_statics_attrs_by_path(tmp_path, monkeypatch):
    """C96-path strings on the composed artifact, #1594's on a direct one."""
    c96 = _c96_run(tmp_path, monkeypatch)
    statics = run.build_statics(c96.config, _wetmask(c96.toy))
    assert statics["areacello"].attrs["long_name"] == C96_PATH_AREACELLO_LONG_NAME
    assert statics["areacello"].attrs[DERIVATION_ATTR] == C96_PATH_AREACELLO_DERIVATION
    assert statics["sftof"].attrs[DERIVATION_ATTR] == C96_PATH_SFTOF_DERIVATION

    (tmp_path / "direct").mkdir()
    _, direct, *_ = _toy_run_config(tmp_path / "direct", monkeypatch)
    statics = run.build_statics(direct, run.load_wetmask(direct))
    assert (
        statics["areacello"].attrs["long_name"]
        == "cell area covered by native ocean-model cells"
    )
    assert (
        "over native cells wet and dry" in (statics["areacello"].attrs[DERIVATION_ATTR])
    )
    assert "0 on land and unmapped cells" in statics["sftof"].attrs[DERIVATION_ATTR]


def _field(toy: Toy, scale: float, level: int | None = None) -> np.ndarray:
    """Distinct values on the wet cells (of ``level``), NaN elsewhere."""
    wet = _wet_ik(toy)[0 if level is None else level]
    values = scale * (1.0 + 0.1 * np.arange(wet.size))
    return np.where(wet, values, np.nan).reshape(O_SHAPE)


def _chunk(c96: C96Run, stream: StreamConfig, fields: dict, level=None):
    """process_chunk on one timestep; ``fields`` maps names to (dims, values)."""
    ds = xr.Dataset(
        {
            name: (dims, np.asarray(values, dtype=np.float32))
            for name, (dims, values) in fields.items()
        }
    ).expand_dims(TIME_DIM)
    staggered = {d: np.arange(ds.sizes[d]) for d in ("xq", "yq") if d in ds.dims}
    ds = ds.assign_coords(staggered)
    offsets = {TIME_DIM: 0}
    if level is not None:
        ds = ds.expand_dims({LEVEL_DIM: LEVEL_CENTERS[[level]]}, axis=1)
        offsets[LEVEL_DIM] = level
    _, out = run.process_chunk(
        xbeam.Key(offsets),
        ds,
        stream,
        _wetmask(c96.toy),
        c96.config.weights_url,
        TARGET,
        _on_source(c96.toy.native_area),
        _on_source(_deptho(c96.toy)),
    )
    return out


def _assert_finite_where_ocean(out: xr.Dataset, ocean: np.ndarray, gained_band):
    """Finite exactly on ocean J, gained_band included (no NaN mass loss)."""
    assert ocean[gained_band].all()
    for name, da in out.data_vars.items():
        np.testing.assert_array_equal(np.isfinite(_on_g(da)), ocean, err_msg=name)


def test_process_chunk_c96_path(tmp_path, monkeypatch):
    """2-D, 3-D, rotated-pair and full-cell variables pass
    check_field_identities, sea_ice passes check_extensive_identities (both
    inside process_chunk); every output is finite on gained_band J."""
    c96 = _c96_run(tmp_path, monkeypatch)
    toy, e = c96.toy, _expected(c96.toy)
    ocean_0 = e.ocean_area > 0

    ocean = StreamConfig(name="ocean", store="toy", variables=["SSH", "thetao"])
    out = _chunk(c96, ocean, {"SSH": (O_DIMS, _field(toy, 0.3))})
    _assert_finite_where_ocean(out, ocean_0, e.gained_band)
    for k in range(len(LEVEL_CENTERS)):
        out = _chunk(c96, ocean, {"thetao": (O_DIMS, _field(toy, 10.0, k))}, level=k)
        assert set(out.data_vars) == {f"thetao_{k}"}
        _assert_finite_where_ocean(out, e.ocean_volume[:, k] > 0, e.gained_band)

    face = 1.0 + 0.1 * np.arange(toy.wet.size).reshape(O_SHAPE)
    flux = StreamConfig(
        name="flux",
        store="toy",
        variables=["hfds", "tauuo", "tauvo"],
        rotated_pairs=[["tauuo", "tauvo"]],
        postprocess=["hfds_total_area"],
    )
    out = _chunk(
        c96,
        flux,
        {
            "hfds": (O_DIMS, _field(toy, 5.0)),
            "tauuo": (("yh", "xq"), face),
            "tauvo": (("yq", "xh"), -0.5 * face),
        },
    )
    assert {"hfds", "hfds_total_area", "tauuo", "tauvo"} <= set(out.data_vars)
    _assert_finite_where_ocean(out, ocean_0, e.gained_band)

    ice = StreamConfig(
        name="ice",
        store="toy",
        variables=["sea_ice_fraction", "HI", "UI", "VI"],
        renaming={"sea_ice_fraction": "ocean_sea_ice_fraction"},
        full_cell_variables=["sea_ice_fraction"],
        postprocess=["sea_ice"],
    )
    concentration = np.clip(_field(toy, 0.5), 0.0, 1.0)
    out = _chunk(
        c96,
        ice,
        {
            "sea_ice_fraction": (O_DIMS, concentration),
            "HI": (O_DIMS, _field(toy, 2.0)),
            "UI": (O_DIMS, _field(toy, 0.0)),
            "VI": (O_DIMS, _field(toy, 0.0)),
        },
    )
    assert "sea_ice_volume" in out.data_vars
    _assert_finite_where_ocean(out, ocean_0, e.gained_band)


def _write_streams(tmp_path, c96: C96Run) -> str:
    """Two timesteps of an ocean stream (SSH, thetao, so) on the toy, the
    wetmask from thetao; returns the config path."""
    toy, config = c96.toy, c96.config
    time = xr.DataArray(
        np.array(["2000-01-01", "2000-01-06"], dtype="datetime64[ns]"), dims=TIME_DIM
    )
    dims_3d = (TIME_DIM, LEVEL_DIM, *O_DIMS)
    levels = [_field(toy, 1.0, k) for k in range(len(LEVEL_CENTERS))]
    ocean = xr.Dataset(
        {
            "SSH": ((TIME_DIM, *O_DIMS), np.stack([_field(toy, 0.3)] * 2)),
            "thetao": (dims_3d, np.stack([np.stack(levels) * 10.0] * 2)),
            "so": (dims_3d, np.stack([np.stack(levels) + 34.0] * 2)),
        },
        coords={TIME_DIM: time, LEVEL_DIM: LEVEL_CENTERS},
    ).astype(np.float32)
    store = str(tmp_path / "ocean.zarr")
    ocean.chunk({TIME_DIM: 1}).to_zarr(store)
    config.streams = [
        StreamConfig(name="ocean", store=store, variables=["SSH", "thetao", "so"])
    ]
    config.wetmask = WetmaskConfig(store=store, variable="thetao")
    config.output.time_shard_size = 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dataclasses.asdict(config)))
    return str(path)


def _run_main(monkeypatch, config_path):
    monkeypatch.setattr(sys, "argv", ["run", "--config", config_path, *DIRECT_FLAGS])
    run.main()


@pytest.mark.parametrize("description", [True, False])
def test_main_store_regrid_path_c96(tmp_path, monkeypatch, description):
    """Store root regrid_path = "c96" from the artifact alone (no config
    field); regrid_path_description copied when the artifact has one."""
    c96 = _c96_run(tmp_path, monkeypatch, description=description)
    _run_main(monkeypatch, _write_streams(tmp_path, c96))
    store = xr.open_zarr(c96.config.output.path)
    assert store.attrs[REGRID_PATH_ATTR] == REGRID_PATH_C96
    if description:
        assert store.attrs[REGRID_PATH_DESCRIPTION_ATTR] == (
            c96_weights.regrid_path_description(TARGET)
        )
    else:
        assert REGRID_PATH_DESCRIPTION_ATTR not in store.attrs
    assert store["areacello"].attrs["long_name"] == C96_PATH_AREACELLO_LONG_NAME
    e = _expected(c96.toy)
    for name in ("SSH", "thetao_0", "thetao_1", "so_0", "so_1"):
        assert np.isfinite(_on_g(store[name].isel(time=-1))[e.gained_band]).all()


def _direct_run_config(tmp_path, monkeypatch) -> tuple[str, PipelineConfig]:
    """#1594's om4 toy run, its artifact built by the updated
    ``python -m pipeline.native_area_weights``."""
    config_path, config, *_ = _toy_run_config(tmp_path, monkeypatch)
    config.weights_url = toy_artifact_built(
        tmp_path, monkeypatch, _toy_grid([0, 1, 2, 3])
    )
    with open(config_path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(config), f)
    return config_path, config


def test_main_store_regrid_path_direct(tmp_path, monkeypatch):
    config_path, config = _direct_run_config(tmp_path, monkeypatch)
    _run_main(monkeypatch, config_path)
    store = xr.open_zarr(config.output.path)
    assert store.attrs[REGRID_PATH_ATTR] == REGRID_PATH_DIRECT
    assert store.attrs[REGRID_PATH_DESCRIPTION_ATTR] == direct_description("toy")
    assert (
        store["areacello"].attrs["long_name"]
        == "cell area covered by native ocean-model cells"
    )


def test_main_raises_without_regrid_path(tmp_path, monkeypatch):
    config_path, config = _direct_run_config(tmp_path, monkeypatch)
    path = os.path.join(config.weights_url, WEIGHTS_FILENAME)
    ds = xr.open_dataset(path).load()
    ds.close()
    del ds.attrs[REGRID_PATH_ATTR]
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match="pipeline.native_area_weights"):
        _run_main(monkeypatch, config_path)
    assert not os.path.exists(config.output.path)
