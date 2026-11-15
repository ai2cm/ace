"""om4 run tests for the native-area change: statics and chunks on the PRD
toy example (table 3)."""

import dataclasses
import sys
from fractions import Fraction as F

import numpy as np
import pytest
import xarray as xr

for _name in ("xesmf", "esmpy", "apache_beam", "xarray_beam", "obstore"):
    pytest.importorskip(_name)

import xarray_beam as xbeam
import yaml
import zarr

from ..config_io import OutputConfig, StaticsConfig, WetmaskConfig
from ..native_area import (
    AREA_RTOL,
    F64_RTOL,
    LEVEL_DIM,
    NATIVE_AREA_CHECKS,
    Z_STAR_DEFINITION,
    assert_close,
)
from ..postprocess import Postprocess, PostprocessConfig
from ..test_native_area import (
    A_NATIVE,
    TOY_C,
    TOY_DEPTHO,
    TOY_HI,
    TOY_SFTOF,
    TOY_SO,
    TOY_SSH,
    TOY_X,
    TOY_ZETA,
    _native,
    _toy,
    _toy_artifact,
    _toy_wetmask,
)
from ..zarr_io import OUTPUT_DTYPE, TIME_DIM
from . import postprocess as om4_postprocess
from . import run
from .check_output import check_output
from .config import PipelineConfig, StreamConfig
from .run import (
    build_statics,
    land_nan_exempt_names,
    load_native_statics,
    native_area_checks,
    process_chunk,
)


def _toy_config(tmp_path, monkeypatch, sftof=TOY_SFTOF) -> PipelineConfig:
    _, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    store = str(tmp_path / "ocean_static.zarr")
    native = {
        "areacello": _native(A_NATIVE, "areacello"),
        "sftof": _native(sftof, "sftof"),
        "deptho": _native(TOY_DEPTHO, "deptho"),
    }
    xr.Dataset(native).to_zarr(store)
    return PipelineConfig(
        streams=[],
        statics=StaticsConfig(store=store, variables=["deptho"]),
        wetmask=WetmaskConfig(store="unused", variable="thetao"),
        target_grid="toy",
        weights_url=url,
        output=OutputConfig(path=str(tmp_path / "out.zarr")),
        expected_level_count=2,
    )


def _assert_stored(da, tabled, name):
    rhs = np.array([float(x) for x in tabled])
    assert_close(
        da.values.ravel(), rhs, rtol=AREA_RTOL, scale=np.abs(rhs).max(), context=name
    )


def test_build_statics_toy_table3(tmp_path, monkeypatch):
    statics = build_statics(_toy_config(tmp_path, monkeypatch), _toy_wetmask())
    tabled = {
        "areacello": [F(3, 2), F(5, 2), 0],
        "sftof": [100, 20, 0],
        "sea_surface_fraction": [1, F(1, 5), 0],
        "mask_0": [1, 1, 0],
        "mask_1": [1, 1, 0],
        "mask_2d": [1, 1, 0],
        "h_prime_0": [1, 1, 0],
        "h_prime_1": [F(5, 3), 1, 0],
    }
    for name, values in tabled.items():
        _assert_stored(statics[name], values, name)
        assert statics[name].dtype == OUTPUT_DTYPE, name
    _assert_stored(statics["deptho"][..., :2], [F(8, 3), 2], "deptho_J")
    assert np.isnan(statics["deptho"].values.ravel()[2]), "deptho_J NaN on Z"
    assert [float(statics[f"idepth_{k}"]) for k in range(3)] == [0.0, 1.0, 3.0]
    assert set(statics.data_vars) == set(tabled) | {"deptho"} | {
        f"idepth_{k}" for k in range(3)
    }


def test_build_statics_raises_on_native_assert(tmp_path, monkeypatch):
    config = _toy_config(tmp_path, monkeypatch, sftof=[100.0, 50.0, np.nan])
    with pytest.raises(AssertionError, match="sftof_i = 100 m_i0"):
        build_statics(config, _toy_wetmask())


def test_land_nan_exempt_includes_new_statics():
    exempt = set(land_nan_exempt_names(2))
    assert {"areacello", "sftof", "sea_surface_fraction", "h_prime_0", "h_prime_1"} <= (
        exempt
    )
    assert "deptho" not in exempt


def _toy_chunk(
    stream: StreamConfig, url: str, fields: dict, level=None, a=A_NATIVE, attrs=None
):
    """process_chunk on one timestep of the toy ``fields`` (native, float32),
    with source ``attrs`` per field name."""
    wetmask = _toy_wetmask()
    attrs = attrs or {}
    ds = xr.Dataset(
        {
            name: _native(values, name)
            .astype(np.float32)
            .expand_dims(TIME_DIM)
            .assign_attrs(attrs.get(name, {}))
            for name, values in fields.items()
        }
    )
    offsets = {TIME_DIM: 0}
    if level is not None:
        ds = ds.expand_dims({LEVEL_DIM: wetmask[LEVEL_DIM].values[[level]]}, axis=1)
        offsets[LEVEL_DIM] = level
    _, out = process_chunk(
        xbeam.Key(offsets),
        ds,
        stream,
        wetmask,
        url,
        "toy",
        _native(a, "areacello"),
        _native(TOY_DEPTHO, "deptho"),
    )
    return out


def _assert_row(da, tabled, name):
    """Tabled values on J1, J2, stored dtype and dims; NaN on J3 (Z)."""
    assert da.dtype == OUTPUT_DTYPE, name
    assert da.dims == (TIME_DIM, "lat", "lon"), name
    _assert_stored(da[..., :2], tabled, name)
    assert np.isnan(da.values.ravel()[2]), f"{name}: NaN on Z"


def test_process_chunk_toy_table3(tmp_path, monkeypatch):
    _, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    ocean = StreamConfig(name="ocean", store="toy", variables=["x"])
    out = _toy_chunk(ocean, url, {"x": TOY_X}, level=1)
    _assert_row(out["x_1"], [F(3, 5), -1], "X^tw_J1")

    flux = StreamConfig(
        name="flux", store="toy", variables=["hfds"], postprocess=["hfds_total_area"]
    )
    out = _toy_chunk(flux, url, {"hfds": TOY_ZETA})
    _assert_row(out["hfds"], [F(1, 3), -1], "X^a_J of ζ")
    _assert_row(out["hfds_total_area"], [F(1, 3), F(-1, 5)], "X_J of ζ (full cell)")

    ice = StreamConfig(
        name="ice",
        store="toy",
        variables=["sea_ice_fraction", "HI", "UI", "VI"],
        renaming={"sea_ice_fraction": "ocean_sea_ice_fraction"},
        full_cell_variables=["sea_ice_fraction"],
        postprocess=["sea_ice"],
        ice_area_weight="sea_ice_fraction",
        ice_area_weighted=["HI", "UI", "VI"],
        ice_area_products={"HI": "ocean_sea_ice_thickness_mean"},
    )
    zero = [0.0, 0.0, np.nan]
    out = _toy_chunk(
        ice, url, {"sea_ice_fraction": TOY_C, "HI": TOY_HI, "UI": zero, "VI": zero}
    )
    _assert_row(out["sea_ice_fraction"], [F(5, 6), F(1, 10)], "sea_ice_fraction_J")
    _assert_row(out["ocean_sea_ice_fraction"], [F(5, 6), F(1, 2)], "X^a_J of c")
    _assert_row(out["sea_ice_volume"], [F(9, 4), F(1, 4)], "sea_ice_volume_J")
    _assert_row(out["ocean_sea_ice_thickness_mean"], [F(3, 2), F(1, 2)], "v_J")
    _assert_row(out["HI"], [F(9, 5), 1], "stored HI (ice-area weighted)")


def test_process_chunk_calving_residue(tmp_path, monkeypatch):
    """calving_residue_J is the formula on the stored per-ocean-area inputs,
    and its ocean-area integral is the native one (no f_J factor)."""
    monkeypatch.setattr(run, "OUTPUT_DTYPE", np.float64)
    config = _toy_config(tmp_path, monkeypatch)
    statics = build_statics(config, _toy_wetmask())
    flux = StreamConfig(
        name="flux",
        store="toy",
        variables=["hflso", "evs", "prsn"],
        postprocess=["calving_residue"],
    )
    fields = {
        "hflso": TOY_ZETA,
        "evs": [3e-6, -1e-6, np.nan],
        "prsn": [2e-6, 5e-6, np.nan],
    }
    out = _toy_chunk(flux, config.weights_url, fields)
    L_v, L_f = om4_postprocess.LATENT_HEAT_VAPOR, om4_postprocess.LATENT_HEAT_FUSION
    residue_J = out["calving_residue"].values.ravel()[:2]
    formula = -out["hflso"] + L_v * out["evs"] - L_f * out["prsn"]
    formula_J = formula.values.ravel()[:2]
    assert_close(
        residue_J,
        formula_J,
        rtol=F64_RTOL,
        scale=np.abs(formula_J).max(),
        context="calving_residue_J",
    )
    assert np.isnan(out["calving_residue"].values.ravel()[2]), "NaN on Z"

    native = {
        n: np.asarray(v, np.float32).astype(np.float64) for n, v in fields.items()
    }
    residue_i = -native["hflso"] + L_v * native["evs"] - L_f * native["prsn"]
    m_i = _toy_wetmask().isel({LEVEL_DIM: 0}).values.ravel()
    W_J = (statics["areacello"] * statics["sea_surface_fraction"]).values.ravel()[:2]
    lhs = np.sum(W_J * residue_J)
    rhs = np.nansum(A_NATIVE * m_i * residue_i)
    assert_close(
        lhs,
        rhs,
        rtol=AREA_RTOL,
        scale=np.abs(residue_i[:2]).max(),
        context="Σ_J W_J X_J = Σ_i a_i m_i X_i",
    )
    assert out["calving_residue"].attrs["units"] == "W/m2"


def test_ice_area_output_attrs(tmp_path, monkeypatch):
    _, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    ice = StreamConfig(
        name="ice",
        store="toy",
        variables=["sea_ice_fraction", "HI", "T1"],
        ice_area_weight="sea_ice_fraction",
        ice_area_weighted=["HI", "T1"],
        ice_area_products={"HI": "ocean_sea_ice_thickness_mean"},
    )
    attrs = {
        "HI": {
            "long_name": "ice thickness",
            "units": "m-ice",
            "cell_methods": "time: point",
        },
        "T1": {
            "long_name": "ice layer 1 temperature",
            "units": "degC",
            "cell_methods": "time: mean",
        },
    }
    out = _toy_chunk(
        ice, url, {"sea_ice_fraction": TOY_C, "HI": TOY_HI, "T1": TOY_ZETA}, attrs=attrs
    )
    for name, method in (("HI", "time: point"), ("T1", "time: mean")):
        assert out[name].attrs["long_name"] == attrs[name]["long_name"]
        assert out[name].attrs["units"] == attrs[name]["units"]
        assert out[name].attrs["cell_methods"] == f"{method} area: mean where sea_ice"
    v = out["ocean_sea_ice_thickness_mean"].attrs
    assert (
        v["long_name"] == "ice thickness times sea_ice_fraction, mean over ocean area"
    )
    assert v["units"] == "m-ice"
    assert v["cell_methods"] == "time: point area: mean where sea"
    assert v["source_variable"] == "HI"
    assert "sea_ice_fraction" in v["derivation"]


def _ice_area_chunk(url, weight, fields, weighted, variables=None):
    """process_chunk on a toy ``ice`` stream weighting ``weighted`` by
    ``sea_ice_fraction`` = ``weight``."""
    names = list(fields)
    ice = StreamConfig(
        name="ice",
        store="toy",
        variables=variables if variables is not None else names,
        ice_area_weight="sea_ice_fraction",
        ice_area_weighted=weighted,
    )
    return _toy_chunk(ice, url, {"sea_ice_fraction": weight, **fields}), ice


def test_ice_area_weight_drops_ice_free_values(tmp_path, monkeypatch):
    _, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    out, _ = _ice_area_chunk(
        url, [0.0, 0.5, np.nan], {"TS": [-5.0, -2.0, np.nan]}, ["TS"]
    )
    _assert_row(out["TS"], [-2, -2], "TS_J")


def test_ice_area_weighted_zero_on_ice_free_ocean(tmp_path, monkeypatch):
    _, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    out, _ = _ice_area_chunk(
        url, [0.0, 0.0, np.nan], {"HS": [0.0, 0.0, np.nan]}, ["HS"]
    )
    assert np.isfinite(out["HS"].values.ravel()[:2]).all()
    _assert_row(out["HS"], [0, 0], "HS_J")


def test_ice_area_weighted_velocity(tmp_path, monkeypatch):
    _, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    out, _ = _ice_area_chunk(
        url, TOY_C, {"UI": TOY_HI, "VI": [0.0, 0.0, np.nan]}, ["UI", "VI"]
    )
    _assert_row(out["UI"], [F(9, 5), 1], "UI_J")
    _assert_row(out["VI"], [0, 0], "VI_J")


def test_ice_area_weight_only_not_written(tmp_path, monkeypatch):
    config = _toy_config(tmp_path, monkeypatch)
    out, ice = _ice_area_chunk(
        config.weights_url, TOY_C, {"TS": TOY_HI}, ["TS"], variables=["TS"]
    )
    config = dataclasses.replace(config, streams=[ice])
    native = xr.Dataset(
        {"TS": _native(TOY_HI, "TS"), "sea_ice_fraction": _native(TOY_C, "c")}
    )
    expected = run._expected_output_names(config, {"ice": native})
    for names in (set(out.data_vars), expected):
        assert names == {"TS"}


def test_interval_ice_fraction_twin(tmp_path, monkeypatch):
    """A block-meaned full-cell ice fraction and its per-ocean-area twin pass
    sea_ice_fraction_consistency."""
    _, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    name, twin = (
        "interval_average_sea_ice_fraction",
        "ocean_interval_average_sea_ice_fraction",
    )
    stream = StreamConfig(
        name="ice_flux_mean",
        store="toy",
        variables=[name],
        time_block_mean_factor=2,
        full_cell_variables=[name],
        renaming={name: twin},
        postprocess=[
            PostprocessConfig(
                "sea_ice_fraction_consistency",
                {"sea_ice_fraction": name, "ocean_sea_ice_fraction": twin},
            )
        ],
    )
    c = _native(TOY_C, name).astype(np.float32)
    c.attrs = {"long_name": "ice concentration", "cell_methods": "time: mean"}
    time = np.array(["2000-01-01T00", "2000-01-01T06"], dtype="datetime64[ns]")
    ds = xr.Dataset({name: xr.concat([c, c], TIME_DIM)}).assign_coords({TIME_DIM: time})
    _, out = process_chunk(
        xbeam.Key({TIME_DIM: 0}),
        ds,
        stream,
        _toy_wetmask(),
        url,
        "toy",
        _native(A_NATIVE, "areacello"),
        _native(TOY_DEPTHO, "deptho"),
    )
    assert set(out.data_vars) == {name, twin}
    _assert_row(out[name], [F(5, 6), F(1, 10)], "full-cell c_J")
    _assert_row(out[twin], [F(5, 6), F(1, 2)], "X^a_J of c")
    assert out[name].attrs["long_name"] == "ice concentration, fraction of cell area"
    assert out[twin].attrs["long_name"] == "ice concentration, fraction of ocean area"


def _toy_time_store(path, name, time) -> str:
    ds = xr.Dataset(
        {name: ((TIME_DIM, "lat", "lon"), np.zeros((time.size, 1, 3)))},
        coords={TIME_DIM: time},
    )
    ds.to_zarr(str(path))
    return str(path)


@pytest.mark.parametrize("shift", [None, np.timedelta64(6, "h")])
def test_open_stream_merges_ice_area_weight_store(tmp_path, monkeypatch, shift):
    config = _toy_config(tmp_path, monkeypatch)
    time = np.array(["2000-01-01", "2000-01-06"], dtype="datetime64[ns]")
    weight_time = time if shift is None else time + shift
    stream = StreamConfig(
        name="ice",
        store=_toy_time_store(tmp_path / "x.zarr", "T_bulkice", time),
        variables=["T_bulkice"],
        ice_area_weight="sea_ice_fraction",
        ice_area_weight_store=_toy_time_store(
            tmp_path / "c.zarr", "sea_ice_fraction", weight_time
        ),
        ice_area_weighted=["T_bulkice"],
    )
    if shift is None:
        ds = run.open_stream(stream, config)
        assert set(ds.data_vars) == {"T_bulkice", "sea_ice_fraction"}
    else:
        with pytest.raises(AssertionError, match="c.zarr differs from time of"):
            run.open_stream(stream, config)


def test_process_chunk_drops_outputs(tmp_path, monkeypatch):
    config = _toy_config(tmp_path, monkeypatch)
    ice = StreamConfig(
        name="ice",
        store="toy",
        variables=["SST"],
        postprocess=[PostprocessConfig("kelvin_sst", {"celsius_sst": "SST"})],
        drop_outputs=["SST"],
    )
    out = _toy_chunk(ice, config.weights_url, {"SST": TOY_ZETA})
    config = dataclasses.replace(config, streams=[ice])
    expected = run._expected_output_names(
        config, {"ice": xr.Dataset({"SST": _native(TOY_ZETA, "SST")})}
    )
    for names in (set(out.data_vars), expected):
        assert "SST_degK" in names
        assert "SST" not in names


def test_process_chunk_raises_on_other_native_areas(tmp_path, monkeypatch):
    """The chunk identities run inside process_chunk: native a_i other than
    the artifact's fail the 3-D row."""
    _, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    ocean = StreamConfig(name="ocean", store="toy", variables=["x"])
    with pytest.raises(AssertionError, match="level 1: Σ_J V'_Jk X\\^tw_Jk"):
        _toy_chunk(ocean, url, {"x": TOY_X}, level=1, a=[1.0, 2.0, 2.0])


def _hi_area_fraction(ds, context):
    """Today's sea_ice_volume = HI · A_J · sea_ice_fraction_J."""
    ds["sea_ice_volume"] = ds["HI"] * context.areacello * ds["sea_ice_fraction"]
    return ds


def test_process_chunk_checks_sea_ice_volume(tmp_path, monkeypatch):
    _, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    monkeypatch.setitem(
        om4_postprocess.POSTPROCESS,
        "sea_ice",
        lambda: Postprocess(
            _hi_area_fraction,
            requires=("sea_ice_fraction", "HI"),
            adds=("sea_ice_volume",),
        ),
    )
    ice = StreamConfig(
        name="ice",
        store="toy",
        variables=["sea_ice_fraction", "HI"],
        renaming={"sea_ice_fraction": "ocean_sea_ice_fraction"},
        full_cell_variables=["sea_ice_fraction"],
        postprocess=["sea_ice"],
    )
    with pytest.raises(AssertionError, match="sea_ice_volume"):
        _toy_chunk(ice, url, {"sea_ice_fraction": TOY_C, "HI": TOY_HI})


def _toy_streams(config: PipelineConfig):
    """Two timesteps: an ocean stream (SSH; thetao, so) with t0 η = TOY_SSH and
    t1 η = H/10, and an ice stream whose SSH is stored as SSH_ice."""
    wetmask = _toy_wetmask()
    time = xr.DataArray(
        np.array(["2000-01-01", "2000-01-06"], dtype="datetime64[ns]"), dims=TIME_DIM
    )
    ssh = np.array([TOY_SSH, [0.1 * h for h in TOY_DEPTHO]])[:, None, :]
    levels = np.array([[TOY_X, TOY_X], [TOY_SO, TOY_SO]])[:, None, :, None, :]
    levels = np.broadcast_to(levels, (2, 2, 2, 1, 3))
    dims_3d = (TIME_DIM, LEVEL_DIM, "lat", "lon")
    coords = {TIME_DIM: time, LEVEL_DIM: wetmask[LEVEL_DIM]}
    ocean = xr.Dataset(
        {
            "SSH": ((TIME_DIM, "lat", "lon"), ssh),
            "thetao": (dims_3d, levels[0]),
            "so": (dims_3d, levels[1]),
        },
        coords=coords,
    ).astype(np.float32)
    ice = xr.Dataset(
        {"SSH": ((TIME_DIM, "lat", "lon"), ssh[::-1])}, coords={TIME_DIM: time}
    ).astype(np.float32)
    config.streams = [
        StreamConfig(name="ocean", store="toy", variables=["SSH", "thetao", "so"]),
        StreamConfig(
            name="ice", store="toy", variables=["SSH"], renaming={"SSH": "SSH_ice"}
        ),
    ]
    return {"ocean": ocean, "ice": ice}, wetmask, time


def test_native_area_checks_toy(tmp_path, monkeypatch):
    """The ocean SSH (not SSH_ice) at the first and last time; the artifact's
    a/ã spread."""
    config = _toy_config(tmp_path, monkeypatch)
    streams, wetmask, time = _toy_streams(config)
    native = load_native_statics(config, wetmask)
    checks = native_area_checks(config, streams, wetmask, native, time)

    _, result = _toy(z_row=True)
    assert checks["a_over_a_tilde_spread"] == result.a_over_a_tilde_spread
    z_star = checks["z_star_residual"]
    assert z_star["definition"] == Z_STAR_DEFINITION
    first, last = (str(t) for t in time.values)
    assert list(z_star["snapshots"]) == [first, last]
    assert_close(
        np.array(z_star["snapshots"][first]["sigma_J"]["vol"]),
        np.array([1 / 128, 1 / 216]),
        rtol=AREA_RTOL,  # native η is float32
        scale=1.0,
        context="t0 σ_J vol_k (η = TOY_SSH)",
    )
    assert_close(
        np.array(z_star["snapshots"][last]["sigma_J"]["vol"]),
        np.zeros(2),
        rtol=AREA_RTOL,  # native η is float32
        scale=1.0,
        context="t1 σ_J vol_k (η = H/10)",
    )


def _toy_run_config(tmp_path, monkeypatch):
    """The toy streams written as stores, and a config file reading them."""
    config = _toy_config(tmp_path, monkeypatch)
    streams, wetmask, time = _toy_streams(config)
    for stream in config.streams:
        stream.store = str(tmp_path / f"{stream.name}.zarr")
        streams[stream.name].chunk({TIME_DIM: 1}).to_zarr(stream.store)
    config.wetmask = WetmaskConfig(store=config.streams[0].store, variable="thetao")
    config.output.time_shard_size = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(dataclasses.asdict(config)))
    return str(config_path), config, streams, wetmask, time


def _run_main(monkeypatch, config_path, *flags):
    monkeypatch.setattr(sys, "argv", ["run", "--config", config_path, *flags])
    run.main()


def test_main_writes_native_area_checks(tmp_path, monkeypatch):
    """main writes native_area_checks to the store's root attrs."""
    config_path, config, streams, wetmask, time = _toy_run_config(tmp_path, monkeypatch)
    _run_main(monkeypatch, config_path, *DIRECT_FLAGS)

    native = load_native_statics(config, wetmask)
    expected = native_area_checks(config, streams, wetmask, native, time)
    store = xr.open_zarr(config.output.path)
    assert store.attrs[NATIVE_AREA_CHECKS] == expected


ACE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
IMAGE_TAG = "2026-10-22-toy-image"
DIRECT_FLAGS = ("--runner", "DirectRunner", "--ace-commit", ACE_COMMIT)
DATAFLOW_FLAGS = (
    "--runner",
    "DataflowRunner",
    "--ace-commit",
    ACE_COMMIT,
    "--image-tag",
    IMAGE_TAG,
)


class _Parsed(Exception):
    pass


def _parse(monkeypatch, *flags):
    """main up to its first read after argument checks."""

    def stop(*args, **kwargs):
        raise _Parsed

    monkeypatch.setattr(run, "load_config", stop)
    _run_main(monkeypatch, "unused.yaml", *flags)


@pytest.mark.parametrize("runner", ["DirectRunner", "DataflowRunner"])
def test_parser_requires_ace_commit(monkeypatch, capsys, runner):
    image_tag = ("--image-tag", IMAGE_TAG) if runner == "DataflowRunner" else ()
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--runner", runner, *image_tag)
    assert "--ace-commit" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags, error",
    [
        (DIRECT_FLAGS, None),
        (("--ace-commit", ACE_COMMIT), None),
        (DATAFLOW_FLAGS, None),
        (("--runner=dataflow", "--ace-commit", ACE_COMMIT, "--image-tag", "t"), None),
        (DATAFLOW_FLAGS[:4], "--image-tag is required under DataflowRunner"),
        (
            ("--runner", "dataflow", "--ace-commit", ACE_COMMIT),
            "--image-tag is required under DataflowRunner",
        ),
        (
            (*DIRECT_FLAGS, "--image-tag", IMAGE_TAG),
            "--image-tag is only accepted under DataflowRunner",
        ),
        (
            ("--ace-commit", ACE_COMMIT, "--image-tag", IMAGE_TAG),
            "--image-tag is only accepted under DataflowRunner",
        ),
    ],
)
def test_parser_image_tag_rule(monkeypatch, capsys, flags, error):
    """--image-tag under DataflowRunner only; no --runner is DirectRunner."""
    if error is None:
        with pytest.raises(_Parsed):
            _parse(monkeypatch, *flags)
    else:
        with pytest.raises(SystemExit):
            _parse(monkeypatch, *flags)
        assert error in capsys.readouterr().err


class _Built(Exception):
    pass


@pytest.mark.parametrize(
    "flags, expected",
    [
        (DIRECT_FLAGS, {"ace_commit": ACE_COMMIT}),
        (DATAFLOW_FLAGS, {"ace_commit": ACE_COMMIT, "image_tag": IMAGE_TAG}),
    ],
)
def test_build_template_provenance_attrs(tmp_path, monkeypatch, flags, expected):
    """main's template, stopped before the pipeline runs: image_tag only under
    DataflowRunner, no ace_dirty (D23)."""
    config_path, *_ = _toy_run_config(tmp_path, monkeypatch)
    built = {}
    build_template = run.build_template

    def capture(*args, **kwargs):
        built["template"] = build_template(*args, **kwargs)
        raise _Built

    monkeypatch.setattr(run, "build_template", capture)
    with pytest.raises(_Built):
        _run_main(monkeypatch, config_path, *flags)
    attrs = built["template"].attrs
    provenance = {
        name: attrs[name] for name in ("ace_commit", "image_tag") if name in attrs
    }
    assert provenance == expected
    assert "ace_dirty" not in attrs
    assert NATIVE_AREA_CHECKS in attrs


def test_check_output_requires_provenance(tmp_path, monkeypatch):
    config_path, config, *_ = _toy_run_config(tmp_path, monkeypatch)
    _run_main(monkeypatch, config_path, *DIRECT_FLAGS)
    check_output(config_path, expected_timesteps=2)
    del zarr.open_group(config.output.path, mode="r+").attrs["ace_commit"]
    with pytest.raises(AssertionError, match="lacks the root attr ace_commit"):
        check_output(config_path, expected_timesteps=2)
