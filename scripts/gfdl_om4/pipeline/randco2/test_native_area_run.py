"""randco2 on the PRD toy (``## Toy example``, table 3): native a = (1, 1, 2),
m_0 = (1, 1, 0), SST ζ = (1, −1, ·), sea ice c = (1, 1/2, ·); target J3 is in
Z = {J : A_J = 0}. The toy is written as a native-area artifact whose source
grid has the tracer dims (yh, xh) and loaded through ``get_regridder``.
"""

import pathlib
import sys
from fractions import Fraction as F

import numpy as np
import pytest
import xarray as xr
import zarr

from .. import weights
from ..native_area import F64_RTOL, NATIVE_AREA_CHECKS, assert_close
from ..native_area_weights import write_native_area_artifact
from ..om4.config import load_config as load_om4_config
from ..om4.test_native_area_run import (
    ACE_COMMIT,
    DATAFLOW_FLAGS,
    DIRECT_FLAGS,
    IMAGE_TAG,
)
from ..test_native_area import TOY_ATTRS, _perturb_areacello_coords, _toy, _toy_grid
from ..zarr_io import TIME_DIM
from . import run
from .check_output import check_output
from .config import load_config

CONFIGS = pathlib.Path(__file__).parents[2] / "configs"

TOY_CONFIG = """
target_grid: F90
weights_url: gs://bucket/weights
output:
  path: gs://bucket/out.zarr
statics:
  store: gs://bucket/ocean_static.zarr
wetmask:
  store: gs://bucket/source.zarr
  variable: SST
stream:
  name: ice_snapshot
  store: gs://bucket/source.zarr
  variables:
    - SST
"""


def test_config_accepts_statics_with_empty_variables(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(TOY_CONFIG)
    config = load_config(str(path))
    assert config.statics.store == "gs://bucket/ocean_static.zarr"
    assert config.statics.variables == []


@pytest.mark.parametrize("resolution", ["1deg", "4deg"])
def test_randco2_configs_set_statics_store(resolution):
    config = load_config(
        str(CONFIGS / f"cm4-like-am4-randco2-sea-surface-{resolution}.yaml"),
        member="member",
    )
    om4 = load_om4_config(str(CONFIGS / "om4-picontrol-1deg-5daily.yaml"))
    assert config.statics.store == om4.statics.store
    assert config.statics.variables == []


NAN = float("nan")
TOY_A = [1.0, 1.0, 2.0]
TOY_SFTOF = [100.0, 100.0, 0.0]
TOY_SST = [1.0, -1.0, NAN]
TOY_C = [1.0, 0.5, NAN]

TOY_RUN_CONFIG = """
target_grid: toy
weights_url: {weights_url}
output:
  path: {output_path}
  time_chunk_size: 1
  time_shard_size: 2
statics:
  store: {statics_store}
wetmask:
  store: {source_store}
  variable: SST
stream:
  name: ice_snapshot
  store: {source_store}
  variables:
    - SST
    - sea_ice_fraction
  renaming:
    sea_ice_fraction: ocean_sea_ice_fraction
  full_cell_variables:
    - sea_ice_fraction
  postprocess:
    - name: kelvin_sst
      sources:
        celsius_sst: SST
    - sea_ice_fraction_consistency
"""


def _native(values) -> xr.DataArray:
    return xr.DataArray(np.array([values], dtype=np.float64), dims=("yh", "xh"))


def _toy_source_grid() -> xr.Dataset:
    """One row of three native cells, with the 2-D tracer dims of an artifact."""
    lon_b = np.array([0.0, 1.0, 2.0, 3.0])
    return xr.Dataset(
        coords={
            "lon": (("yh", "xh"), 0.5 * (lon_b[None, :-1] + lon_b[None, 1:])),
            "lat": (("yh", "xh"), np.zeros((1, 3))),
            "lon_b": (("yh_b", "xh_b"), np.tile(lon_b, (2, 1))),
            "lat_b": (("yh_b", "xh_b"), np.array([[-0.5] * 4, [0.5] * 4])),
        }
    )


def _toy_run(tmp_path, monkeypatch, a=TOY_A, sftof=TOY_SFTOF, c=TOY_C):
    """Write the toy artifact, statics store, source store and config.

    Returns the config path. Two timesteps: the toy fields, then their
    negation (sea ice: unchanged)."""
    _, result = _toy(z_row=True)
    target = _toy_grid([0.0, 1.5, 3.0, 4.0])
    weights_url = str(tmp_path / "artifact")
    attrs = TOY_ATTRS | {"a_over_a_tilde_spread": result.a_over_a_tilde_spread}
    write_native_area_artifact(result, _toy_source_grid(), target, attrs, weights_url)
    monkeypatch.setattr(weights, "make_target_grid", lambda name: target)
    monkeypatch.setattr(weights, "_REGRIDDER_CACHE", {})
    monkeypatch.setattr(weights, "_AREACELLO_CACHE", {})

    statics_store = str(tmp_path / "ocean_static.zarr")
    xr.Dataset({"areacello": _native(a), "sftof": _native(sftof)}).to_zarr(
        statics_store
    )
    source_store = str(tmp_path / "source.zarr")
    time = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
    sst = np.array([TOY_SST, [-x for x in TOY_SST]])
    xr.Dataset(
        {
            "SST": ((TIME_DIM, "yh", "xh"), sst[:, None, :].astype(np.float32)),
            "sea_ice_fraction": (
                (TIME_DIM, "yh", "xh"),
                np.array([c, c])[:, None, :].astype(np.float32),
            ),
        },
        coords={TIME_DIM: time},
    ).chunk({TIME_DIM: 1}).to_zarr(source_store)
    path = tmp_path / "config.yaml"
    path.write_text(
        TOY_RUN_CONFIG.format(
            weights_url=weights_url,
            output_path=tmp_path / "out.zarr",
            statics_store=statics_store,
            source_store=source_store,
        )
    )
    return str(path)


def _assert_tabled(da, tabled, context):
    rhs = np.array([float(x) for x in tabled])
    lhs = np.asarray(da, dtype=np.float64).ravel()
    finite = np.isfinite(rhs)
    assert np.array_equal(np.isfinite(lhs), finite), f"{context}: NaN pattern"
    assert_close(
        lhs[finite],
        rhs[finite],
        rtol=F64_RTOL,
        scale=np.abs(rhs[finite]),
        context=context,
    )


def _toy_statics(config_path):
    config = load_config(config_path)
    wetmask = run.load_wetmask(config)
    return config, wetmask, run.build_statics(config, wetmask)


def _toy_chunk(config_path, time=0):
    config = load_config(config_path)
    wetmask = run.load_wetmask(config)
    ds = run.open_stream(config.stream, config).isel({TIME_DIM: [time]}).load()
    native_areacello = run.load_native_statics(config, wetmask)["areacello"]
    return run._process_chunk(
        ds,
        config.stream,
        wetmask,
        config.weights_url,
        config.target_grid,
        native_areacello,
    )


def test_statics_hold_area_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUTPUT_DTYPE", np.float64)
    _, _, statics = _toy_statics(_toy_run(tmp_path, monkeypatch))
    _assert_tabled(statics["areacello"], [F(3, 2), F(5, 2), 0], "areacello_J = A_J")
    _assert_tabled(statics["sftof"], [100, 20, 0], "sftof_J")
    _assert_tabled(
        statics["sea_surface_fraction"], [1, F(1, 5), 0], "sea_surface_fraction_J"
    )
    assert set(statics.data_vars) <= set(run.land_nan_exempt_names())
    assert {"areacello", "sftof", "sea_surface_fraction"} <= set(statics.data_vars)


def test_artifact_coords_off_by_ulps_keep_every_row(tmp_path, monkeypatch):
    """A_J stored with lat, lon a few ulp off the target grid (F90)."""
    monkeypatch.setattr(run, "OUTPUT_DTYPE", np.float64)
    config_path = _toy_run(tmp_path, monkeypatch)
    _perturb_areacello_coords(load_config(config_path).weights_url, ulps=4)
    _, _, statics = _toy_statics(config_path)
    _assert_tabled(statics["areacello"], [F(3, 2), F(5, 2), 0], "areacello_J = A_J")
    out = _toy_chunk(config_path)
    _assert_tabled(out["SST"], [F(1, 3), -1, NAN], "X^a_J of ζ")


def test_chunk_values_on_toy(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUTPUT_DTYPE", np.float64)
    out = _toy_chunk(_toy_run(tmp_path, monkeypatch))
    _assert_tabled(out["SST"], [F(1, 3), -1, NAN], "X^a_J of ζ")
    _assert_tabled(
        out["sea_ice_fraction"], [F(5, 6), F(1, 10), NAN], "sea_ice_fraction_J"
    )
    _assert_tabled(out["ocean_sea_ice_fraction"], [F(5, 6), F(1, 2), NAN], "X^a_J of c")


def test_chunk_identity_raises_on_other_native_areas(tmp_path, monkeypatch):
    config_path = _toy_run(tmp_path, monkeypatch)
    config = load_config(config_path)
    wetmask = run.load_wetmask(config)
    ds = run.open_stream(config.stream, config).isel({TIME_DIM: [0]}).load()
    other = _native([1.0, 2.0, 2.0]).assign_coords(wetmask.coords)
    with pytest.raises(AssertionError, match="X\\^a_J = Σ_i a_i m_i0 X_i, SST"):
        run._process_chunk(
            ds, config.stream, wetmask, config.weights_url, config.target_grid, other
        )


def test_chunk_raises_on_native_sea_ice_fraction_above_one(tmp_path, monkeypatch):
    config_path = _toy_run(tmp_path, monkeypatch, c=[1.5, 0.5, NAN])
    with pytest.raises(AssertionError, match="native c_i ≤ 1"):
        _toy_chunk(config_path)


def test_statics_raise_on_sftof_other_than_100_m0(tmp_path, monkeypatch):
    config_path = _toy_run(tmp_path, monkeypatch, sftof=[100.0, 90.0, 0.0])
    with pytest.raises(AssertionError, match="sftof_i = 100 m_i0"):
        _toy_statics(config_path)


def test_statics_raise_on_areas_other_than_the_artifact(tmp_path, monkeypatch):
    config_path = _toy_run(tmp_path, monkeypatch, a=[1.0, 1.5, 1.5])
    with pytest.raises(AssertionError, match="ŵ_Ji A_J = a_i"):
        _toy_statics(config_path)


def test_history_names_statics_store(tmp_path, monkeypatch):
    config_path = _toy_run(tmp_path, monkeypatch)
    config = load_config(config_path)
    _run_main(monkeypatch, config_path, *DIRECT_FLAGS)
    check_output(config_path, expected_timesteps=2)
    store = xr.open_zarr(config.output.path)
    assert config.statics.store in store.attrs["history"]
    _, result = _toy(z_row=True)
    assert store.attrs[NATIVE_AREA_CHECKS] == {
        "a_over_a_tilde_spread": result.a_over_a_tilde_spread
    }
    for name in run.land_nan_exempt_names():
        assert np.isfinite(store[name].values).all(), name


def _run_main(monkeypatch, config_path, *flags):
    monkeypatch.setattr(sys, "argv", ["run", "--config", config_path, *flags])
    run.main()


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
        (DATAFLOW_FLAGS, None),
        (DATAFLOW_FLAGS[:4], "--image-tag is required under DataflowRunner"),
        (
            (*DIRECT_FLAGS, "--image-tag", IMAGE_TAG),
            "--image-tag is only accepted under DataflowRunner",
        ),
    ],
)
def test_parser_image_tag_rule(monkeypatch, capsys, flags, error):
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
    config_path = _toy_run(tmp_path, monkeypatch)
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
    config_path = _toy_run(tmp_path, monkeypatch)
    _run_main(monkeypatch, config_path, *DIRECT_FLAGS)
    check_output(config_path, expected_timesteps=2)
    output_path = load_config(config_path).output.path
    del zarr.open_group(output_path, mode="r+").attrs["ace_commit"]
    with pytest.raises(AssertionError, match="lacks the root attr ace_commit"):
        check_output(config_path, expected_timesteps=2)
