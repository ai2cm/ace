"""randco2 on the C96-path toy of ``pipeline/test_c96_weights.py`` with the
coast at lon 20° (``om4/test_c96_path_run.py``: gained_band on lon 20–40,
land-only J on lon 40–60), and the store root ``regrid_path`` of C96 and
direct artifacts. i: toy O cell; J: toy G cell.
"""

import os
import sys

import numpy as np
import pytest
import xarray as xr

for _name in ("xesmf", "esmpy", "apache_beam", "xarray_beam", "obstore"):
    pytest.importorskip(_name)

from .. import c96_weights
from ..native_area_weights import regrid_path_description as direct_description
from ..om4.test_c96_path_run import COAST_LON, _assert_g, _expected, _on_g
from ..om4.test_native_area_run import DIRECT_FLAGS
from ..postprocess import DERIVATION_ATTR
from ..test_c96_weights import O_DIMS, TARGET, Toy, _argv, _inputs, _on_source, _toy
from ..test_native_area import toy_artifact_built
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
from .config import load_config
from .test_native_area_run import TOY_RUN_CONFIG, _toy_run, _toy_source_grid


def _drop_weights_attr(url: str, name: str) -> None:
    path = os.path.join(url, WEIGHTS_FILENAME)
    ds = xr.open_dataset(path).load()
    ds.close()
    del ds.attrs[name]
    ds.to_netcdf(path)


def _wet_field(toy: Toy, scale: float) -> np.ndarray:
    values = scale * (1.0 + 0.1 * np.arange(toy.wet.size))
    return np.where(toy.wet == 1, values, np.nan)


def _c96_run(tmp_path, monkeypatch, description: bool = True) -> tuple[str, Toy]:
    """The toy C96 artifact built by ``python -m pipeline.c96_weights``,
    statics and source stores and a config; returns (config path, toy)."""
    toy = _toy(COAST_LON)
    inputs = _inputs(tmp_path, monkeypatch, toy)
    c96_weights.main(_argv(inputs))
    if not description:
        _drop_weights_attr(inputs.output_url, REGRID_PATH_DESCRIPTION_ATTR)
    statics_store = str(tmp_path / "run_static.zarr")
    xr.Dataset(
        {
            "areacello": _on_source(toy.native_area),
            "sftof": _on_source(100.0 * toy.wet),
        }
    ).to_zarr(statics_store)
    source_store = str(tmp_path / "source.zarr")
    time = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
    sst = _on_source(_wet_field(toy, 1.0)).values
    concentration = _on_source(np.clip(_wet_field(toy, 0.5), 0.0, 1.0)).values
    dims = (TIME_DIM, *O_DIMS)
    xr.Dataset(
        {
            "SST": (dims, np.stack([sst, -sst]).astype(np.float32)),
            "sea_ice_fraction": (
                dims,
                np.stack([concentration] * 2).astype(np.float32),
            ),
        },
        coords={TIME_DIM: time},
    ).chunk({TIME_DIM: 1}).to_zarr(source_store)
    path = tmp_path / "config.yaml"
    path.write_text(
        TOY_RUN_CONFIG.format(
            weights_url=inputs.output_url,
            output_path=tmp_path / "out.zarr",
            statics_store=statics_store,
            source_store=source_store,
        )
    )
    return str(path), toy


def test_build_statics_c96_path(tmp_path, monkeypatch):
    """areacello = areacello_c96path, > 0 and every static finite on the
    land-only J; sea_surface_fraction = W'_J0 / A_J; check_static_identities
    runs inside."""
    monkeypatch.setattr(run, "OUTPUT_DTYPE", np.float64)
    config_path, toy = _c96_run(tmp_path, monkeypatch)
    e = _expected(toy)
    config = load_config(config_path)
    statics = run.build_statics(config, run.load_wetmask(config))

    _assert_g(statics["areacello"], e.area, "areacello = areacello_c96path")
    _assert_g(
        statics["sea_surface_fraction"],
        e.ocean_area / e.area,
        "sea_surface_fraction = ocean_area_c96path_J0 / areacello_c96path_J",
    )
    _assert_g(statics["sftof"], 100 * e.ocean_area / e.area, "sftof")
    assert (_on_g(statics["areacello"])[e.land_only] > 0).all()
    for name, da in statics.data_vars.items():
        assert np.isfinite(_on_g(da)).all(), name

    assert statics["areacello"].attrs["long_name"] == C96_PATH_AREACELLO_LONG_NAME
    assert statics["areacello"].attrs[DERIVATION_ATTR] == C96_PATH_AREACELLO_DERIVATION
    assert statics["sftof"].attrs[DERIVATION_ATTR] == C96_PATH_SFTOF_DERIVATION


def test_build_statics_direct_attrs(tmp_path, monkeypatch):
    """#1594's strings on a direct artifact."""
    config = load_config(_toy_run(tmp_path, monkeypatch))
    statics = run.build_statics(config, run.load_wetmask(config))
    assert (
        statics["areacello"].attrs["long_name"]
        == "target cell area covered by native ocean-grid cells"
    )
    assert (
        "0 where no native cell overlaps"
        in (statics["areacello"].attrs[DERIVATION_ATTR])
    )
    assert "0 where no native cell overlaps" in statics["sftof"].attrs[DERIVATION_ATTR]


def test_process_chunk_c96_path(tmp_path, monkeypatch):
    """2-D and full-cell variables pass check_field_identities (inside
    _process_chunk); every output is finite exactly on ocean J, gained_band
    included."""
    config_path, toy = _c96_run(tmp_path, monkeypatch)
    e = _expected(toy)
    config = load_config(config_path)
    wetmask = run.load_wetmask(config)
    ds = run.open_stream(config.stream, config).isel({TIME_DIM: [0]}).load()
    out = run._process_chunk(
        ds,
        config.stream,
        wetmask,
        config.weights_url,
        config.target_grid,
        run.load_native_statics(config, wetmask)["areacello"],
    )
    assert {"SST", "sea_ice_fraction", "ocean_sea_ice_fraction"} <= set(out.data_vars)
    ocean = e.ocean_area > 0
    assert ocean[e.gained_band].all()
    for name, da in out.data_vars.items():
        np.testing.assert_array_equal(np.isfinite(_on_g(da)), ocean, err_msg=name)


def _run_main(monkeypatch, config_path):
    monkeypatch.setattr(sys, "argv", ["run", "--config", config_path, *DIRECT_FLAGS])
    run.main()


@pytest.mark.parametrize("description", [True, False])
def test_main_store_regrid_path_c96(tmp_path, monkeypatch, description):
    config_path, _ = _c96_run(tmp_path, monkeypatch, description=description)
    _run_main(monkeypatch, config_path)
    store = xr.open_zarr(load_config(config_path).output.path)
    assert store.attrs[REGRID_PATH_ATTR] == REGRID_PATH_C96
    if description:
        assert store.attrs[REGRID_PATH_DESCRIPTION_ATTR] == (
            c96_weights.regrid_path_description(TARGET)
        )
    else:
        assert REGRID_PATH_DESCRIPTION_ATTR not in store.attrs
    assert store["areacello"].attrs["long_name"] == C96_PATH_AREACELLO_LONG_NAME


def _direct_run(tmp_path, monkeypatch) -> str:
    """#1594's randco2 toy run, its artifact built by the updated
    ``python -m pipeline.native_area_weights``."""
    config_path = _toy_run(tmp_path, monkeypatch)
    url = toy_artifact_built(tmp_path, monkeypatch, _toy_source_grid())
    text = open(config_path).read()
    old = load_config(config_path).weights_url
    with open(config_path, "w") as f:
        f.write(text.replace(f"weights_url: {old}", f"weights_url: {url}"))
    return config_path


def test_main_store_regrid_path_direct(tmp_path, monkeypatch):
    config_path = _direct_run(tmp_path, monkeypatch)
    _run_main(monkeypatch, config_path)
    store = xr.open_zarr(load_config(config_path).output.path)
    assert store.attrs[REGRID_PATH_ATTR] == REGRID_PATH_DIRECT
    assert store.attrs[REGRID_PATH_DESCRIPTION_ATTR] == direct_description("toy")


def test_main_raises_without_regrid_path(tmp_path, monkeypatch):
    config_path = _direct_run(tmp_path, monkeypatch)
    config = load_config(config_path)
    _drop_weights_attr(config.weights_url, REGRID_PATH_ATTR)
    with pytest.raises(ValueError, match="pipeline.c96_weights"):
        _run_main(monkeypatch, config_path)
    assert not os.path.exists(config.output.path)
