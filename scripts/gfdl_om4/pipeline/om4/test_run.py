import dataclasses

import numpy as np
import pytest
import xarray as xr
import xarray_beam as xbeam

from ..config_io import OutputConfig, WetmaskConfig
from ..postprocess import ChunkContext
from . import postprocess, run
from .config import PipelineConfig, StaticsConfig, StreamConfig

BLOCK = 4
NY, NX = 3, 5


def _times(n: int) -> xr.CFTimeIndex:
    return xr.date_range(
        "0001-01-01T06", periods=n, freq="6h", calendar="noleap", use_cftime=True
    )


def _wetmask() -> xr.DataArray:
    wet = np.ones((NY, NX), dtype=bool)
    wet[0, 0] = False
    return xr.DataArray(wet, dims=("yh", "xh"))


def _source(n: int) -> xr.Dataset:
    values = np.arange(n, dtype="float32")[:, None, None] * np.ones((NY, NX))
    da = xr.DataArray(
        values.astype("float32"), dims=("time", "yh", "xh"), coords={"time": _times(n)}
    ).where(_wetmask())
    return xr.Dataset({"SW": da, "LW": 2 * da})


def _stream(**overrides) -> StreamConfig:
    base = StreamConfig(
        name="ice_flux_mean",
        store="local",
        variables=["SW", "LW"],
        time_block_mean=BLOCK,
        full_cell_variables=["SW", "LW"],
        full_cell_only=True,
    )
    return dataclasses.replace(base, **overrides)


def _config(streams, **overrides) -> PipelineConfig:
    base = PipelineConfig(
        streams=streams,
        statics=StaticsConfig(store="local", variables=[]),
        wetmask=WetmaskConfig(store="local", variable="thetao"),
        target_grid="F90",
        weights_url="local",
        output=OutputConfig(path="out.zarr", time_chunk_size=1, time_shard_size=4),
    )
    return dataclasses.replace(base, **overrides)


def test_block_mean_values_and_labels():
    ds = _source(3 * BLOCK)
    out = run.block_mean(ds, BLOCK)
    expected = np.arange(3 * BLOCK).reshape(3, BLOCK).mean(axis=1)
    np.testing.assert_allclose(out["SW"].isel(yh=1, xh=1).values, expected)
    assert list(out.time.values) == list(ds.time.values[BLOCK - 1 :: BLOCK])
    assert out["SW"].isel(yh=0, xh=0).isnull().all()
    assert "mean of 4" in out["SW"].attrs["derivation"]


def test_block_mean_propagates_nan():
    ds = _source(BLOCK)
    ds["SW"][1, 1, 1] = np.nan
    assert np.isnan(run.block_mean(ds, BLOCK)["SW"].values[0, 1, 1])


def test_block_mean_rejects_partial_block():
    with pytest.raises(AssertionError, match="whole"):
        run.block_mean(_source(BLOCK + 1), BLOCK)


@pytest.mark.parametrize(
    "n_raw, start, end, expected_blocks",
    [
        (3 * BLOCK + 2, None, None, [0, 1, 2]),
        (4 * BLOCK, "0001-01-02T06", "0001-01-04T00", [1, 2]),
    ],
)
def test_open_stream_selects_whole_blocks(tmp_path, n_raw, start, end, expected_blocks):
    path = str(tmp_path / "source.zarr")
    _source(n_raw).to_zarr(path)
    stream = _stream(store=path)
    config = _config([stream], start_time=start, end_time=end)
    ds = run.open_stream(stream, config)
    raw = _times(n_raw)
    assert list(ds.time.values) == list(
        raw[expected_blocks[0] * BLOCK : (expected_blocks[-1] + 1) * BLOCK]
    )
    labels = run.stream_output_time(stream, ds)
    assert list(labels.values) == [raw[k * BLOCK + BLOCK - 1] for k in expected_blocks]


def test_block_labels_align_with_subsample_stride(tmp_path):
    path = str(tmp_path / "source.zarr")
    _source(3 * BLOCK).to_zarr(path)
    block_stream = _stream(store=path)
    snapshot_stream = StreamConfig(
        name="snapshot", store=path, variables=["SW"], time_subsample_stride=BLOCK
    )
    config = _config([block_stream, snapshot_stream])
    times = {
        s.name: run.stream_output_time(s, run.open_stream(s, config))
        for s in (block_stream, snapshot_stream)
    }
    run._assert_time_alignment(times)


def test_process_chunk_block_mean_full_cell_only(monkeypatch):
    def identity_regridder(da, keep_attrs=False):
        return da.rename({"yh": "lat", "xh": "lon"})

    monkeypatch.setattr(run, "get_regridder", lambda *args: identity_regridder)
    stream = _stream()
    ds = _source(2 * BLOCK)
    key, out = run.process_chunk(
        xbeam.Key({"time": 2 * BLOCK}),
        ds,
        stream=stream,
        wetmask=_wetmask(),
        weights_url="local",
        target_grid_name="F90",
    )
    assert key.offsets["time"] == 2
    assert set(out.data_vars) == {"SW", "LW"}
    assert out.sizes["time"] == 2
    np.testing.assert_allclose(
        out["LW"].isel(lat=1, lon=1).values, 2 * np.array([1.5, 5.5])
    )
    assert out["SW"].isel(lat=0, lon=0).isnull().all()
    assert out["SW"].dtype == np.float32
    assert out["SW"].attrs["derivation"].startswith("mean of 4")


def test_process_chunk_rejects_offset_off_block_boundary():
    with pytest.raises(AssertionError, match="block boundary"):
        run.process_chunk(
            xbeam.Key({"time": 1}),
            _source(BLOCK),
            stream=_stream(),
            wetmask=_wetmask(),
            weights_url="local",
            target_grid_name="F90",
        )


def test_expected_output_names_full_cell_only():
    stream = _stream()
    config = _config([stream])
    assert run._expected_output_names(config, {stream.name: _source(BLOCK)}) == {
        "SW",
        "LW",
    }


def test_full_cell_only_writes_renamed_outputs(monkeypatch):
    def identity_regridder(da, keep_attrs=False):
        return da.rename({"yh": "lat", "xh": "lon"})

    monkeypatch.setattr(run, "get_regridder", lambda *args: identity_regridder)
    stream = _stream(renaming={"SW": "SW_total_area"})
    _, out = run.process_chunk(
        xbeam.Key({"time": 0}),
        _source(BLOCK),
        stream=stream,
        wetmask=_wetmask(),
        weights_url="local",
        target_grid_name="F90",
    )
    expected = {"SW_total_area", "LW"}
    assert set(out.data_vars) == expected
    config = _config([stream])
    assert run._expected_output_names(config, {stream.name: _source(BLOCK)}) == expected


def test_full_cell_only_needs_no_renaming():
    assert _stream().renaming == {}


def test_full_cell_only_requires_every_variable():
    with pytest.raises(ValueError, match="must list every variable"):
        _stream(full_cell_variables=["SW"])


def test_full_cell_without_full_cell_only_still_requires_renaming():
    with pytest.raises(ValueError, match="needs a renaming entry"):
        _stream(full_cell_only=False)


def test_block_mean_excludes_subsample_stride():
    with pytest.raises(ValueError, match="choose one"):
        _stream(time_subsample_stride=BLOCK)


def test_block_mean_excludes_midpoint_shift():
    with pytest.raises(ValueError, match="block-end"):
        _config([_stream()], shift_timestamps_to_avg_interval_midpoint=True)


def test_calving_residue_total_area_values_and_drops_components():
    ones = xr.DataArray(np.ones((NY, NX)), dims=("lat", "lon"))
    ds = xr.Dataset({"hflso": 3.0 * ones, "evs": 2e-6 * ones, "prsn": 1e-6 * ones})
    context = ChunkContext(ocean_fraction=0.5 * ones, store="local")
    spec = postprocess.POSTPROCESS["calving_residue_total_area"]()
    out = spec.fn(ds, context)
    expected = 0.5 * (
        -3.0
        + postprocess.LATENT_HEAT_VAPORIZATION * 2e-6
        - postprocess.LATENT_HEAT_FUSION * 1e-6
    )
    assert set(out.data_vars) == {"calving_residue_total_area"}
    np.testing.assert_allclose(out["calving_residue_total_area"].values, expected)
    assert out["calving_residue_total_area"].attrs["units"] == "W/m2"


def test_expected_output_names_drops_combined_components():
    stream = StreamConfig(
        name="ice_snapshot",
        store="local",
        variables=["simass", "sisnmass"],
        postprocess=["frozen_mass_total_area"],
    )
    source = xr.Dataset({v: _source(1)["SW"] for v in stream.variables})
    names = run._expected_output_names(_config([stream]), {stream.name: source})
    assert names == {"frozen_mass_total_area"}
