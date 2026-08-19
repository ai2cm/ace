"""End-to-end tests for create_coupled_datasets.write_datasets_and_stats on
tiny synthetic atmosphere and ocean zarr stores in a temporary directory."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("xpartition")

import create_coupled_datasets  # noqa: E402
from coupled_dataset_utils import (  # noqa: E402
    CoupledSeaIceConfig,
    CoupledSeaSurfaceConfig,
    CoupledSurfaceTemperatureConfig,
    compute_coupled_sea_ice,
)
from create_coupled_datasets import (  # noqa: E402
    CoupledDatasetsConfig,
    CoupledInputDatasetConfig,
    CoupledStatsConfig,
    CreateCoupledDatasetsConfig,
    InputDatasetsConfig,
    InputStatsConfig,
)
from create_window_avg_dataset import WindowAvgDatasetConfig  # noqa: E402
from merge_stats import STATS_NC_FILE_NAMES  # noqa: E402
from writer_utils import OutputWriterConfig  # noqa: E402

NLAT = 4
NLON = 8
N_ATMOS_TIMES = 40  # 6-hourly steps covering ten days
FIRST_WINDOW_END = "2000-01-06T00:00:00"


def _times(start, periods, freq):
    return xr.date_range(
        start, periods=periods, freq=freq, calendar="noleap", use_cftime=True
    )


def _field(times, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    data = offset + rng.uniform(0.0, 1.0, size=(len(times), NLAT, NLON))
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat": np.linspace(-80, 80, NLAT),
            "lon": np.linspace(0, 315, NLON),
        },
    )


def _write_input_zarrs(input_dir):
    atmos_times = _times("2000-01-01T06:00:00", N_ATMOS_TIMES, "6h")
    # 5-daily snapshots at the ends of the two 120h windows
    ocean_times = _times(FIRST_WINDOW_END, 2, "120h")

    atmos = xr.Dataset(
        {
            "surface_temperature": _field(atmos_times, offset=280.0, seed=1),
            "sea_ice_fraction": _field(atmos_times, seed=2),
            "ocean_fraction": _field(atmos_times, seed=3),
            "latent_heat_flux": _field(atmos_times, offset=100.0, seed=4),
        }
    )
    atmos["land_fraction"] = _field(atmos_times, seed=5).isel(time=0, drop=True)

    ocean = xr.Dataset(
        {
            "sst": _field(ocean_times, offset=275.0, seed=6),
            "hfds": _field(ocean_times, offset=10.0, seed=7),
        }
    )
    ocean["sea_surface_fraction"] = _field(ocean_times, seed=8).isel(time=0, drop=True)

    atmos_path = str(input_dir / "atmosphere.zarr")
    ocean_path = str(input_dir / "ocean.zarr")
    atmos.to_zarr(atmos_path)
    ocean.to_zarr(ocean_path)
    return atmos_path, ocean_path


def _write_fake_stats_dir(stats_dir, var_name):
    """Write the four stats netCDFs an uncoupled input's stats directory holds."""
    stats_dir.mkdir(parents=True)
    ds = xr.Dataset({var_name: xr.DataArray(1.0)})
    ds.attrs["input_samples"] = N_ATMOS_TIMES
    for fname in STATS_NC_FILE_NAMES:
        ds.to_netcdf(str(stats_dir / fname))


def _make_config(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    atmos_path, ocean_path = _write_input_zarrs(input_dir)
    _write_fake_stats_dir(input_dir / "atmosphere-stats", "uncoupled_atmos_var")
    _write_fake_stats_dir(input_dir / "ocean-stats", "uncoupled_ocean_var")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    return CreateCoupledDatasetsConfig(
        version="v1",
        family_name="synthetic",
        output_directory=str(output_dir),
        coupled_datasets=CoupledDatasetsConfig(
            coupled_sea_ice=CoupledSeaIceConfig(
                window_avg=WindowAvgDatasetConfig(
                    window_timedelta="120h", first_timestamp=FIRST_WINDOW_END
                ),
                # this config deliberately window-averages, which routes off the
                # ocean-sourced path onto the atmosphere's own sea ice fraction
                use_atmosphere_sea_ice_fraction_fallback=True,
            ),
            coupled_ts=CoupledSurfaceTemperatureConfig(
                how="threshold", ocean_fraction_threshold=0.9
            ),
            coupled_sea_surface=CoupledSeaSurfaceConfig(
                surface_flux_window_avg=WindowAvgDatasetConfig(
                    window_timedelta="120h",
                    first_timestamp=FIRST_WINDOW_END,
                    subset_names=["latent_heat_flux"],
                ),
                sst_threshold=275.5,
            ),
            output_writer=OutputWriterConfig(n_split=1),
        ),
        input_datasets=InputDatasetsConfig(
            climate_data_type="CM4",
            stats=InputStatsConfig(
                atmosphere_dir=str(input_dir / "atmosphere-stats"),
                ocean_dir=str(input_dir / "ocean-stats"),
            ),
            atmosphere=CoupledInputDatasetConfig(
                zarr_path=atmos_path, time_chunk_size=20
            ),
            ocean=CoupledInputDatasetConfig(zarr_path=ocean_path, time_chunk_size=2),
        ),
        stats=CoupledStatsConfig(),
    )


class _DummyDaskClient:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def no_distributed_client(monkeypatch):
    """Stats computation on tiny data doesn't need a distributed cluster."""
    distributed = pytest.importorskip("distributed")
    monkeypatch.setattr(distributed, "Client", _DummyDaskClient)


def _output_paths(config):
    return [
        config.sea_ice_output_store,
        config.ocean_output_store,
        config.atmosphere_output_store,
    ]


def test_write_datasets_and_stats_end_to_end_and_resume(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.write_coupled_datasets(debug=False, subsample=False)

    for store in _output_paths(config):
        assert (
            xr.open_zarr(store).sizes["time"] > 0
        ), f"expected non-empty zarr at {store}"

    for scenario in ["uncoupled_atmosphere", "coupled_atmosphere", "ocean"]:
        for fname in STATS_NC_FILE_NAMES:
            merged = tmp_path / "outputs" / "v1-synthetic-stats" / scenario / fname
            assert merged.exists(), f"missing merged stats file {merged}"

    # A re-run must resume on the existing outputs rather than recompute:
    # fail loudly if any compute stage or writer is invoked again.
    def _fail(*args, **kwargs):
        raise AssertionError("recomputed a stage that already has outputs")

    for name in [
        "compute_coupled_sea_ice",
        "compute_coupled_ocean",
        "compute_coupled_atmosphere",
    ]:
        monkeypatch.setattr(create_coupled_datasets, name, _fail)
    monkeypatch.setattr(OutputWriterConfig, "write", _fail)

    config.write_coupled_datasets(debug=False, subsample=False)


def test_write_datasets_and_stats_debug_writes_nothing(tmp_path):
    config = _make_config(tmp_path)
    config.write_coupled_datasets(debug=True, subsample=False)

    output_dir = tmp_path / "outputs"
    assert list(output_dir.iterdir()) == []


STRIDE_STEPS = 20  # "5D" in 6-hourly steps
N_STRIDE_TIMES = 45  # two whole 5-day strides plus a trailing remainder
STRIDE_START = "0152-10-01T06:00:00"


def _write_zarr(path, times, name="sst"):
    xr.Dataset({name: _field(times, seed=9)}).to_zarr(str(path))
    return str(path)


@pytest.mark.parametrize(
    "time_stride, expected_steps", [("6h", 1), ("5D", STRIDE_STEPS)]
)
def test_time_stride_selects_every_nth_instant(tmp_path, time_stride, expected_steps):
    times = _times(STRIDE_START, N_STRIDE_TIMES, "6h")
    path = _write_zarr(tmp_path / "input.zarr", times)
    ds = CoupledInputDatasetConfig(
        zarr_path=path, time_chunk_size=10, time_stride=time_stride
    ).get_dataset()
    expected = list(times[::expected_steps])
    assert list(ds.time.values) == expected
    # the record is not a whole multiple of a 5-day stride, so its last
    # timestamp is left unselected rather than truncating the range
    assert (times[-1] in expected) == (expected_steps == 1)


def test_time_stride_composes_with_timestamp_range(tmp_path):
    times = _times(STRIDE_START, N_STRIDE_TIMES, "6h")
    path = _write_zarr(tmp_path / "input.zarr", times)
    first, last = times[2], times[-3]
    ds = CoupledInputDatasetConfig(
        zarr_path=path,
        time_chunk_size=10,
        first_timestamp=str(first),
        last_timestamp=str(last),
        time_stride="5D",
    ).get_dataset()
    in_range = times[(times >= first) & (times <= last)]
    assert list(ds.time.values) == list(in_range[::STRIDE_STEPS])
    assert ds.time.values[0] == first


def test_time_stride_none_is_todays_behavior(tmp_path):
    times = _times(STRIDE_START, N_STRIDE_TIMES, "6h")
    path = _write_zarr(tmp_path / "input.zarr", times)
    ds = CoupledInputDatasetConfig(zarr_path=path, time_chunk_size=10).get_dataset()
    assert list(ds.time.values) == list(times)


@pytest.mark.parametrize(
    "cause, time_stride, match",
    [
        ("non-uniform", "5D", "uniform input time axis"),
        ("uniform", "7h", "whole multiple"),
        ("uniform", "1h", "whole multiple"),
        ("single-timestamp", "5D", "fewer than two"),
    ],
)
def test_time_stride_invalid_raises(tmp_path, cause, time_stride, match):
    times = _times(STRIDE_START, N_STRIDE_TIMES, "6h")
    if cause == "non-uniform":
        times = times[[0, 1, 2, 5, 6]]
    elif cause == "single-timestamp":
        times = times[:1]
    path = _write_zarr(tmp_path / "input.zarr", times)
    config = CoupledInputDatasetConfig(
        zarr_path=path, time_chunk_size=10, time_stride=time_stride
    )
    with pytest.raises(ValueError, match=match):
        config.get_dataset()


def _strided_route_inputs(tmp_path):
    """A 6-hourly atmosphere store and a 6-hourly sea-surface store whose sea ice
    concentration varies every step, so a stride is what makes the emitted
    fractions 5-daily. Column 0 is all land; row 0 is ice-free open ocean."""
    times = _times(STRIDE_START, N_STRIDE_TIMES, "6h")
    steps = np.arange(len(times))
    land = np.zeros((NLAT, NLON))
    land[:, 0] = 1.0
    sic = np.broadcast_to(
        (steps / len(times)).reshape(-1, 1, 1), (len(times), NLAT, NLON)
    ).copy()
    sic[:, 0, :] = 0.0

    def _da(data, dims=("time", "lat", "lon")):
        coords = {"lat": np.linspace(-80, 80, NLAT), "lon": np.linspace(0, 315, NLON)}
        if "time" in dims:
            coords["time"] = times
        return xr.DataArray(data, dims=dims, coords=coords)

    atmos = xr.Dataset(
        {
            "land_fraction": _da(land, dims=("lat", "lon")),
            "sea_ice_fraction": _da(np.zeros_like(sic)),
            "ocean_fraction": _da(np.ones_like(sic)),
            "surface_temperature": _da(
                np.broadcast_to(
                    (270.0 + steps).reshape(-1, 1, 1), (len(times), NLAT, NLON)
                ).copy()
            ),
        }
    )
    ocean = xr.Dataset(
        {
            "sea_surface_fraction": _da(1.0 - land, dims=("lat", "lon")),
            "ocean_sea_ice_fraction": _da(sic),
            "sst": _da(np.full_like(sic, 275.0)),
        }
    )
    atmos_path = str(tmp_path / "atmos.zarr")
    ocean_path = str(tmp_path / "sea_surface.zarr")
    atmos.to_zarr(atmos_path)
    ocean.to_zarr(ocean_path)
    return atmos_path, ocean_path, times, steps, sic


def test_strided_ocean_input_gives_5daily_coherent_fractions(tmp_path):
    atmos_path, ocean_path, times, steps, sic = _strided_route_inputs(tmp_path)
    atmos = CoupledInputDatasetConfig(
        zarr_path=atmos_path, time_chunk_size=10
    ).get_dataset()
    ocean = CoupledInputDatasetConfig(
        zarr_path=ocean_path, time_chunk_size=10, time_stride="5D"
    ).get_dataset()

    result = compute_coupled_sea_ice(
        atmos, CoupledSeaIceConfig(include_ts=True), ocean=ocean
    )

    # the output keeps the atmosphere's own 6-hourly axis
    assert list(result.time.values) == list(times)
    total = (
        result["land_fraction"] + result["ocean_fraction"] + result["sea_ice_fraction"]
    )
    np.testing.assert_allclose(total.values, 1.0)

    # the fractions step on the strided cadence: each output step carries the
    # concentration of the most recent selected snapshot
    held = steps - (steps % STRIDE_STEPS)
    np.testing.assert_allclose(result["ocean_sea_ice_fraction"].values, sic[held, :, :])
    # ... and land is unaffected by the stride
    np.testing.assert_allclose(result["sea_ice_fraction"].values[:, :, 0], 0.0)

    # include_ts: piecewise constant on the stride where ocean_fraction is 1,
    # raw 6-hourly over land
    ts = result["surface_temperature"].values
    np.testing.assert_allclose(ts[:, 0, 1], 270.0 + held)
    np.testing.assert_allclose(ts[:, 0, 0], 270.0 + steps)
