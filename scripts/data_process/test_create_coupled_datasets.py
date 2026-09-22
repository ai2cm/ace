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
