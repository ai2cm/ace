import numpy as np
import pytest
import xarray as xr
from coupled_dataset_utils import (
    CoupledSeaIceConfig,
    CoupledSeaSurfaceConfig,
    CoupledSurfaceTemperatureConfig,
    compute_coupled_atmosphere,
    compute_coupled_ocean,
    compute_coupled_sea_ice,
)
from create_window_avg_dataset import WindowAvgDatasetConfig

LAT = [0.0, 1.0]
LON = [0.0, 1.0]
N_ATMOS_TIMES = 41  # ten days of 6-hourly steps, endpoints inclusive
OCEAN_STRIDE = 20  # 120h expressed in 6-hourly steps
WINDOW_AVG = WindowAvgDatasetConfig(
    window_timedelta="120h", first_timestamp="2000-01-06T00:00:00"
)


def _atmos_times() -> xr.CFTimeIndex:
    return xr.date_range(
        "2000-01-01",
        periods=N_ATMOS_TIMES,
        freq="6h",
        calendar="noleap",
        use_cftime=True,
    )


def _ocean_times() -> xr.CFTimeIndex:
    # 5-daily instants coinciding with 6-hourly atmosphere timestamps
    return _atmos_times()[::OCEAN_STRIDE]


def _time_varying(values, times) -> xr.DataArray:
    """Broadcast a scalar or per-time list to a (time, lat, lon) DataArray."""
    data = np.broadcast_to(
        np.asarray(values, dtype=float).reshape(-1, 1, 1),
        (len(times), len(LAT), len(LON)),
    )
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": LAT, "lon": LON},
    )


def _static(values) -> xr.DataArray:
    data = np.broadcast_to(np.asarray(values, dtype=float), (len(LAT), len(LON))).copy()
    return xr.DataArray(data, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})


def _synthetic_atmos(land_fraction=0.0, sea_ice_fraction=0.2) -> xr.Dataset:
    times = _atmos_times()
    lfrac = _static(land_fraction)
    ifrac = _time_varying(sea_ice_fraction, times)
    ts = _time_varying(270.0 + np.arange(len(times)), times)
    return xr.Dataset(
        {
            "land_fraction": lfrac,
            "sea_ice_fraction": ifrac,
            "ocean_fraction": (1 - lfrac - ifrac).clip(0, 1),
            "surface_temperature": ts,
            "latent_heat_flux": _time_varying(3.0, times),
        }
    )


def _synthetic_ocean(
    sea_surface_fraction=1.0,
    sea_ice_fraction=None,
    ocean_sea_ice_fraction=None,
    hfds_total_area=None,
) -> xr.Dataset:
    times = _ocean_times()
    ds = xr.Dataset(
        {
            "sst": _time_varying(1.0, times),
            "hfds": _time_varying(2.0, times),
            "sea_surface_fraction": _static(sea_surface_fraction),
        }
    )
    ds["hfds"].attrs["units"] = "W/m2"
    if sea_ice_fraction is not None:
        ds["sea_ice_fraction"] = _time_varying(sea_ice_fraction, times)
    if ocean_sea_ice_fraction is not None:
        ds["ocean_sea_ice_fraction"] = _time_varying(ocean_sea_ice_fraction, times)
    if hfds_total_area is not None:
        ds["hfds_total_area"] = _time_varying(hfds_total_area, times)
        ds["hfds_total_area"].attrs = {"long_name": "native", "units": "W/m2"}
    return ds


def _synthetic_sea_ice(sea_ice_fraction=0.3, sea_surface_fraction=1.0) -> xr.Dataset:
    times = _atmos_times()
    return xr.Dataset(
        {
            "sea_ice_fraction": _time_varying(sea_ice_fraction, times),
            "sea_surface_fraction": _static(sea_surface_fraction),
        }
    )


@pytest.mark.parametrize(
    ("with_sea_ice_dataset", "ocean_has_sea_ice", "expected_ifrac"),
    [
        pytest.param(True, True, 0.3, id="sea-ice-dataset-wins"),
        pytest.param(False, True, 0.5, id="ocean-wins-over-atmosphere"),
        pytest.param(False, False, 0.2, id="atmosphere-fallback"),
    ],
)
def test_sea_ice_source_priority(
    with_sea_ice_dataset, ocean_has_sea_ice, expected_ifrac
):
    atmos = _synthetic_atmos(sea_ice_fraction=0.2)
    ocean = _synthetic_ocean(ocean_sea_ice_fraction=0.5 if ocean_has_sea_ice else None)
    sea_ice = _synthetic_sea_ice(0.3) if with_sea_ice_dataset else None
    result = compute_coupled_sea_ice(
        atmos, CoupledSeaIceConfig(), sea_ice=sea_ice, ocean=ocean
    )
    np.testing.assert_allclose(result["sea_ice_fraction"].values, expected_ifrac)


def test_window_avg_keeps_legacy_path():
    # with a window_avg configured and no separate sea ice dataset, the ocean
    # dataset's sea ice fields must be ignored: the output matches the
    # atmosphere-sourced windowed result from an ocean without sea ice fields
    ramp = np.linspace(0.0, 1.0, N_ATMOS_TIMES)
    atmos = _synthetic_atmos(sea_ice_fraction=ramp)
    config = CoupledSeaIceConfig(window_avg=WINDOW_AVG)
    result_ice_carrying_ocean = compute_coupled_sea_ice(
        atmos, config, ocean=_synthetic_ocean(ocean_sea_ice_fraction=0.5)
    )
    result_plain_ocean = compute_coupled_sea_ice(
        atmos, config, ocean=_synthetic_ocean()
    )
    xr.testing.assert_identical(result_ice_carrying_ocean, result_plain_ocean)
    # sanity check that the ocean's constant concentration was not used
    assert not np.allclose(result_ice_carrying_ocean["sea_ice_fraction"].values, 0.5)


@pytest.mark.parametrize("native_present", [True, False])
def test_ocean_sourced_native_vs_derived_concentration(native_present):
    atmos = _synthetic_atmos()
    if native_present:
        # native concentration wins over the full-cell sea ice fraction
        ocean = _synthetic_ocean(ocean_sea_ice_fraction=0.5, sea_ice_fraction=0.8)
        expected = [0.5, 0.5, 0.5]
    else:
        # derived as sea_ice_fraction / sea_surface_fraction, clipped to [0, 1]
        ocean = _synthetic_ocean(
            sea_surface_fraction=0.8, sea_ice_fraction=[0.4, 1.0, 0.4]
        )
        expected = [0.5, 1.0, 0.5]
    result = compute_coupled_sea_ice(atmos, CoupledSeaIceConfig(), ocean=ocean)
    sic_at_ocean_instants = result["ocean_sea_ice_fraction"].isel(
        time=slice(None, None, OCEAN_STRIDE)
    )
    np.testing.assert_allclose(
        sic_at_ocean_instants.values,
        np.asarray(expected).reshape(-1, 1, 1) * np.ones((1, len(LAT), len(LON))),
    )


def test_ocean_sourced_fractions_sum_to_one():
    land_fraction = [[0.0, 0.3], [1.0, 0.6]]
    sea_surface_fraction = [[1.0, 0.7], [0.0, 0.0]]
    atmos = _synthetic_atmos(land_fraction=land_fraction)
    ocean = _synthetic_ocean(
        sea_surface_fraction=sea_surface_fraction,
        ocean_sea_ice_fraction=[0.1, 0.4, 0.9],
    )
    result = compute_coupled_sea_ice(atmos, CoupledSeaIceConfig(), ocean=ocean)
    total = (
        result["land_fraction"] + result["ocean_fraction"] + result["sea_ice_fraction"]
    )
    np.testing.assert_allclose(total.values, 1.0)


def test_ocean_sourced_ffill_step_function():
    atmos = _synthetic_atmos()
    ocean = _synthetic_ocean(ocean_sea_ice_fraction=[0.1, 0.4, 0.9])
    result = compute_coupled_sea_ice(atmos, CoupledSeaIceConfig(), ocean=ocean)
    expected = np.concatenate(
        [
            np.full(OCEAN_STRIDE, 0.1),
            np.full(OCEAN_STRIDE, 0.4),
            np.full(1, 0.9),
        ]
    ).reshape(-1, 1, 1) * np.ones((1, len(LAT), len(LON)))
    np.testing.assert_allclose(result["sea_ice_fraction"].values, expected)


def test_ocean_sourced_skips_window_average():
    # the ocean-sourced output is a pure step function of the ocean instants;
    # in particular the value at each ocean instant is that instant's value,
    # not a trailing average
    atmos = _synthetic_atmos()
    ocean = _synthetic_ocean(ocean_sea_ice_fraction=[0.1, 0.4, 0.9])
    result = compute_coupled_sea_ice(atmos, CoupledSeaIceConfig(), ocean=ocean)
    np.testing.assert_allclose(
        result["sea_ice_fraction"]
        .isel(time=slice(None, None, OCEAN_STRIDE))
        .values.reshape(3, -1)[:, 0],
        [0.1, 0.4, 0.9],
    )


@pytest.mark.parametrize("cause", ["no-ocean-sea-ice-fields", "window-avg-configured"])
def test_atmosphere_fallback_disabled_raises(cause):
    atmos = _synthetic_atmos()
    if cause == "no-ocean-sea-ice-fields":
        ocean = _synthetic_ocean()
        window_avg = None
    else:
        ocean = _synthetic_ocean(ocean_sea_ice_fraction=0.5)
        window_avg = WINDOW_AVG

    disabled = CoupledSeaIceConfig(
        window_avg=window_avg, use_atmosphere_sea_ice_fraction_fallback=False
    )
    with pytest.raises(ValueError, match="use_atmosphere_sea_ice_fraction_fallback"):
        compute_coupled_sea_ice(atmos, disabled, ocean=ocean)

    # same inputs succeed with the default fallback behavior
    enabled = CoupledSeaIceConfig(window_avg=window_avg)
    compute_coupled_sea_ice(atmos, enabled, ocean=ocean)


def test_ocean_sourced_include_ts_blend():
    land_fraction = [[0.0, 1.0], [0.0, 0.0]]
    sea_surface_fraction = [[1.0, 0.0], [1.0, 1.0]]
    atmos = _synthetic_atmos(land_fraction=land_fraction)
    ocean = _synthetic_ocean(
        sea_surface_fraction=sea_surface_fraction, ocean_sea_ice_fraction=0.0
    )
    config = CoupledSeaIceConfig(include_ts=True)
    result = compute_coupled_sea_ice(atmos, config, ocean=ocean)
    ts = result["surface_temperature"].values
    steps = np.arange(N_ATMOS_TIMES)
    # over open ocean, the step function of ts at the ocean instants
    expected_ocean = 270.0 + OCEAN_STRIDE * (steps // OCEAN_STRIDE)
    np.testing.assert_allclose(ts[:, 0, 0], expected_ocean)
    np.testing.assert_allclose(
        ts[:, 1, :], np.broadcast_to(expected_ocean[:, None], (N_ATMOS_TIMES, 2))
    )
    # over the land cell, the instantaneous atmosphere value
    np.testing.assert_allclose(ts[:, 0, 1], 270.0 + steps)


@pytest.mark.parametrize("with_sea_ice_dataset", [True, False])
def test_legacy_mode_unchanged(with_sea_ice_dataset):
    atmos = _synthetic_atmos(land_fraction=0.25, sea_ice_fraction=0.3)
    if with_sea_ice_dataset:
        sea_ice = _synthetic_sea_ice(sea_ice_fraction=0.6, sea_surface_fraction=0.8)
        # sic = 0.6 / 0.8; sfrac_mod = 1 - 0.25
        expected_sic = 0.75
        expected_sfrac = 0.8
    else:
        sea_ice = None
        # sfrac = 1 - lfrac = 0.75; sic = 0.3 / 0.75
        expected_sic = 0.4
        expected_sfrac = 0.75
    result = compute_coupled_sea_ice(atmos, CoupledSeaIceConfig(), sea_ice=sea_ice)
    sfrac_mod = 0.75
    np.testing.assert_allclose(result["ocean_sea_ice_fraction"].values, expected_sic)
    np.testing.assert_allclose(result["sea_surface_fraction"].values, expected_sfrac)
    np.testing.assert_allclose(result["land_fraction"].values, 0.25)
    np.testing.assert_allclose(
        result["sea_ice_fraction"].values, expected_sic * sfrac_mod
    )
    np.testing.assert_allclose(
        result["ocean_fraction"].values, (1 - expected_sic) * sfrac_mod
    )


def _sea_surface_config() -> CoupledSeaSurfaceConfig:
    return CoupledSeaSurfaceConfig(
        surface_flux_window_avg=WindowAvgDatasetConfig(
            window_timedelta="120h",
            first_timestamp="2000-01-06T00:00:00",
            subset_names=["latent_heat_flux"],
        ),
        sst_threshold=300.0,
    )


def test_ocean_sourced_full_chain():
    land_fraction = [[0.0, 0.3], [1.0, 0.0]]
    sea_surface_fraction = [[1.0, 0.7], [0.0, 1.0]]
    atmos = _synthetic_atmos(land_fraction=land_fraction)
    ocean = _synthetic_ocean(
        sea_surface_fraction=sea_surface_fraction,
        ocean_sea_ice_fraction=[0.1, 0.4, 0.9],
    )
    config = CoupledSeaIceConfig(use_atmosphere_sea_ice_fraction_fallback=False)
    coupled_sea_ice = compute_coupled_sea_ice(atmos, config, ocean=ocean)
    coupled_ocean = compute_coupled_ocean(
        ocean, atmos, coupled_sea_ice, _sea_surface_config()
    )
    coupled_atmos = compute_coupled_atmosphere(
        atmos,
        ocean,
        coupled_ocean,
        CoupledSurfaceTemperatureConfig(how="threshold"),
    )
    total = (
        coupled_atmos["land_fraction"]
        + coupled_atmos["ocean_fraction"]
        + coupled_atmos["sea_ice_fraction"]
    )
    np.testing.assert_allclose(total.values, 1.0)
    # step-function sea ice fraction on the 6-hourly index at the wet cells
    steps = np.arange(N_ATMOS_TIMES)
    expected_sic = np.asarray([0.1, 0.4, 0.9])[steps // OCEAN_STRIDE]
    np.testing.assert_allclose(
        coupled_atmos["sea_ice_fraction"].values[:, 0, 0], expected_sic
    )
    np.testing.assert_allclose(
        coupled_atmos["sea_ice_fraction"].values[:, 0, 1], expected_sic * 0.7
    )


def test_coupled_ocean_drops_surface_temperature():
    # the coupled sea ice dataset's surface temperature (include_ts) is an
    # atmosphere-side field and must not leak into the coupled ocean dataset
    atmos = _synthetic_atmos()
    ocean = _synthetic_ocean(ocean_sea_ice_fraction=0.2)
    coupled_sea_ice = compute_coupled_sea_ice(
        atmos, CoupledSeaIceConfig(include_ts=True), ocean=ocean
    )
    assert "surface_temperature" in coupled_sea_ice.data_vars
    coupled_ocean = compute_coupled_ocean(
        ocean, atmos, coupled_sea_ice, _sea_surface_config()
    )
    assert "surface_temperature" not in coupled_ocean.data_vars


@pytest.mark.parametrize("native_present", [True, False])
def test_hfds_total_area_passthrough(native_present):
    atmos = _synthetic_atmos()
    ocean = _synthetic_ocean(
        sea_surface_fraction=0.5,
        ocean_sea_ice_fraction=0.2,
        hfds_total_area=7.0 if native_present else None,
    )
    coupled_sea_ice = compute_coupled_sea_ice(atmos, CoupledSeaIceConfig(), ocean=ocean)
    coupled_ocean = compute_coupled_ocean(
        ocean, atmos, coupled_sea_ice, _sea_surface_config()
    )
    if native_present:
        # values distinct from hfds * sea_surface_fraction pin non-recomputation
        np.testing.assert_allclose(coupled_ocean["hfds_total_area"].values, 7.0)
        assert coupled_ocean["hfds_total_area"].attrs["long_name"] == "native"
    else:
        np.testing.assert_allclose(coupled_ocean["hfds_total_area"].values, 2.0 * 0.5)
