import re

import numpy as np
import pytest
import xarray as xr
from compute_dataset import (
    compute_column_advective_moisture_tendency,
    compute_ocean_fraction,
    compute_pressure_thickness,
    compute_tendencies,
    validate_vertical_coarsening_indices,
    weighted_mean,
)


@pytest.mark.parametrize(
    ("size", "indices", "valid"),
    [
        pytest.param(5, [(0, 2), (2, 5)], True, id="valid"),
        pytest.param(5, [(0, 2), (1, 5)], False, id="invalid-overlapping"),
        pytest.param(5, [(0, 2), (2, 4)], False, id="invalid-incomplete"),
        pytest.param(5, [(0, 2), (2, 6)], False, id="invalid-out-of-bounds"),
    ],
)
def test_validate_vertical_coarsening_indices(size, indices, valid):
    component = "atmosphere"
    control_flag = "validate_vertical_coarsening_indices"
    if valid:
        validate_vertical_coarsening_indices(size, indices, component, control_flag)
    else:
        # Check that both the component and control flag appear in the error message.
        match = re.compile(rf"(?=.*\b{component}\b)(?=.*\b{control_flag}\b)")
        with pytest.raises(ValueError, match=match):
            validate_vertical_coarsening_indices(size, indices, component, control_flag)


@pytest.mark.parametrize("include_nans", [False, True])
def test_weighted_mean(include_nans):
    weights = xr.DataArray([1, 3], dims=["x"], attrs={"units": "Pa"})
    if include_nans:
        da = xr.DataArray([5, np.nan], dims=["x"], attrs={"units": "m"})
        expected = xr.DataArray(np.nan, attrs={"units": "m"})
    else:
        da = xr.DataArray([5, 1], dims=["x"], attrs={"units": "m"})
        expected = xr.DataArray(2, attrs={"units": "m"})

    with xr.set_options(keep_attrs=True):
        result = weighted_mean(da, weights, ["x"])
    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("ocean_fraction_present", [False, True])
def test_compute_ocean_fraction(ocean_fraction_present):
    units = "unitless"
    long_name = "fraction of grid cell area occupied by ocean"
    attrs = {"units": units, "long_name": long_name}
    other_attrs = {"units": units, "other": "a"}

    ds = xr.Dataset()
    if ocean_fraction_present:
        ds["ocean_fraction"] = xr.DataArray([0.0, 0.5, 0.0], attrs=attrs)
    ds["land_fraction"] = xr.DataArray([1.0, 0.25, 0.5], attrs=other_attrs)
    ds["sea_ice_fraction"] = xr.DataArray([np.nan, 0.25, 0.6], attrs=other_attrs)

    expected = xr.Dataset()
    expected["ocean_fraction"] = xr.DataArray([0.0, 0.5, 0.0], attrs=attrs)
    if ocean_fraction_present:
        expected["land_fraction"] = ds["land_fraction"]
        expected["sea_ice_fraction"] = ds["sea_ice_fraction"]
    else:
        expected["land_fraction"] = ds["land_fraction"]
        expected["sea_ice_fraction"] = xr.DataArray([0.0, 0.25, 0.5], attrs=other_attrs)

    with xr.set_options(keep_attrs=True):
        result = compute_ocean_fraction(
            ds, "ocean_fraction", "land_fraction", "sea_ice_fraction"
        )
    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("pressure_thickness_exists", [False, True])
def test_compute_pressure_thickness(tmp_path, pressure_thickness_exists):
    vertical_coordinate_file = tmp_path / "vertical_coordinate.nc"
    vertical_coordinate = xr.Dataset()
    vertical_coordinate["ak"] = xr.DataArray(
        [[0.0, 1.0, 3.0]],
        dims=["Time", "xaxis_1"],
        coords={"xaxis_1": [0, 1, 2]},
        attrs={"units": "Pa", "other": "a"},
    )
    vertical_coordinate["bk"] = xr.DataArray(
        [[0.0, 0.1, 0.2]],
        dims=["Time", "xaxis_1"],
        coords={"xaxis_1": [0, 1, 2]},
        attrs={"units": "unitless", "other": "a"},
    )
    vertical_coordinate.to_netcdf(vertical_coordinate_file)

    ds = xr.Dataset()
    ds["surface_pressure"] = xr.DataArray(
        [[[1000.0]]], dims=["time", "grid_yt", "grid_xt"], attrs={"units": "Pa"}
    )
    ds["pfull"] = xr.DataArray([0, 1], dims=["pfull"])
    if pressure_thickness_exists:
        ds["pressure_thickness"] = xr.DataArray(
            [[[[101.0, 102.0]]]],
            dims=["time", "grid_yt", "grid_xt", "pfull"],
            attrs={"units": "Pa", "long_name": "pressure thickness"},
        )

    if pressure_thickness_exists:
        expected = ds
    else:
        expected = xr.Dataset()
        expected["surface_pressure"] = ds["surface_pressure"]
        expected["pfull"] = ds["pfull"]
        expected["pressure_thickness"] = xr.DataArray(
            [[[[101.0, 102.0]]]],
            dims=["time", "grid_yt", "grid_xt", "pfull"],
            attrs={"units": "Pa", "long_name": "pressure thickness"},
        )

    with xr.set_options(keep_attrs=True):
        result = compute_pressure_thickness(
            ds,
            vertical_coordinate_file,
            "pfull",
            "surface_pressure",
            "pressure_thickness",
            vertical_coordinate_file_is_local=True,
        )
    xr.testing.assert_identical(result, expected)


def test_compute_tendencies():
    name = "a"
    tendency_name = f"tendency_of_{name}"
    attrs = {"units": "m", "long_name": "long name of a"}
    periods = 4
    timestep_seconds = 21600.0
    times = xr.date_range("2000", freq="6h", periods=periods)
    dims = ["time"]
    coords = {"time": times}
    da = xr.DataArray(range(periods), dims=dims, coords=coords, name=name, attrs=attrs)
    # Include an extra attribute on time coordinate to ensure it does not end up
    # on the computed tendency variable.
    da.time.attrs["axis"] = "T"
    ds = da.to_dataset()

    expected = xr.Dataset()
    expected[name] = da
    expected[tendency_name] = da.diff("time") / timestep_seconds
    expected[tendency_name].attrs["units"] = "m/s"
    expected[tendency_name].attrs["long_name"] = "time derivative of long name of a"

    with xr.set_options(keep_attrs=True):
        result = compute_tendencies(ds, [name], "time")
    xr.testing.assert_identical(result, expected)


def test_compute_column_advective_moisture_tendency():
    ds = xr.Dataset()
    ds["pwat_tendency"] = xr.DataArray(
        [np.nan, 2.0, 3.0],
        dims=["time"],
        attrs={"units": "kg/m^2/s", "long_name": "tendency of total water path"},
    )
    ds["latent_heat_flux"] = xr.DataArray(
        [2.0, 4.0, 6.0],
        dims=["time"],
        attrs={"units": "W/m^2", "long_name": "latent heat flux"},
    )
    ds["precip"] = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=["time"],
        attrs={"units": "kg/m**2/s", "long_name": "precipitation rate"},
    )
    latent_heat_of_vaporization = 2.0

    expected = xr.Dataset()
    expected["pwat_tendency"] = ds["pwat_tendency"]
    expected["latent_heat_flux"] = ds["latent_heat_flux"]
    expected["precip"] = ds["precip"]
    expected["pwat_tendency_due_to_advection"] = xr.DataArray(
        [np.nan, 2.0, 3.0],
        dims=["time"],
        attrs={
            "units": "kg/m^2/s",
            "long_name": "tendency of total water path due to advection",
        },
    )
    with xr.set_options(keep_attrs=True):
        result = compute_column_advective_moisture_tendency(
            ds,
            pwat_tendency="pwat_tendency",
            latent_heat_flux="latent_heat_flux",
            precip="precip",
            latent_heat_of_vaporization=latent_heat_of_vaporization,
        )
    xr.testing.assert_identical(result, expected)
