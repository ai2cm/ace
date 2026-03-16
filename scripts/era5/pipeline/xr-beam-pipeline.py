import argparse
import datetime
import logging
from typing import Sequence

import apache_beam as beam
import numpy as np
import pandas as pd
import xarray as xr
import xarray_beam as xbeam
import xesmf as xe
from apache_beam.options.pipeline_options import PipelineOptions
from obstore.store import from_url
from zarr.storage import ObjectStore

GRAVITY = 9.80665
DENSITY_OF_LIQUID_WATER = 1000.0  # kg/m**3
TIME_STEP = 6  # hours between output timesteps
DEFAULT_OUTPUT_GRID = "F90"
# The input data is on the L137 ECMWF grid. See
# https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions.
# The indices below are chosen for closest alignment with the ACE vertical
# grid defined in Table 2 of https://arxiv.org/pdf/2310.02074.pdf except
# that the uppermost layer uses the higher model top of ECMWF model.
N_INPUT_LAYERS = 137  # this is the number of full layers, not interfaces
DEFAULT_OUTPUT_LAYER_INDICES = [0, 48, 67, 79, 90, 100, 109, 119, 137]

OUTPUT_PRESSURE_LEVELS = [1000, 850, 700, 500, 250, 200, 100, 50, 10]
OUTPUT_PRESSURE_LEVELS_GEOPOTENTIAL = [1000, 850, 700, 500, 300, 250, 200, 100, 50, 10]

# Gaussian grid specs: name -> N (grid number; nlat=2N, nlon=4N)
GAUSSIAN_GRID_N = {
    "F22.5": 22.5,
    "F90": 90,
    "F360": 360,
}

URL_FULL_37 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
URL_MODEL_LEVEL = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
URL_CO2 = "gs://vcm-ml-raw-flexible-retention/2026-03-12-co2-annual-mean-for-era5.zarr"

# Variables to read from each source
FULL_37_MEAN_FLUX_VARS = [
    "mean_top_downward_short_wave_radiation_flux",
    "mean_top_net_short_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_sensible_heat_flux",
    "mean_surface_latent_heat_flux",
    "mean_total_precipitation_rate",
    "mean_vertically_integrated_moisture_divergence",
    "mean_snowfall_rate",
    "mean_top_net_short_wave_radiation_flux_clear_sky",
    "mean_top_net_long_wave_radiation_flux_clear_sky",
    "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "mean_surface_net_short_wave_radiation_flux_clear_sky",
    "mean_surface_downward_long_wave_radiation_flux_clear_sky",
    "mean_surface_net_long_wave_radiation_flux_clear_sky",
    "mean_snowfall_rate",
    "mean_runoff_rate",
    "mean_eastward_gravity_wave_surface_stress",
    "mean_eastward_turbulent_surface_stress",
    "mean_northward_gravity_wave_surface_stress",
    "mean_northward_turbulent_surface_stress",
]

FULL_37_SURFACE_ANALYSIS_VARS = [
    "sea_ice_cover",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "snow_depth",
    "snow_density",
    "sea_surface_temperature",
    "skin_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
]

FULL_37_INVARIANT_VARS = [
    "land_sea_mask",
    "geopotential_at_surface",
    "soil_type",
]

FULL_37_PRESSURE_LEVEL_VARS = [
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]

MODEL_LEVEL_3D_VARS = [
    "temperature",
    "specific_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "u_component_of_wind",
    "v_component_of_wind",
]
FULL_37_MODEL_LEVEL_SURFACE_VARS = [
    "surface_pressure",
    "mean_sea_level_pressure",
    "skin_temperature",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# Soil type definitions from the ECMWF documentation: https://codes.ecmwf.int/grib/param-db/43
# undefined is not part of the defintions, but it appears to be the fill value.
# Some cells with land_fraction > 0 have this value, so it still seems relevant
# to track.
SOIL_TYPES = {
    "undefined": 0,
    "coarse": 1,
    "medium": 2,
    "medium_fine": 3,
    "fine": 4,
    "very_fine": 5,
    "organic": 6,
    "tropical_organic": 7,
}

RENAME_PRESSURE_LEVEL = {
    **{f"specific_humidity_{p}": f"Q{p}" for p in OUTPUT_PRESSURE_LEVELS},
    **{f"temperature_{p}": f"TMP{p}" for p in OUTPUT_PRESSURE_LEVELS},
    **{f"u_component_of_wind_{p}": f"UGRD{p}" for p in OUTPUT_PRESSURE_LEVELS},
    **{f"v_component_of_wind_{p}": f"VGRD{p}" for p in OUTPUT_PRESSURE_LEVELS},
    **{f"geopotential_{p}": f"h{p}" for p in OUTPUT_PRESSURE_LEVELS_GEOPOTENTIAL},
}

DESIRED_ATTRS = {
    "DSWRFtoa": {"long_name": "Downward SW radiative flux at TOA", "units": "W/m**2"},
    "USWRFtoa": {"long_name": "Upward SW radiative flux at TOA", "units": "W/m**2"},
    "ULWRFtoa": {"long_name": "Upward LW radiative flux at TOA", "units": "W/m**2"},
    "DSWRFsfc": {
        "long_name": "Downward SW radiative flux at surface",
        "units": "W/m**2",
    },
    "USWRFsfc": {"long_name": "Upward SW radiative flux at surface", "units": "W/m**2"},
    "DLWRFsfc": {
        "long_name": "Downward LW radiative flux at surface",
        "units": "W/m**2",
    },
    "ULWRFsfc": {"long_name": "Upward LW radiative flux at surface", "units": "W/m**2"},
    "UCSWRFtoa": {
        "long_name": "Upward SW radiative flux at TOA assuming clear sky",
        "units": "W/m**2",
    },
    "UCLWRFtoa": {
        "long_name": "Upward LW radiative flux at TOA assuming clear sky",
        "units": "W/m**2",
    },
    "DCSWRFsfc": {
        "long_name": "Downward SW radiative flux at surface assuming clear sky",
        "units": "W/m**2",
    },
    "UCSWRFsfc": {
        "long_name": "Upward SW radiative flux at surface assuming clear sky",
        "units": "W/m**2",
    },
    "DCLWRFsfc": {
        "long_name": "Downward LW radiative flux at surface assuming clear sky",
        "units": "W/m**2",
    },
    "UCLWRFsfc": {
        "long_name": "Upward LW radiative flux at surface assuming clear sky",
        "units": "W/m**2",
    },
    "LHTFLsfc": {"long_name": "Latent heat flux", "units": "W/m**2"},
    "SHTFLsfc": {"long_name": "Sensible heat flux", "units": "W/m**2"},
    "PRATEsfc": {"long_name": "Surface precipitation rate", "units": "kg/m**2/s"},
    "tendency_of_total_water_path_due_to_advection": {
        "long_name": "Tendency of total water path due to advection",
        "units": "kg/m**2/s",
    },
    "HGTsfc": {"long_name": "Topography height", "units": "m"},
    "land_fraction": {"long_name": "land fraction"},
    "sea_ice_fraction": {"long_name": "sea ice fraction"},
    "ocean_fraction": {"long_name": "ocean fraction"},
    "PRESsfc": {"long_name": "Surface pressure", "units": "Pa"},
    "TMP2m": {"long_name": "2m air temperature", "units": "K"},
    "Q2m": {"long_name": "2m specific humidity", "units": "kg/kg"},
    "DPT2m": {"long_name": "2m dewpoint temperature", "units": "K"},
    "UGRD10m": {"long_name": "10m U component of wind", "units": "m/s"},
    "VGRD10m": {"long_name": "10m V component of wind", "units": "m/s"},
    "runoff_flux": {"long_name": "Runoff flux", "units": "kg/m**2/s"},
    "total_frozen_precipitation_rate": {
        "long_name": "Total frozen precipitation rate",
        "units": "kg/m**2/s",
    },
    "eastward_surface_stress": {
        "long_name": "Eastward surface stress",
        "units": "N/m**2",
    },
    "northward_surface_stress": {
        "long_name": "Northward surface stress",
        "units": "N/m**2",
    },
    "merged_sea_surface_and_skin_temperature": {
        "long_name": "Merged sea surface and skin temperature",
        "units": "K",
    },
    "surface_snow_amount": {
        "long_name": "Surface snow amount",
        "units": "kg/m**2",
    },
    **{
        f"{soil_type}_soil_type_fraction": {
            "long_name": f"Fraction of {soil_type} soil type",
            "units": "fraction",
        }
        for soil_type in SOIL_TYPES.keys()
    },
}

VARIABLES_WITH_SOME_MISSING_VALUES = [
    "sea_ice_cover",
    "sea_surface_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
]

# ---------------------------------------------------------------------------
# Regridding utilities
# ---------------------------------------------------------------------------


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Compute cell boundaries from centers, clipping to [lo, hi]."""
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    bounds = np.concatenate([[lo], midpoints, [hi]])
    return bounds


def _gaussian_latitudes(n: int | float) -> np.ndarray:
    """Compute Gaussian grid latitudes for grid number N (2N latitudes).

    Returns latitudes in degrees, sorted south-to-north.
    These are the roots of the Legendre polynomial P_{2N}(sin(lat)).
    """
    from numpy.polynomial.legendre import leggauss

    nlat = round(2 * n)
    x, _ = leggauss(nlat)
    lat = np.degrees(np.arcsin(x))
    return np.sort(lat)


def _make_target_grid(output_grid: str) -> xr.Dataset:
    """Create Gaussian target grid dataset for xESMF regridding."""
    n = GAUSSIAN_GRID_N[output_grid]
    lat = _gaussian_latitudes(n)
    nlon = round(4 * n)
    dlon = 360.0 / nlon
    # Longitude centers offset by half a grid spacing (matching old pipeline)
    lon = np.linspace(dlon / 2, 360 - dlon / 2, nlon)
    lat_b = _cell_bounds(lat, -90, 90)
    lon_b = _cell_bounds(lon, 0, 360)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], lat_b),
            "lon_b": (["lon_b"], lon_b),
        }
    )


def _make_source_grid() -> xr.Dataset:
    """Create 0.25° regular lat-lon source grid dataset for xESMF."""
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 359.75, 1440)
    lat_b = _cell_bounds(lat, -90, 90)
    lon_b = _cell_bounds(lon, -0.125, 360 - 0.125)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], lat_b),
            "lon_b": (["lon_b"], lon_b),
        }
    )


# Global cache for regridder (created once per worker process)
_REGRIDDER_CACHE = {}


def _get_regridder(output_grid: str):
    """Get or create a cached xESMF regridder."""
    if output_grid not in _REGRIDDER_CACHE:
        src = _make_source_grid()
        dst = _make_target_grid(output_grid)
        _REGRIDDER_CACHE[output_grid] = xe.Regridder(
            src, dst, "conservative", periodic=True
        )
    return _REGRIDDER_CACHE[output_grid]


def _regrid(
    ds: xr.Dataset, output_grid: str, keep_attrs: bool = True, **other_regridder_kwargs
) -> xr.Dataset:
    """Regrid a dataset from 0.25° regular lat-lon to a Gaussian target grid."""
    regridder = _get_regridder(output_grid)
    # Rename coords to lat/lon for xESMF
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    # Ensure south-to-north
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sortby("lat")
    regridded = regridder(ds, keep_attrs=keep_attrs, **other_regridder_kwargs)
    # Rename back to latitude/longitude for output consistency
    regridded = regridded.rename({"lat": "latitude", "lon": "longitude"})
    return regridded


# ---------------------------------------------------------------------------
# Physics utilities
# ---------------------------------------------------------------------------


def _saturation_vapor_pressure(t: xr.DataArray) -> xr.DataArray:
    """https://metview.readthedocs.io/en/latest/api/functions/saturation_vapour_pressure.html"""  # noqa: E501
    a1 = 611.21
    a2 = 273.16
    a3 = 17.502
    a4 = 32.19
    return a1 * np.exp(a3 * (t - a2) / (t - a4))


def _specific_humidity_from_dewpoint(
    dewpoint: xr.DataArray, pressure: xr.DataArray
) -> xr.DataArray:
    """https://metview.readthedocs.io/en/latest/api/functions/specific_humidity_from_dewpoint.html
    https://metview.readthedocs.io/en/latest/api/functions/vapour_pressure.html"""
    ewsat = _saturation_vapor_pressure(dewpoint)
    eps = 0.621981
    result = eps * ewsat / (pressure - (1 - eps) * ewsat)
    result.attrs = {"units": "kg / kg", "long_name": "Specific humidity"}
    return result


def _to_geopotential_height(geopotential: xr.DataArray) -> xr.DataArray:
    output = geopotential / GRAVITY
    output.attrs = {
        "long_name": "Geopotential height",
        "units": "m",
        "standard_name": "geopotential_height",
    }
    return output


def _to_merged_sea_surface_and_skin_temperature(
    sea_surface_temperature: xr.DataArray,
    skin_temperature: xr.DataArray,
    ocean_fraction: xr.DataArray,
) -> xr.DataArray:
    """Merge the sea surface and skin temperature into a single variable.

    Note this is meant to be called after regridding, which is the only time we
    define an ocean_fraction. Our criteria for merging is based on how we merge
    the prescribed SST and land and sea ice temperature in ACE using
    interpolate=False. If the ocean fraction is less than 0.5, we use the skin
    temperature. If the ocean fraction is greater than or equal to 0.5, we use
    the sea surface temperature. There are some edge cases where the ocean
    fraction is greater than or equal to 0.5, but the sea surface temperature is
    undefined; in those circumstances, we fall back to using the skin
    temperature.
    """
    land_and_sea_ice_mask = (ocean_fraction < 0.5) | sea_surface_temperature.isnull()
    output = xr.where(land_and_sea_ice_mask, skin_temperature, sea_surface_temperature)
    output.attrs = {
        "long_name": "Merged sea surface and skin temperature",
        "units": "K",
    }
    return output


def _to_surface_snow_amount(snow_depth: xr.DataArray) -> xr.DataArray:
    """Convert the snow depth to a surface snow amount in kg/m**2.

    The snow depth in ERA5 is in meters of liquid water equivalent; we convert
    it to a mass per unit area by multiplying by the density of liquid water.
    See this page for more information: https://codes.ecmwf.int/grib/param-db/141
    """
    output = DENSITY_OF_LIQUID_WATER * snow_depth
    output.attrs = {"long_name": "Surface snow amount", "units": "kg/m**2"}
    return output


def _to_surface_snow_area_fraction(
    snow_depth: xr.DataArray, snow_density: xr.DataArray
) -> xr.DataArray:
    """See Guidelines Section 11 on this page for the ERA5 formula for the
    surface snow area fraction: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#heading-Parameterlistings"""
    output = (DENSITY_OF_LIQUID_WATER * snow_depth / snow_density) / 0.1
    output = xr.where(output > 1, 1, output)
    output.attrs = {"long_name": "Surface snow area fraction", "units": "fraction"}
    return output


def _to_surface_snow_thickness(
    surface_snow_amount: xr.DataArray,
    snow_density: xr.DataArray,
    surface_snow_area_fraction: xr.DataArray,
) -> xr.DataArray:
    """See Guidelines Section 11 on this page for the ERA5 formula for the
    physical depth of snow: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#heading-Parameterlistings"""
    output = surface_snow_amount / (snow_density * surface_snow_area_fraction)
    output = output.fillna(0.0)
    output.attrs = {"long_name": "Surface snow thickness", "units": "m"}
    return output


def _isclose(
    a: xr.DataArray | int | float, b: xr.DataArray | int | float, **kwargs
) -> xr.DataArray:
    """Check if inputs are close, with optional keyword arguments.

    Accepts the same keyword arguments as np.isclose.
    """
    return xr.apply_ufunc(
        np.isclose, a, b, kwargs=kwargs, dask="parallelized", output_dtypes=[bool]
    )


def _to_soil_type_fractions(soil_type: xr.DataArray) -> xr.Dataset:
    """Convert the soil type to a dataset of fractions via one-hot encoding.

    We use an abolute tolerance of 1.0e-3 due to floating point imprecision.
    """
    output = xr.Dataset()
    for soil_type_name, soil_type_id in SOIL_TYPES.items():
        name = f"{soil_type_name}_soil_type_fraction"
        output[name] = _isclose(soil_type, soil_type_id, atol=1.0e-3, rtol=0.0).astype(
            np.float32
        )
        output[name].attrs = DESIRED_ATTRS[name]
    return output


# ---------------------------------------------------------------------------
# Data opening
# ---------------------------------------------------------------------------


def _make_zarr_store(url: str, read_only: bool = True):
    """Create a zarr store from a URL using obstore. If local, just return the path."""
    if url.startswith("gs://"):
        return ObjectStore(from_url(url), read_only=read_only)
    else:
        return url


def open_full_37(variables, time_slice) -> xr.Dataset:
    """Open variables from the full_37 ARCO-ERA5 store."""
    ds = xr.open_zarr(_make_zarr_store(URL_FULL_37), chunks=None)
    ds = ds[variables]
    _check_time_bounds(ds, time_slice)
    ds = ds.sel(time=time_slice)
    return ds


def open_model_level(variables, time_slice) -> xr.Dataset:
    """Open variables from the model-level ARCO-ERA5 store."""
    ds = xr.open_zarr(_make_zarr_store(URL_MODEL_LEVEL), chunks=None)
    ds = ds[variables]
    _check_time_bounds(ds, time_slice)
    ds = ds.sel(time=time_slice)
    return ds


def open_co2_dataset(start_time, end_time) -> xr.Dataset:
    co2 = xr.open_zarr(_make_zarr_store(URL_CO2), chunks=None)
    ds_start = pd.Timestamp(co2.time.values[0])
    ds_stop = pd.Timestamp(co2.time.values[-1])
    assert start_time >= ds_start, "CO2 dataset time start out of bounds"
    assert end_time <= ds_stop, "CO2 dataset time stop out of bounds"
    co2 = co2.sel(time=slice(start_time, end_time))
    co2 = co2.load()
    return co2


def _check_time_bounds(ds, time_slice):
    ds_start = pd.Timestamp(ds.time.min().values.item())
    ds_stop = pd.Timestamp(ds.time.max().values.item())
    desired_start = pd.Timestamp(time_slice.start)
    desired_stop = pd.Timestamp(time_slice.stop)
    assert desired_start >= ds_start, f"Dataset time start out of bounds"
    assert desired_stop <= ds_stop, f"Dataset time stop out of bounds"


def _check_data_validity(ds):
    """Check for time slices with an unexpected number of missing values.

    We cannot rely solely on the valid_stop_time and valid_stop_time_era5t
    attributes, since they may be updated prior to all the variables:
    https://github.com/google-research/arco-era5/issues/128.
    """
    for name, da in ds.data_vars.items():
        reduction_dims = [dim for dim in da.dims if dim != "time"]
        if name in VARIABLES_WITH_SOME_MISSING_VALUES:
            # Some 2D variables are masked, so we only raise if all values are
            # missing in a time slice.
            missing_indices = da.isnull().all(reduction_dims)
        else:
            # For all other variables, we use a stricter check and raise if any
            # values are missing in a time slice.
            missing_indices = da.isnull().any(reduction_dims)

        # Accomdate variables where the time dimension has been squeezed out,
        # like in the case of the invariant dataset.
        if "time" not in missing_indices.dims:
            missing_indices = missing_indices.expand_dims("time")

        if missing_indices.any():
            missing_times = da.time.isel(time=missing_indices).to_numpy()
            raise ValueError(f"Missing values in {name!r} at times: {missing_times}.")


# ---------------------------------------------------------------------------
# Stream 1: Mean Flux processing
# ---------------------------------------------------------------------------


def _process_mean_flux(ds: xr.Dataset, output_grid: str) -> xr.Dataset:
    """Compute derived fluxes from hourly mean flux data (already time-averaged)."""
    logging.info("Processing mean flux data")
    xr.set_options(keep_attrs=True)
    output = xr.Dataset()

    # All-sky radiative fluxes
    output["DSWRFtoa"] = ds["mean_top_downward_short_wave_radiation_flux"]
    output["USWRFtoa"] = (
        ds["mean_top_downward_short_wave_radiation_flux"]
        - ds["mean_top_net_short_wave_radiation_flux"]
    )
    output["ULWRFtoa"] = -ds["mean_top_net_long_wave_radiation_flux"]
    output["DSWRFsfc"] = ds["mean_surface_downward_short_wave_radiation_flux"]
    output["USWRFsfc"] = (
        ds["mean_surface_downward_short_wave_radiation_flux"]
        - ds["mean_surface_net_short_wave_radiation_flux"]
    )
    output["DLWRFsfc"] = ds["mean_surface_downward_long_wave_radiation_flux"]
    output["ULWRFsfc"] = (
        ds["mean_surface_downward_long_wave_radiation_flux"]
        - ds["mean_surface_net_long_wave_radiation_flux"]
    )

    # Clear-sky radiative fluxes
    output["UCSWRFtoa"] = (
        ds["mean_top_downward_short_wave_radiation_flux"]
        - ds["mean_top_net_short_wave_radiation_flux_clear_sky"]
    )
    output["UCLWRFtoa"] = -ds["mean_top_net_long_wave_radiation_flux_clear_sky"]
    output["DCSWRFsfc"] = ds[
        "mean_surface_downward_short_wave_radiation_flux_clear_sky"
    ]
    output["UCSWRFsfc"] = (
        ds["mean_surface_downward_short_wave_radiation_flux_clear_sky"]
        - ds["mean_surface_net_short_wave_radiation_flux_clear_sky"]
    )
    output["DCLWRFsfc"] = ds["mean_surface_downward_long_wave_radiation_flux_clear_sky"]
    output["UCLWRFsfc"] = (
        ds["mean_surface_downward_long_wave_radiation_flux_clear_sky"]
        - ds["mean_surface_net_long_wave_radiation_flux_clear_sky"]
    )

    output["SHTFLsfc"] = -ds["mean_surface_sensible_heat_flux"]
    output["LHTFLsfc"] = -ds["mean_surface_latent_heat_flux"]
    output["PRATEsfc"] = ds["mean_total_precipitation_rate"]
    output["total_frozen_precipitation_rate"] = ds["mean_snowfall_rate"]
    output["runoff_flux"] = ds["mean_runoff_rate"]
    output["tendency_of_total_water_path_due_to_advection"] = -ds[
        "mean_vertically_integrated_moisture_divergence"
    ]

    output["eastward_surface_stress"] = (
        ds["mean_eastward_gravity_wave_surface_stress"]
        + ds["mean_eastward_turbulent_surface_stress"]
    )
    output["northward_surface_stress"] = (
        ds["mean_northward_gravity_wave_surface_stress"]
        + ds["mean_northward_turbulent_surface_stress"]
    )

    for name, attrs in DESIRED_ATTRS.items():
        if name in output:
            output[name] = output[name].assign_attrs(**attrs)

    regridded = _regrid(output, output_grid)
    regridded = regridded.drop_vars(["latitude", "longitude"])
    return regridded


def _average_hourly_to_6hourly(ds: xr.Dataset) -> xr.Dataset:
    """Average 6 consecutive hourly values to produce one 6-hourly output.

    For output time T, average hours [T-5h, T-4h, T-3h, T-2h, T-1h, T].
    Input chunk has exactly 6 hourly timesteps; output is labeled by last time.
    """
    xr.set_options(keep_attrs=True)
    output_time = ds.time.values[-1:]  # label by end of averaging window
    averaged = ds.mean("time", keepdims=True)
    averaged["time"] = output_time
    return averaged


def process_mean_flux(
    key, ds, output_grid=DEFAULT_OUTPUT_GRID, check_data_validity=False
):
    if check_data_validity:
        _check_data_validity(ds)
    averaged = _average_hourly_to_6hourly(ds)
    output = _process_mean_flux(averaged, output_grid)
    # Convert from hourly input offset to 6-hourly output offset
    output_time_offset = key.offsets["time"] // TIME_STEP
    new_key = key.replace(
        offsets={"time": output_time_offset, "latitude": 0, "longitude": 0},
        vars=frozenset(output.keys()),
    )
    return new_key, output


# ---------------------------------------------------------------------------
# Stream 2: Surface analysis / invariant
# ---------------------------------------------------------------------------


def _process_invariant(
    ds: xr.Dataset, output_grid: str, check_data_validity: bool = False
) -> xr.Dataset:
    """Process invariant fields (land_sea_mask, geopotential_at_surface, soil_type)."""
    logging.info("Processing invariant data")
    if check_data_validity:
        _check_data_validity(ds)
    output = xr.Dataset()
    output["HGTsfc"] = ds["geopotential_at_surface"] / GRAVITY
    output["land_fraction"] = ds["land_sea_mask"]
    soil_type_fractions = _to_soil_type_fractions(ds["soil_type"])
    output = output.merge(soil_type_fractions)
    regridded = _regrid(output, output_grid)
    return regridded


def _process_surface_analysis(
    ds: xr.Dataset, invariant_ds: xr.Dataset, output_grid: str
) -> xr.Dataset:
    """Process surface analysis fields and combine with invariant."""
    logging.info("Processing surface analysis data")
    xr.set_options(keep_attrs=True)
    output = xr.Dataset()
    output["sea_ice_fraction"] = ds["sea_ice_cover"].fillna(0.0)

    output["soil_moisture_0"] = ds["volumetric_soil_water_layer_1"]
    output["soil_moisture_1"] = ds["volumetric_soil_water_layer_2"]
    output["soil_moisture_2"] = ds["volumetric_soil_water_layer_3"]
    output["soil_moisture_3"] = ds["volumetric_soil_water_layer_4"]
    output["soil_temperature_0"] = ds["soil_temperature_level_1"]
    output["soil_temperature_1"] = ds["soil_temperature_level_2"]
    output["soil_temperature_2"] = ds["soil_temperature_level_3"]
    output["soil_temperature_3"] = ds["soil_temperature_level_4"]

    output["surface_snow_amount"] = _to_surface_snow_amount(ds["snow_depth"])
    output["surface_snow_area_fraction"] = _to_surface_snow_area_fraction(
        ds["snow_depth"], ds["snow_density"]
    )
    output["surface_snow_thickness"] = _to_surface_snow_thickness(
        output["surface_snow_amount"],
        ds["snow_density"],
        output["surface_snow_area_fraction"],
    )

    regridded = _regrid(output, output_grid)

    # Handle regridding the sea surface temperature and wave heights using
    # adaptive masking to ensure coastal points have a defined value.
    regridded["sea_surface_temperature"] = _regrid(
        ds["sea_surface_temperature"], output_grid, skipna=True, na_thres=1.0
    )
    regridded["significant_height_of_combined_wind_waves_and_swell"] = _regrid(
        ds["significant_height_of_combined_wind_waves_and_swell"],
        output_grid,
        skipna=True,
        na_thres=1.0,
    )
    # For convenience we fill missing wave heights with 0.0.
    regridded["significant_height_of_combined_wind_waves_and_swell"] = regridded[
        "significant_height_of_combined_wind_waves_and_swell"
    ].fillna(0.0)

    regridded = regridded.drop_vars(["latitude", "longitude"])
    invariant_ds = invariant_ds.drop_vars(["latitude", "longitude"])

    regridded["ocean_fraction"] = (
        1 - invariant_ds["land_fraction"] - regridded["sea_ice_fraction"]
    )

    # Correct negative ocean fraction
    negative_ocean = xr.where(regridded.ocean_fraction < 0, regridded.ocean_fraction, 0)
    regridded["ocean_fraction"] -= negative_ocean
    regridded["sea_ice_fraction"] += negative_ocean

    for name in ["ocean_fraction", "sea_ice_fraction"]:
        regridded[name] = regridded[name].assign_attrs(DESIRED_ATTRS[name])

    regridded_skin_temperature = _regrid(ds["skin_temperature"], output_grid)
    regridded["merged_sea_surface_and_skin_temperature"] = (
        _to_merged_sea_surface_and_skin_temperature(
            regridded["sea_surface_temperature"],
            regridded_skin_temperature,
            regridded["ocean_fraction"],
        )
    )
    return regridded


def process_surface_analysis(
    key,
    ds,
    invariant_ds=None,
    output_grid=DEFAULT_OUTPUT_GRID,
    check_data_validity: bool = False,
):
    if check_data_validity:
        _check_data_validity(ds)
    output = _process_surface_analysis(ds, invariant_ds, output_grid)
    new_key = key.replace(vars=frozenset(output.keys()))
    return new_key, output


# ---------------------------------------------------------------------------
# Stream 3: Pressure levels
# ---------------------------------------------------------------------------


def _process_pressure_level_data(ds: xr.Dataset, output_grid: str) -> xr.Dataset:
    """Select pressure levels and regrid."""
    select_levels = xr.Dataset()
    for name in ds.data_vars:
        if "level" in ds[name].dims:
            if name == "geopotential":
                pressure_levels = OUTPUT_PRESSURE_LEVELS_GEOPOTENTIAL
            else:
                pressure_levels = OUTPUT_PRESSURE_LEVELS
            logging.info(f"Subselecting desired pressure levels for {name}")
            for pressure in pressure_levels:
                out_name = f"{name}_{pressure}"
                select_levels[out_name] = ds[name].sel(level=pressure)
                if name == "geopotential":
                    select_levels[out_name] = _to_geopotential_height(
                        select_levels[out_name]
                    )
                    select_levels[out_name].attrs["long_name"] += f" at {pressure} hPa"
                else:
                    select_levels[out_name].attrs["long_name"] = (
                        ds[name].attrs.get("long_name", name) + f" at {pressure} hPa"
                    )
        else:
            select_levels[name] = ds[name]

    select_levels = select_levels.drop_vars("level", errors="ignore")
    regridded = _regrid(select_levels, output_grid)
    regridded = regridded.rename(RENAME_PRESSURE_LEVEL)
    regridded = regridded.drop_vars(["latitude", "longitude"])
    return regridded


def process_pressure_level_data(
    key, ds, output_grid=DEFAULT_OUTPUT_GRID, check_data_validity: bool = False
):
    if check_data_validity:
        _check_data_validity(ds)
    output = _process_pressure_level_data(ds, output_grid)
    new_key = key.replace(
        offsets={"time": key.offsets["time"], "latitude": 0, "longitude": 0},
        vars=frozenset(output.keys()),
    )
    return new_key, output


# ---------------------------------------------------------------------------
# Stream 4: Model level (most complex)
# ---------------------------------------------------------------------------


def _get_ak_bk(ds_model_level: xr.Dataset) -> tuple:
    """Extract ak/bk from GRIB_pv attribute on a model-level variable."""
    for name in ds_model_level.data_vars:
        if "GRIB_pv" in ds_model_level[name].attrs:
            pv = ds_model_level[name].attrs["GRIB_pv"]
            break
    else:
        raise ValueError("No variable with GRIB_pv attribute found in model-level data")
    ak = np.array(pv[: N_INPUT_LAYERS + 1])
    bk = np.array(pv[N_INPUT_LAYERS + 1 :])

    # Treat the top-most layer interface as the midpoint between the top-most
    # and second-to-top-most model-native levels to avoid an implicit model top
    # pressure of 0.0 Pa. This comes out to essentially setting the model top
    # pressure to 1.0 Pa. The IFS does not use a finite-volume vertical
    # coordinate so these vertical coordinates have somewhat of an artificial
    # meaning to begin with.
    ak[0] = (ak[0] + ak[1]) / 2.0
    return ak, bk


def _compute_layer_thicknesses(
    ak: np.ndarray, bk: np.ndarray, surface_pressure: xr.DataArray
) -> xr.DataArray:
    """Compute pressure thickness dp for each model level.

    dp(k) = [ak(k+1) + bk(k+1)*ps] - [ak(k) + bk(k)*ps]
    where k is the interface index (0 at top, 138 at surface).
    Level k spans from interface k to interface k+1.
    """
    # ak, bk have shape (138,) for 137 levels + 1
    # dp has shape (137,) for 137 levels
    # ak[k+1] - ak[k] gives the "a" contribution to thickness of level k
    dak = ak[1:] - ak[:-1]  # (137,)
    dbk = bk[1:] - bk[:-1]  # (137,)

    # Broadcast over spatial dims of surface_pressure
    dp = dak[:, None, None] + dbk[:, None, None] * surface_pressure.values[None, :, :]
    dp = xr.DataArray(
        dp,
        dims=["hybrid", "latitude", "longitude"],
        coords={
            "latitude": surface_pressure.latitude,
            "longitude": surface_pressure.longitude,
        },
    )
    return dp


def _vertical_coarsen(
    var: xr.DataArray,
    dp: xr.DataArray,
    output_layer_indices: Sequence[int],
) -> dict:
    """Pressure-weighted vertical coarsening from 137 levels to coarse layers."""
    n_output_layers = len(output_layer_indices) - 1
    results = {}
    for i in range(n_output_layers):
        fine_slice = slice(output_layer_indices[i], output_layer_indices[i + 1])
        var_fine = var.isel(hybrid=fine_slice)
        dp_fine = dp.isel(hybrid=fine_slice)
        weighted = (var_fine * dp_fine).sum("hybrid")
        total_dp = dp_fine.sum("hybrid")
        results[i] = (weighted / total_dp).astype(np.float32)
    return results


def _get_vertical_coordinate(
    ak: np.ndarray, bk: np.ndarray, output_layer_indices: Sequence[int]
) -> xr.Dataset:
    """Get ak/bk on coarse layer interfaces."""
    ak_coarse = [float(ak[i]) for i in output_layer_indices]
    bk_coarse = [float(bk[i]) for i in output_layer_indices]
    ak_ds = xr.Dataset({f"ak_{i}": v for i, v in enumerate(ak_coarse)})
    bk_ds = xr.Dataset({f"bk_{i}": v for i, v in enumerate(bk_coarse)})
    for name in ak_ds:
        ak_ds[name].attrs = {"long_name": "ak", "units": "Pa"}
    for name in bk_ds:
        bk_ds[name].attrs = {"long_name": "bk", "units": ""}
    return xr.merge([ak_ds, bk_ds])


def _process_model_level_data(
    ds_model: xr.Dataset,
    ds_surface: xr.Dataset,
    ak: np.ndarray,
    bk: np.ndarray,
    output_grid: str,
    output_layer_indices: Sequence[int],
) -> xr.Dataset:
    """Process model-level data: vertical coarsen at 0.25°, then regrid."""
    n_output_layers = len(output_layer_indices) - 1
    xr.set_options(keep_attrs=True)

    # Must squeeze singleton time for processing
    ds_model = ds_model.squeeze("time")
    ds_surface = ds_surface.squeeze("time")

    logging.info("Computing total specific water")
    total_specific_water = (
        ds_model["specific_humidity"]
        + ds_model["specific_cloud_liquid_water_content"]
        + ds_model["specific_cloud_ice_water_content"]
        + ds_model["specific_rain_water_content"]
        + ds_model["specific_snow_water_content"]
    )

    logging.info("Computing layer thicknesses")
    surface_pressure = ds_surface["surface_pressure"]
    dp = _compute_layer_thicknesses(ak, bk, surface_pressure)

    logging.info("Vertical coarsening at 0.25 degrees")
    output_2d = xr.Dataset()

    # Vertical coarsen 3D fields
    for short_name, full_name in [
        ("t", "temperature"),
        ("q", None),
        ("u", "u_component_of_wind"),
        ("v", "v_component_of_wind"),
    ]:
        if short_name == "q":
            data = total_specific_water
        else:
            data = ds_model[full_name]
        coarsened = _vertical_coarsen(data, dp, output_layer_indices)
        for i in range(n_output_layers):
            out_name = f"{short_name}_{i}"
            output_2d[out_name] = coarsened[i]
            logging.info(f"Vertical coarsened {short_name} layer {i}")

    # Add surface fields (already 2D)
    output_2d["PRESsfc"] = surface_pressure
    output_2d["PRMSL"] = ds_surface["mean_sea_level_pressure"]
    output_2d["skt"] = ds_surface["skin_temperature"]
    output_2d["t2m"] = ds_surface["2m_temperature"]
    output_2d["d2m"] = ds_surface["2m_dewpoint_temperature"]
    output_2d["u10"] = ds_surface["10m_u_component_of_wind"]
    output_2d["v10"] = ds_surface["10m_v_component_of_wind"]

    logging.info(f"Regridding {len(output_2d.data_vars)} 2D fields to output grid")
    regridded = _regrid(output_2d, output_grid)

    # Compute Q2m from regridded fields
    regridded["Q2m"] = _specific_humidity_from_dewpoint(
        regridded["d2m"], regridded["PRESsfc"]
    )

    # Rename to output names
    rename_dict = {
        **{f"q_{i}": f"specific_total_water_{i}" for i in range(n_output_layers)},
        **{f"t_{i}": f"air_temperature_{i}" for i in range(n_output_layers)},
        **{f"u_{i}": f"eastward_wind_{i}" for i in range(n_output_layers)},
        **{f"v_{i}": f"northward_wind_{i}" for i in range(n_output_layers)},
        "skt": "surface_temperature",
        "t2m": "TMP2m",
        "u10": "UGRD10m",
        "v10": "VGRD10m",
        "d2m": "DPT2m",
    }
    regridded = regridded.rename(rename_dict)

    for name, attrs in DESIRED_ATTRS.items():
        if name in regridded:
            regridded[name] = regridded[name].assign_attrs(**attrs)

    regridded = regridded.drop_vars(["latitude", "longitude"])
    return regridded


def process_model_level_data(
    key,
    ds,
    ds_surface=None,
    ak=None,
    bk=None,
    output_grid=DEFAULT_OUTPUT_GRID,
    output_layer_indices=DEFAULT_OUTPUT_LAYER_INDICES,
    check_data_validity: bool = False,
):
    if check_data_validity:
        _check_data_validity(ds)
        _check_data_validity(ds_surface.sel(time=ds.time))
    output = _process_model_level_data(
        ds, ds_surface.sel(time=ds.time), ak, bk, output_grid, output_layer_indices
    )
    new_key = key.replace(
        offsets={"time": key.offsets["time"], "latitude": 0, "longitude": 0},
        vars=frozenset(output.keys()),
    )
    return new_key, output


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def _make_template(
    ds_model_level,
    ds_model_level_surface,
    ds_flux,
    ds_surface_analysis,
    ds_pressure_level,
    ds_co2,
    ds_akbk,
    ak,
    bk,
    output_chunks,
    output_grid,
    output_layer_indices,
    output_time,
    check_data_validity: bool = False,
):
    """Eagerly process one timestep of each stream to build the output template."""
    logging.info("Building template from first timestep of each stream")

    # Process one timestep from each stream
    flux_one = _average_hourly_to_6hourly(ds_flux.isel(time=slice(0, 6)).load())
    ds_flux_regridded = _process_mean_flux(flux_one, output_grid)

    # Use a time from the output range for invariant data (values are constant
    # in time, but early times in the store may contain NaN fill values)
    ds_invariant = (
        open_full_37(FULL_37_INVARIANT_VARS, slice(output_time[0], output_time[0]))
        .isel(time=0)
        .load()
    )
    ds_inv_regridded = _process_invariant(
        ds_invariant, output_grid, check_data_validity
    )

    ds_sfc_an_regridded = _process_surface_analysis(
        ds_surface_analysis.isel(time=0).load(), ds_inv_regridded, output_grid
    )

    ds_pl_regridded = _process_pressure_level_data(
        ds_pressure_level.isel(time=0).load(), output_grid
    )

    ds_ml_one = ds_model_level.isel(time=slice(0, 1)).load()
    ds_ml_sfc_one = ds_model_level_surface.isel(time=slice(0, 1)).load()
    ds_ml_regridded = _process_model_level_data(
        ds_ml_one, ds_ml_sfc_one, ak, bk, output_grid, output_layer_indices
    )

    ds_regridded = xr.merge(
        [
            ds_flux_regridded,
            ds_sfc_an_regridded,
            ds_ml_regridded,
            ds_inv_regridded,
            ds_pl_regridded,
            ds_akbk,
        ],
        compat="override",
        join="outer",
    ).squeeze()

    # drop encoding (otherwise hit https://github.com/pydata/xarray/issues/10032)
    ds_regridded = ds_regridded.drop_encoding()

    # Build template with full time coordinate
    template = xbeam.make_template(ds_regridded.drop_vars("time", errors="ignore"))
    template = template.expand_dims(dim={"time": output_time}, axis=0)
    template = template.chunk(output_chunks)

    # Invariant + ak/bk written eagerly (not chunked)
    inv_fields = xr.merge([ds_inv_regridded, ds_akbk])
    inv_fields = inv_fields.drop_vars("time", errors="ignore")
    inv_fields = inv_fields.drop_encoding()
    ds_co2 = ds_co2.drop_encoding()
    template.update(inv_fields)
    template.update(ds_co2)

    return template, inv_fields


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_path", type=str, help="Output path for the processed zarr dataset"
    )
    parser.add_argument("start_time", type=str, help="Desired start of output dataset")
    parser.add_argument("end_time", type=str, help="Desired end of output dataset")
    parser.add_argument(
        "--output_grid",
        type=str,
        default="F90",
        help="Output grid specification: 'F90' for 1 degree, 'F360' for 0.25 degree.",
    )
    parser.add_argument(
        "--output_time_chunksize",
        type=int,
        default=1,
        help="Number of times per output chunk (zarr inner chunk size).",
    )
    parser.add_argument(
        "--output_time_shardsize",
        type=int,
        default=120,
        help="Number of times per output shard (zarr shard size).",
    )
    parser.add_argument(
        "--process_time_chunksize",
        type=int,
        default=6,
        help=(
            "Time chunk size for intermediate processing of pressure level and "
            "surface analysis streams. Must divide evenly into output_time_shardsize."
        ),
    )
    parser.add_argument(
        "--check_data_validity",
        action="store_true",
        help="Check for unexpected NaN values before processing.",
    )
    parser.add_argument(
        "--output-layer-indices",
        type=int,
        nargs="+",
        default=DEFAULT_OUTPUT_LAYER_INDICES,
        help=(
            "Specify the ERA5 layer indices to use when defining the vertically "
            "coarsened output levels."
        ),
    )
    return parser


def main():
    parser = _get_parser()
    args, pipeline_args = parser.parse_known_args()
    print(pipeline_args)

    start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.datetime.strptime(args.end_time, "%Y-%m-%dT%H:%M:%S")

    # Validate time alignment
    assert start_time.hour % 6 == 0, "start_time hour must be a multiple of 6"
    assert end_time.hour % 6 == 0, "end_time hour must be a multiple of 6"

    # Mean flux: need hourly data starting 5 hours before first output time
    flux_start = start_time - datetime.timedelta(hours=5)
    flux_time_slice = slice(flux_start, end_time)

    # 6-hourly output time slice
    output_time_slice = slice(start_time, end_time, TIME_STEP)

    # Validation
    msg = (
        "output_time_shardsize must be a multiple of process_time_chunksize, "
        f"got {args.output_time_shardsize} and {args.process_time_chunksize}"
    )
    assert args.output_time_shardsize % args.process_time_chunksize == 0, msg
    msg = (
        "output_time_shardsize must be a multiple of output_time_chunksize, "
        f"got {args.output_time_shardsize} and {args.output_time_chunksize}"
    )
    assert args.output_time_shardsize % args.output_time_chunksize == 0, msg

    output_chunks = {"time": args.output_time_chunksize}
    output_shards = {"time": args.output_time_shardsize}
    process_chunks = {"time": args.process_time_chunksize}

    logging.info("Opening datasets")

    # Stream 1: Mean flux (hourly data)
    ds_flux = open_full_37(FULL_37_MEAN_FLUX_VARS, flux_time_slice)

    # Stream 2: Surface analysis (6-hourly)
    ds_surface_analysis = open_full_37(FULL_37_SURFACE_ANALYSIS_VARS, output_time_slice)

    # Stream 3: Pressure levels (6-hourly)
    ds_pressure_level = open_full_37(FULL_37_PRESSURE_LEVEL_VARS, output_time_slice)

    # Stream 4: Model level (6-hourly)
    ds_model_level = open_model_level(MODEL_LEVEL_3D_VARS, output_time_slice)
    ds_model_level_surface = open_full_37(
        FULL_37_MODEL_LEVEL_SURFACE_VARS, output_time_slice
    )

    # CO2
    ds_co2 = open_co2_dataset(start_time, end_time)

    # ak/bk from model-level data
    logging.info("Getting vertical coordinate")
    ak, bk = _get_ak_bk(ds_model_level)
    ds_akbk = _get_vertical_coordinate(ak, bk, args.output_layer_indices)

    # Output time coordinate
    output_time = pd.date_range(start_time, end_time, freq=f"{TIME_STEP}h")

    logging.info("Generating template")
    template, ds_inv_regridded = _make_template(
        ds_model_level,
        ds_model_level_surface,
        ds_flux,
        ds_surface_analysis,
        ds_pressure_level,
        ds_co2,
        ds_akbk,
        ak,
        bk,
        output_chunks,
        args.output_grid,
        args.output_layer_indices,
        output_time,
        args.check_data_validity,
    )

    logging.info("Template finished generating. Starting pipeline.")
    output_store = _make_zarr_store(args.output_path, read_only=False)
    print(PipelineOptions(pipeline_args).get_all_options())
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        # Stream 1: Mean flux
        # Chunk by 6 hourly timesteps; each chunk of 6 hours -> 1 output timestep
        (
            p
            | xbeam.DatasetToChunks(ds_flux, chunks={"time": 6})
            | beam.MapTuple(
                process_mean_flux,
                output_grid=args.output_grid,
                check_data_validity=args.check_data_validity,
            )
            | "mean_flux_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "mean_flux_to_zarr"
            >> xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )

        # Stream 2: Surface analysis / invariant
        (
            p
            | "sfc_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_surface_analysis, chunks=process_chunks)
            | beam.MapTuple(
                process_surface_analysis,
                invariant_ds=ds_inv_regridded,
                output_grid=args.output_grid,
                check_data_validity=args.check_data_validity,
            )
            | "sfc_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "sfc_to_zarr"
            >> xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )

        # Stream 3: Pressure levels
        (
            p
            | "pl_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_pressure_level, chunks=process_chunks)
            | beam.MapTuple(
                process_pressure_level_data,
                output_grid=args.output_grid,
                check_data_validity=args.check_data_validity,
            )
            | "pl_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "pl_to_zarr"
            >> xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )

        # Stream 4: Model level
        (
            p
            | "ml_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_model_level, chunks={"time": 1})
            | beam.MapTuple(
                process_model_level_data,
                ds_surface=ds_model_level_surface,
                ak=ak,
                bk=bk,
                output_grid=args.output_grid,
                output_layer_indices=args.output_layer_indices,
                check_data_validity=args.check_data_validity,
            )
            | "ml_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "ml_to_zarr"
            >> xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    main()
