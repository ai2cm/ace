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

GRAVITY = 9.80665
TIME_STEP = 6  # hours between output timesteps
DEFAULT_OUTPUT_GRID = "F90"
N_INPUT_LAYERS = 137
DEFAULT_OUTPUT_LAYER_INDICES = [0, 48, 67, 79, 90, 100, 109, 119, 137]

OUTPUT_PRESSURE_LEVELS = [850, 500, 200]
OUTPUT_PRESSURE_LEVELS_GEOPOTENTIAL = [1000, 850, 700, 500, 300, 250, 200]

# Gaussian grid specs: name -> N (grid number; nlat=2N, nlon=4N)
GAUSSIAN_GRID_N = {
    "F90": 90,
    "F360": 360,
}

URL_FULL_37 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
URL_MODEL_LEVEL = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
URL_CO2 = "gs://vcm-ml-raw-flexible-retention/2024-11-11-co2-annual-mean-for-era5.zarr"

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
]

FULL_37_SURFACE_ANALYSIS_VARS = [
    "sea_ice_cover",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]

FULL_37_INVARIANT_VARS = [
    "land_sea_mask",
    "geopotential_at_surface",
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
    "skin_temperature",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

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
}


# ---------------------------------------------------------------------------
# Regridding utilities
# ---------------------------------------------------------------------------


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Compute cell boundaries from centers, clipping to [lo, hi]."""
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    bounds = np.concatenate([[lo], midpoints, [hi]])
    return bounds


def _gaussian_latitudes(n: int) -> np.ndarray:
    """Compute Gaussian grid latitudes for grid number N (2N latitudes).

    Returns latitudes in degrees, sorted south-to-north.
    These are the roots of the Legendre polynomial P_{2N}(sin(lat)).
    """
    from numpy.polynomial.legendre import leggauss

    x, _ = leggauss(2 * n)
    lat = np.degrees(np.arcsin(x))
    return np.sort(lat)


def _make_target_grid(output_grid: str) -> xr.Dataset:
    """Create Gaussian target grid dataset for xESMF regridding."""
    n = GAUSSIAN_GRID_N[output_grid]
    lat = _gaussian_latitudes(n)
    nlon = 4 * n
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
    """Get or create a cached xESMF conservative regridder."""
    if output_grid not in _REGRIDDER_CACHE:
        src = _make_source_grid()
        dst = _make_target_grid(output_grid)
        _REGRIDDER_CACHE[output_grid] = xe.Regridder(
            src, dst, "conservative", periodic=True
        )
    return _REGRIDDER_CACHE[output_grid]


def _regrid(ds: xr.Dataset, output_grid: str) -> xr.Dataset:
    """Regrid a dataset from 0.25° regular lat-lon to a Gaussian target grid."""
    regridder = _get_regridder(output_grid)
    # Rename coords to lat/lon for xESMF
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    # Ensure south-to-north
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])
    regridded = regridder(ds)
    # Rename back to latitude/longitude for output consistency
    regridded = regridded.rename({"lat": "latitude", "lon": "longitude"})
    return regridded


# ---------------------------------------------------------------------------
# Physics utilities
# ---------------------------------------------------------------------------


def _saturation_vapor_pressure(t: xr.DataArray) -> xr.DataArray:
    a1 = 611.21
    a2 = 273.16
    a3 = 17.502
    a4 = 32.19
    return a1 * np.exp(a3 * (t - a2) / (t - a4))


def _specific_humidity_from_dewpoint(
    dewpoint: xr.DataArray, pressure: xr.DataArray
) -> xr.DataArray:
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


# ---------------------------------------------------------------------------
# Data opening
# ---------------------------------------------------------------------------


def open_full_37(variables, time_slice) -> xr.Dataset:
    """Open variables from the full_37 ARCO-ERA5 store."""
    ds = xr.open_zarr(URL_FULL_37, chunks=None)
    ds = ds[variables]
    if time_slice is not None:
        _check_time_bounds(ds, time_slice)
        ds = ds.sel(time=time_slice)
    return ds


def open_model_level(variables, time_slice) -> xr.Dataset:
    """Open variables from the model-level ARCO-ERA5 store."""
    ds = xr.open_zarr(URL_MODEL_LEVEL, chunks=None)
    ds = ds[variables]
    _check_time_bounds(ds, time_slice)
    ds = ds.sel(time=time_slice)
    return ds


def open_co2_dataset(start_time, end_time) -> xr.Dataset:
    co2 = xr.open_zarr(URL_CO2, chunks=None)
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


# ---------------------------------------------------------------------------
# Stream 1: Mean Flux processing
# ---------------------------------------------------------------------------


def _process_mean_flux(ds: xr.Dataset, output_grid: str) -> xr.Dataset:
    """Compute derived fluxes from hourly mean flux data (already time-averaged)."""
    logging.info("Processing mean flux data")
    xr.set_options(keep_attrs=True)
    output = xr.Dataset()
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
    output["SHTFLsfc"] = -ds["mean_surface_sensible_heat_flux"]
    output["LHTFLsfc"] = -ds["mean_surface_latent_heat_flux"]
    output["PRATEsfc"] = ds["mean_total_precipitation_rate"]
    output["tendency_of_total_water_path_due_to_advection"] = -ds[
        "mean_vertically_integrated_moisture_divergence"
    ]

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


def process_mean_flux(key, ds, output_grid=DEFAULT_OUTPUT_GRID):
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


def _process_invariant(ds: xr.Dataset, output_grid: str) -> xr.Dataset:
    """Process invariant fields (land_sea_mask, geopotential_at_surface)."""
    logging.info("Processing invariant data")
    output = xr.Dataset()
    output["HGTsfc"] = ds["geopotential_at_surface"] / GRAVITY
    output["land_fraction"] = ds["land_sea_mask"]
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

    regridded = _regrid(output, output_grid)
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

    return regridded


def process_surface_analysis(
    key, ds, invariant_ds=None, output_grid=DEFAULT_OUTPUT_GRID
):
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
            for pressure in pressure_levels:
                logging.info(f"Selecting {name} at {pressure} hPa")
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


def process_pressure_level_data(key, ds, output_grid=DEFAULT_OUTPUT_GRID):
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
        results[i] = weighted / total_dp
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
):
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
    ds_inv_regridded = _process_invariant(ds_invariant, output_grid)

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

    # Build template with full time coordinate
    template = xbeam.make_template(ds_regridded.drop_vars("time", errors="ignore"))
    template = template.expand_dims(dim={"time": output_time}, axis=0)
    template = template.chunk(output_chunks)

    # Invariant + ak/bk written eagerly (not chunked)
    inv_fields = xr.merge([ds_inv_regridded, ds_akbk])
    inv_fields = inv_fields.drop_vars("time", errors="ignore")
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
    )

    logging.info("Template finished generating. Starting pipeline.")
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        # Stream 1: Mean flux
        # Chunk by 6 hourly timesteps; each chunk of 6 hours -> 1 output timestep
        (
            p
            | xbeam.DatasetToChunks(ds_flux, chunks={"time": 6})
            | beam.MapTuple(process_mean_flux, output_grid=args.output_grid)
            | "mean_flux_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "mean_flux_to_zarr"
            >> xbeam.ChunksToZarr(
                args.output_path,
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
            )
            | "sfc_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "sfc_to_zarr"
            >> xbeam.ChunksToZarr(
                args.output_path,
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
            | beam.MapTuple(process_pressure_level_data, output_grid=args.output_grid)
            | "pl_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "pl_to_zarr"
            >> xbeam.ChunksToZarr(
                args.output_path,
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
            )
            | "ml_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "ml_to_zarr"
            >> xbeam.ChunksToZarr(
                args.output_path,
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
