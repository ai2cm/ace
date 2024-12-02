import argparse
import datetime
import logging
import os
from typing import Sequence

import apache_beam as beam
import metview
import numpy as np
import pandas as pd
import xarray as xr
import xarray_beam as xbeam
from apache_beam.options.pipeline_options import PipelineOptions


def attribute_fix(ds):
    """Needed to fix a low-level bug in ecCodes.

    Sometimes, shortNames get overloaded in ecCodes's table.
    To eliminate ambiguity in their string matching, we
    force ecCodes to make use of the paramId, which is a
    consistent source-of-truth.
    """
    for var in ds:
        ds[var].attrs.pop("GRIB_cfName", None)
        ds[var].attrs.pop("GRIB_cfVarName", None)
        ds[var].attrs.pop("GRIB_shortName", None)

    return ds


def grid_attribute_fix(ds, names_to_fix, reference_name):
    """Avoid an error raised by MetView when trying to regrid certain 2D variables."""

    attrs_to_replace = [
        "GRIB_NV",
        "GRIB_gridDefinitionDescription",
        "GRIB_gridType",
        "GRIB_latitudeOfFirstGridPointInDegrees",
        "GRIB_latitudeOfLastGridPointInDegrees",
        "GRIB_longitudeOfFirstGridPointInDegrees",
        "GRIB_longitudeOfLastGridPointInDegrees",
    ]
    for name in names_to_fix:
        new_attrs = {k: ds[reference_name].attrs[k] for k in attrs_to_replace}
        ds[name] = ds[name].assign_attrs(new_attrs)
    return ds


TEMPLATE_PATH = "gs://vcm-ml-scratch/oliwm/2024-04-22-era5-regrid-template.zarr"

GRID_DOCS_URL = "https://confluence.ecmwf.int/display/OIFS/4.3+OpenIFS%3A+Horizontal+Resolution+and+Configurations"  # noqa: E501
DEFAULT_OUTPUT_GRID = "F90"  # 1° regular Gaussian grid. See GRID_DOCS_URL linked above.
TIME_STEP = 6  # in same units as resolution of time coordinate of data (i.e. hours)
GRAVITY = 9.80665  # value used in metview according to https://metview.readthedocs.io/en/latest/metview/macro/functions/fieldset.html#id0 # noqa: E501

URL_GOOGLE_ARCO_ERA5 = "gs://gcp-public-data-arco-era5/co"
URL_WIND_TEMP = f"{URL_GOOGLE_ARCO_ERA5}/model-level-wind.zarr-v2"
URL_SURFACE = f"{URL_GOOGLE_ARCO_ERA5}/single-level-surface.zarr-v2"
URL_SURFACE_REANALYSIS = f"{URL_GOOGLE_ARCO_ERA5}/single-level-reanalysis.zarr-v2"
URL_MOISTURE = f"{URL_GOOGLE_ARCO_ERA5}/model-level-moisture.zarr-v2"
URL_GOOGLE_LATLON = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
# following dataset was manually generated in https://github.com/ai2cm/explore/blob/master/oliwm/2024-07-31-generate-ERA5-co2.ipynb # noqa: E501
URL_CO2 = "gs://vcm-ml-raw-flexible-retention/2024-11-11-co2-annual-mean-for-era5.zarr"

URL_NCAR_ERA5 = (
    "gs://vcm-ml-intermediate/2024-05-17-era5-025deg-2D-variables-from-NCAR-as-zarr"  # noqa: E501
)
URL_INVARIANT = f"{URL_NCAR_ERA5}/e5.oper.invariant.zarr"
URL_SURFACE_ANALYSIS_LATLON = f"{URL_NCAR_ERA5}/e5.oper.an.sfc.zarr"
URL_MEAN_FLUX = f"{URL_NCAR_ERA5}/e5.oper.fc.sfc.meanflux.zarr"

WIND_TEMP = "wind_temp"
SURFACE = "surface"
SURFACE_REANALYSIS = "surface_reanalysis"
MOISTURE = "moisture"
INVARIANT = "invariant_latlon"
SURFACE_ANALYSIS_LATLON = "surface_analysis_latlon"
MEAN_FLUX = "mean_flux_latlon"
GOOGLE_LATLON = "google_latlon"

URLS = {
    WIND_TEMP: URL_WIND_TEMP,
    SURFACE: URL_SURFACE,
    MOISTURE: URL_MOISTURE,
    SURFACE_REANALYSIS: URL_SURFACE_REANALYSIS,
    INVARIANT: URL_INVARIANT,
    SURFACE_ANALYSIS_LATLON: URL_SURFACE_ANALYSIS_LATLON,
    MEAN_FLUX: URL_MEAN_FLUX,
    GOOGLE_LATLON: URL_GOOGLE_LATLON,
}

# these versions of the zarr datasets have variables with the necessary attrs for
# Metview to work (just stripping the '-v2' from the ends of the URLs)
URLS_WITH_REQUIRED_ATTRS = {
    WIND_TEMP: URL_WIND_TEMP[:-3],
    SURFACE: URL_SURFACE[:-3],
    MOISTURE: URL_MOISTURE[:-3],
    SURFACE_REANALYSIS: URL_SURFACE_REANALYSIS[:-3],
}

VARIABLE_NAMES = {
    WIND_TEMP: ["d", "vo", "t"],
    SURFACE: ["lnsp"],
    MOISTURE: ["q", "clwc", "ciwc", "crwc", "cswc"],
    SURFACE_REANALYSIS: ["skt", "t2m", "u10", "v10", "d2m"],
    INVARIANT: ["LSM", "Z"],
    SURFACE_ANALYSIS_LATLON: ["CI", "SWVL1", "SWVL2", "SWVL3", "SWVL4"],
    MEAN_FLUX: [
        "MER",
        "MSDWLWRF",
        "MSDWSWRF",
        "MSLHF",
        "MSNLWRF",
        "MSNSWRF",
        "MSR",
        "MSSHF",
        "MTDWSWRF",
        "MTNLWRF",
        "MTNSWRF",
        "MTPR",
        "MVIMD",
    ],
    GOOGLE_LATLON: [
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
        "total_column_water_vapour",
    ],
}

VALUE_COORD_NAME = {
    WIND_TEMP: "spharm_index",
    SURFACE: "spharm_index",
    MOISTURE: "reduced_gg_index",
    SURFACE_REANALYSIS: "reduced_gg_index",
}

SCALAR_COORDS_TO_DROP = [
    "hybrid",
    "valid_time",
    "step",
    "number",
    "heightAboveGround",
    "surface",
    "entireAtmosphere",
    "depthBelowLandLayer",
    "level",
]

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

# The input data is on the L137 ECMWF grid. See
# https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions.
# The indices below are chosen for closest alignment with the ACE vertical
# grid defined in Table 2 of https://arxiv.org/pdf/2310.02074.pdf except
# that the uppermost layer uses the higher model top of ECMWF model.
N_INPUT_LAYERS = 137  # this is the number of full layers, not interfaces
DEFAULT_OUTPUT_LAYER_INDICES = [0, 48, 67, 79, 90, 100, 109, 119, 137]

OUTPUT_PRESSURE_LEVELS = [850, 500, 200]  # additionally save these pressure levels
OUTPUT_PRESSURE_LEVELS_GEOPOTENTIAL = [1000, 850, 700, 500, 300, 250, 200]

RENAME_Q_PRES = {f"specific_humidity_{p}": f"Q{p}" for p in OUTPUT_PRESSURE_LEVELS}
RENAME_T_PRES = {f"temperature_{p}": f"TMP{p}" for p in OUTPUT_PRESSURE_LEVELS}
RENAME_U_PRES = {f"u_component_of_wind_{p}": f"UGRD{p}" for p in OUTPUT_PRESSURE_LEVELS}
RENAME_V_PRES = {f"v_component_of_wind_{p}": f"VGRD{p}" for p in OUTPUT_PRESSURE_LEVELS}
RENAME_Z_PRES = {
    f"geopotential_{p}": f"h{p}" for p in OUTPUT_PRESSURE_LEVELS_GEOPOTENTIAL
}
RENAME_PRESSURE_LEVEL = {
    **RENAME_Q_PRES,
    **RENAME_T_PRES,
    **RENAME_U_PRES,
    **RENAME_V_PRES,
    **RENAME_Z_PRES,
}


def _get_native_rename_dict(n_output_layers):
    rename_q = {f"q_{i}": f"specific_total_water_{i}" for i in range(n_output_layers)}
    rename_t = {f"t_{i}": f"air_temperature_{i}" for i in range(n_output_layers)}
    rename_u = {f"u_{i}": f"eastward_wind_{i}" for i in range(n_output_layers)}
    rename_v = {f"v_{i}": f"northward_wind_{i}" for i in range(n_output_layers)}
    rename_etc = {
        "skt": "surface_temperature",
        "t2m": "TMP2m",
        "u10": "UGRD10m",
        "v10": "VGRD10m",
        "d2m": "DPT2m",
    }
    rename_native = {
        **rename_q,
        **rename_t,
        **rename_u,
        **rename_v,
        **rename_etc,
    }
    return rename_native


def _open_zarr(key, sel_indices):
    ds = xr.open_zarr(URLS[key], chunks=None)
    ds = ds[VARIABLE_NAMES[key]]
    if key in VALUE_COORD_NAME:
        ds = ds.rename({"values": VALUE_COORD_NAME[key]})
    # xarray does not raise an error if selecting beyond bounds of time coord
    # so we manually check here that all desired data is available.
    if key != INVARIANT:
        dims = list(sel_indices[key])
        for dim in dims:
            desired_start = sel_indices[key][dim].start
            desired_stop = sel_indices[key][dim].stop
            ds_start = pd.Timestamp(ds[dim].min().values.item())
            ds_stop = pd.Timestamp(ds[dim].max().values.item())
            assert desired_start >= ds_start, f"{key} dataset {dim} start out of bounds"
            assert desired_stop <= ds_stop, f"{key} dataset {dim} stop out of bounds"
    ds = ds.sel(**sel_indices[key])
    if key == INVARIANT:
        ds = ds.drop_vars("time")
    if key in URLS_WITH_REQUIRED_ATTRS:
        tmp = xr.open_zarr(URLS_WITH_REQUIRED_ATTRS[key])
        for name in ds:
            if name in tmp:
                ds[name].attrs = tmp[name].attrs
    ds = attribute_fix(ds)
    return ds


def open_native_datasets(indices) -> xr.Dataset:
    native_dataset_keys = [WIND_TEMP, SURFACE, SURFACE_REANALYSIS, MOISTURE]
    datasets = [_open_zarr(k, indices) for k in native_dataset_keys]
    merged = xr.merge(datasets, compat="override", join="override")
    merged = grid_attribute_fix(merged, VARIABLE_NAMES[SURFACE_REANALYSIS], "q")
    return merged


def open_google_latlon_dataset(indices) -> xr.Dataset:
    ds = _open_zarr(GOOGLE_LATLON, indices)
    return ds


def open_meanflux_dataset(indices) -> xr.Dataset:
    ds = _open_zarr(MEAN_FLUX, indices)
    return ds


def open_quarter_degree_datasets(indices) -> xr.Dataset:
    sfc = _open_zarr(SURFACE_ANALYSIS_LATLON, indices)
    invariant = _open_zarr(INVARIANT, indices)
    return sfc, invariant


def open_co2_dataset(start_time, end_time) -> xr.Dataset:
    co2 = xr.open_zarr(URL_CO2, chunks=None)
    ds_start = pd.Timestamp(co2.time.values[0])
    ds_stop = pd.Timestamp(co2.time.values[-1])
    assert start_time >= ds_start, f"CO2 dataset time start out of bounds"
    assert end_time <= ds_stop, f"CO2 dataset time stop out of bounds"
    co2 = co2.sel(time=slice(start_time, end_time))
    co2 = co2.load()
    return co2


def _to_dataset(fs: metview.Fieldset) -> xr.Dataset:
    return fs.to_dataset().load()


def _to_dataarray(fs: metview.Fieldset, name: str) -> xr.DataArray:
    return fs.to_dataset()[name].load()


def _delete_fs(fs: metview.Fieldset):
    # manually delete the temporary grib file that MetView creates
    path = fs.url()
    if os.path.exists(path):
        os.remove(path)
    del fs


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
    result.attrs["units"] = "kg / kg"
    result.attrs["long_name"] = "Specific humidity"
    return result


def _to_geopotential_height(geopotential: xr.DataArray) -> xr.DataArray:
    output = geopotential / GRAVITY
    output.attrs["long_name"] = "Geopotential height"
    output.attrs["units"] = "m"
    output.attrs["standard_name"] = "geopotential_height"
    return output


def _process_pressure_level_data(ds: xr.Dataset, output_grid: str) -> xr.Dataset:
    """Select pressure levels from 0.25° pressure level dataset."""
    # convert to 2D fields at desired pressure levels
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
            select_levels[name] = ds[name]

    regridded = _regrid_quarter_degree(select_levels, output_grid)
    regridded = regridded.rename(RENAME_PRESSURE_LEVEL)
    # coordinates will be written by template, so drop here to avoid possible conflicts
    regridded = regridded.drop_vars(["latitude", "longitude"])

    return regridded


def process_pressure_level_data(key, ds, output_grid=DEFAULT_OUTPUT_GRID):
    output = _process_pressure_level_data(ds, output_grid)
    new_key = key.replace(
        offsets={"time": key.offsets["time"], "latitude": 0, "longitude": 0},
        vars=frozenset(output.keys()),
    )
    return new_key, output


def _process_native_data(
    ds: xr.Dataset, output_grid: str, output_layer_indices: Sequence[int]
) -> xr.Dataset:
    n_output_layers = len(output_layer_indices) - 1
    rename_dict = _get_native_rename_dict(n_output_layers)

    xr.set_options(keep_attrs=True)
    # singleton time dimension interferes with metview
    ds = ds.squeeze()

    logging.info("Starting calculation of horizontal winds on output grid")
    vort_div_fieldset = metview.dataset_to_fieldset(ds[["d", "vo"]].load())
    winds_spharm = metview.uvwind(data=vort_div_fieldset)
    winds_gg = metview.regrid(data=winds_spharm, grid=output_grid, truncation="none")

    logging.info("Starting calculation of total specific water")
    total_specific_water = ds.q + ds.clwc + ds.ciwc + ds.crwc + ds.cswc
    total_specific_water.attrs = ds.q.attrs
    ds["total_specific_water"] = total_specific_water

    logging.info("Starting regrid of T, q and log surface pressure to output grid")
    names = ["total_specific_water", "t", "lnsp"]
    # next step converts total_specific_water to q because of attributes
    fieldset = metview.dataset_to_fieldset(ds[names].load())
    fieldset_gg = metview.regrid(data=fieldset, grid=output_grid, truncation="none")

    logging.info("Merging with wind fieldset")
    fieldset_gg = fieldset_gg.merge(winds_gg)

    logging.info("Computing vertical integrals of T, q, u and v")
    output = xr.Dataset()
    thicknesses_fs = metview.unithickness(fieldset_gg.select(shortName="lnsp"))
    thicknesses = _to_dataarray(thicknesses_fs, "pres")
    for short_name in ["q", "t", "u", "v"]:
        variable = _to_dataarray(fieldset_gg.select(shortName=short_name), short_name)
        for output_index in range(n_output_layers):
            logging.info(
                f"Computing vertical integral of {short_name} "
                f"for output layer {output_index}."
            )

            fine_levels = slice(
                output_layer_indices[output_index],
                output_layer_indices[output_index + 1],
            )
            coarse_level_thicknesses = thicknesses.isel(hybrid=fine_levels)
            total_thickness = coarse_level_thicknesses.sum("hybrid")
            coarse_level_variable = variable.isel(hybrid=fine_levels)
            pressure_weighted = coarse_level_variable * coarse_level_thicknesses
            integrated = pressure_weighted.sum("hybrid") / total_thickness

            out_name = f"{short_name}_{output_index}"
            output[out_name] = integrated
            if short_name == "q":
                output[out_name].attrs["long_name"] = "Specific total water"
                output[out_name].attrs["standard_name"] = "specific_total_water"
            output[out_name].attrs["long_name"] += f" level-{output_index}"
    del thicknesses
    del variable

    logging.info("Computing surface pressure and inserting into output dataset")
    output["PRESsfc"] = np.exp(
        _to_dataarray(fieldset_gg.select(shortName="lnsp"), "lnsp")
    )

    logging.info("Regridding additional 2D fields and inserting into output dataset")
    # keys are names in zarr dataset, values are what metview change the names to
    grib_names = {
        "skt": "skt",
        "u10": "u",
        "v10": "v",
        "t2m": "t",
        "d2m": "dpt",
    }
    fieldset_2d = metview.dataset_to_fieldset(ds[list(grib_names)].load())
    fieldset_2d_gg = metview.regrid(data=fieldset_2d, grid=output_grid)
    dataset_2d_gg = _to_dataset(fieldset_2d_gg)
    for name in grib_names:
        output[name] = dataset_2d_gg[grib_names[name]]

    # insert 2m specific humidity
    output["Q2m"] = _specific_humidity_from_dewpoint(output["d2m"], output["PRESsfc"])

    output = _adjust_latlon(output)

    output = output.rename(rename_dict)
    for name, attrs in DESIRED_ATTRS.items():
        if name in output:
            output[name] = output[name].assign_attrs(**attrs)

    output = output.drop_vars(SCALAR_COORDS_TO_DROP, errors="ignore")

    # coordinates will be written by template, so drop here to avoid possible conflicts
    output = output.drop_vars(["latitude", "longitude"])

    # MetView creates temporary grib files which need to be deleted manually
    for fs in [
        vort_div_fieldset,
        winds_spharm,
        fieldset,
        fieldset_gg,
        fieldset_2d,
        fieldset_2d_gg,
        winds_gg,
        thicknesses_fs,
    ]:
        _delete_fs(fs)

    return output


def process_native_data(
    key,
    ds,
    output_grid=DEFAULT_OUTPUT_GRID,
    output_layer_indices=DEFAULT_OUTPUT_LAYER_INDICES,
):
    output = _process_native_data(ds, output_grid, output_layer_indices)
    new_key = key.replace(
        offsets={"time": key.offsets["time"], "latitude": 0, "longitude": 0},
        vars=frozenset(output.keys()),
    )
    return new_key, output


def _split_and_average_over_forecast_hour(ds, **merge_kwargs):
    """Convert forecast_initial_time to time and average over forecast_hour."""
    xr.set_options(keep_attrs=True)
    logging.info("Splitting and averaging over forecast_hour")
    first = ds.isel(forecast_hour=slice(0, 6))
    second = ds.isel(forecast_hour=slice(6, 12))
    second["forecast_hour"] = first.forecast_hour
    # adjust times to label by end of the averaging period
    period = np.timedelta64(6, "h")
    first["forecast_initial_time"] = first.forecast_initial_time + period
    second["forecast_initial_time"] = second.forecast_initial_time + 2 * period

    first = first.rename({"forecast_initial_time": "time"})
    second = second.rename({"forecast_initial_time": "time"})
    output_ds = xr.merge([first, second], join="outer", **merge_kwargs)

    output_ds = output_ds.mean("forecast_hour")
    return output_ds


def split_and_average_over_forecast_hour(key, ds):
    output_ds = _split_and_average_over_forecast_hour(ds)
    # assuming we are splitting forecast_initial_time span by two. I.e. into a 6-hourly
    # time coordinate.
    new_key = key.with_offsets(
        time=2 * key.offsets["forecast_initial_time"],
        forecast_initial_time=None,
        forecast_hour=None,
    )
    return new_key, output_ds


def _regrid_quarter_degree(ds, output_grid):
    for name, attrs in DESIRED_ATTRS.items():
        if name in ds:
            ds[name] = ds[name].assign_attrs(**attrs)

    # metview chokes regridding length 1 time dimension data
    if ds.sizes.get("time", None) == 1:
        ds = ds.squeeze("time")
        restore_time = True
    else:
        restore_time = False

    # regrid to desired output grid
    regridded = xr.Dataset()
    for name in ds.data_vars:
        logging.info(f"Regridding {name} to output grid")
        fieldset = metview.dataset_to_fieldset(ds[[name]].load())
        fieldset_regridded = metview.regrid(data=fieldset, grid=output_grid)
        # for some reason, metview always sets the name to "t" when regridding
        # this may have something to do with the attrs of the input dataset
        regridded[name] = _to_dataarray(fieldset_regridded, "t")
        regridded[name].attrs = ds[name].attrs
        _delete_fs(fieldset)
        _delete_fs(fieldset_regridded)

    # time gets added back in for some reason
    regridded = regridded.drop_vars("time", errors="ignore")

    regridded = _adjust_latlon(regridded)

    # drop these scalar coords that get added by metview
    regridded = regridded.drop_vars(SCALAR_COORDS_TO_DROP, errors="ignore")

    if restore_time:
        regridded = regridded.expand_dims("time", axis=0)

    return regridded


def _adjust_latlon(ds):
    """Linearly interpolate to centerpoint between longitudes and flip latitude."""
    longitude_shift = 0.5 * (ds.longitude.values[1] - ds.longitude.values[0])
    # add cyclic point to avoid extrapolation
    cyclic_point = ds.isel(longitude=0)
    cyclic_point["longitude"] = 360 + cyclic_point.longitude
    ds = xr.concat([ds, cyclic_point], dim="longitude")
    output = ds.rolling(dim={"longitude": 2}).mean()
    # outputs of rolling mean are labeled by right side of window so first value is NaN
    output = output.isel(longitude=slice(1, None))
    output["longitude"] = output.longitude - longitude_shift
    output = output.reindex(latitude=output.latitude[::-1])
    return output


def _process_quarter_degree_data_mean_flux(ds, output_grid):
    logging.info("Processing 'mean-flux' quarter degree data")
    xr.set_options(keep_attrs=True)
    output = xr.Dataset()
    output["DSWRFtoa"] = ds.MTDWSWRF
    output["USWRFtoa"] = ds.MTDWSWRF - ds.MTNSWRF
    output["ULWRFtoa"] = -ds.MTNLWRF
    output["DSWRFsfc"] = ds.MSDWSWRF
    output["USWRFsfc"] = ds.MSDWSWRF - ds.MSNSWRF
    output["DLWRFsfc"] = ds.MSDWLWRF
    output["ULWRFsfc"] = ds.MSDWLWRF - ds.MSNLWRF
    output["SHTFLsfc"] = -ds.MSSHF  # opposite sign convention as FV3
    output["LHTFLsfc"] = -ds.MSLHF  # opposite sign convention as FV3
    output["PRATEsfc"] = ds.MTPR
    output["tendency_of_total_water_path_due_to_advection"] = -ds.MVIMD
    regridded = _regrid_quarter_degree(output, output_grid)

    # coordinates will be written by template, so drop here to avoid possible conflicts
    regridded = regridded.drop_vars(["latitude", "longitude"])

    return regridded


def _process_quarter_degree_data_sfc_an(ds, invariant_ds, output_grid):
    logging.info("Processing 'surface analysis' quarter degree data")
    xr.set_options(keep_attrs=True)
    output = xr.Dataset()
    output["sea_ice_fraction"] = ds.CI.fillna(0.0)
    output["soil_moisture_0"] = ds.SWVL1
    output["soil_moisture_1"] = ds.SWVL2
    output["soil_moisture_2"] = ds.SWVL3
    output["soil_moisture_3"] = ds.SWVL4
    regridded = _regrid_quarter_degree(output, output_grid)

    # coordinates will be written by template, so drop here to avoid possible conflicts
    regridded = regridded.drop_vars(["latitude", "longitude"])
    invariant_ds = invariant_ds.drop_vars(["latitude", "longitude"])

    regridded["ocean_fraction"] = (
        1 - invariant_ds.land_fraction - regridded.sea_ice_fraction
    )

    # In regridded data, sometimes land_fraction + sea_ice_fraction is greater than 1 so
    # we make a correction here to ensure there are no negative ocean_fraction values.
    # In principle, could result in negative sea_ice_fraction values, but we don't
    # find this to happen in practice.
    negative_ocean = xr.where(regridded.ocean_fraction < 0, regridded.ocean_fraction, 0)
    regridded["ocean_fraction"] -= negative_ocean
    regridded["sea_ice_fraction"] += negative_ocean

    for name in ["ocean_fraction", "sea_ice_fraction"]:
        regridded[name] = regridded[name].assign_attrs(DESIRED_ATTRS[name])

    return regridded


def _process_quarter_degree_data_invariant(ds, output_grid):
    logging.info("Renaming and fixing sign conventions for lat-lon invariant data")

    output = xr.Dataset()
    output["HGTsfc"] = ds.Z / GRAVITY
    output["land_fraction"] = ds.LSM

    regridded = _regrid_quarter_degree(output, output_grid)

    return regridded


def process_quarter_degree_data_mean_flux(key, ds, output_grid=DEFAULT_OUTPUT_GRID):
    output_ds = _process_quarter_degree_data_mean_flux(ds, output_grid)
    new_key = key.replace(vars=frozenset(output_ds.keys()))
    return new_key, output_ds


def process_quarter_degree_data_sfc_an(
    key, ds, invariant_ds=None, output_grid=DEFAULT_OUTPUT_GRID
):
    output_ds = _process_quarter_degree_data_sfc_an(ds, invariant_ds, output_grid)
    new_key = key.replace(vars=frozenset(output_ds.keys()))
    return new_key, output_ds


def _get_vertical_coordinate(
    ds: xr.Dataset, name: str, output_layer_indices: Sequence[int]
) -> xr.Dataset:
    """Get the ak/bk vertical coordinate on coarse layer interfaces.

    Assuming that ds[name] is a 3D variable which includes
    the vertical coordinate as an attribute named 'GRIB_pv'.
    """
    hybrid_sigma_values = ds[name].attrs["GRIB_pv"]
    ak = hybrid_sigma_values[: N_INPUT_LAYERS + 1]
    bk = hybrid_sigma_values[N_INPUT_LAYERS + 1 :]
    ak_coarse = [ak[i] for i in output_layer_indices]
    bk_coarse = [bk[i] for i in output_layer_indices]
    ak_coarse_ds = xr.Dataset({f"ak_{i}": value for i, value in enumerate(ak_coarse)})
    bk_coarse_ds = xr.Dataset({f"bk_{i}": value for i, value in enumerate(bk_coarse)})
    for name in ak_coarse_ds:
        ak_coarse_ds[name].attrs = {"long_name": f"ak", "units": "Pa"}
    for name in bk_coarse_ds:
        bk_coarse_ds[name].attrs = {"long_name": f"bk", "units": ""}
    return xr.merge([ak_coarse_ds, bk_coarse_ds])


def _make_template(
    ds_native,
    ds_meanflux,
    ds_quarter_degree_sfc,
    ds_quarter_degree_invariant,
    ds_google_latlon,
    ds_co2,
    ds_akbk,
    output_chunks,
    reuse_template,
    output_grid,
    output_layer_indices,
):
    """Here we (mostly) lazily process the data to make a reference zarr store
    for the output. This function mirrors what the pipeline does."""

    akbk_names = list(ds_akbk.keys())
    if reuse_template:
        logging.info(f"Using existing template at {TEMPLATE_PATH}")
        ds_regridded = xr.open_zarr(TEMPLATE_PATH).load()
        inv_fields = ds_regridded[["land_fraction", "HGTsfc"] + akbk_names]
    else:
        logging.info(f"Creating new template at {TEMPLATE_PATH}")
        ds_reshaped = _split_and_average_over_forecast_hour(
            ds_meanflux.isel(forecast_initial_time=0), compat="override"
        )
        ds_mean_flux_regridded = _process_quarter_degree_data_mean_flux(
            ds_reshaped, output_grid
        )
        ds_invariant_regridded = _process_quarter_degree_data_invariant(
            ds_quarter_degree_invariant, output_grid
        )
        ds_sfc_an_regridded = _process_quarter_degree_data_sfc_an(
            ds_quarter_degree_sfc.isel(time=0), ds_invariant_regridded, output_grid
        )
        ds_native_regridded = _process_native_data(
            ds_native.isel(time=0), output_grid, output_layer_indices
        )
        ds_google_latlon_regridded = _process_pressure_level_data(
            ds_google_latlon.isel(time=0), output_grid
        )
        ds_regridded = xr.merge(
            [
                ds_mean_flux_regridded,
                ds_sfc_an_regridded,
                ds_native_regridded,
                ds_invariant_regridded,
                ds_google_latlon_regridded,
                ds_akbk,
            ],
            compat="override",
            join="outer",
        ).squeeze()

        ds_regridded = ds_regridded.chunk({"latitude": -1, "longitude": -1})
        ds_regridded.to_zarr(TEMPLATE_PATH, mode="w")
        inv_fields = xr.merge([ds_invariant_regridded, ds_akbk])

    # manually expand time dim to include full time coordinate
    desired_time = ds_native.time.drop_vars(SCALAR_COORDS_TO_DROP, errors="ignore")
    template = xbeam.make_template(ds_regridded.drop_vars("time"))
    template = template.expand_dims(dim={"time": desired_time}, axis=0)
    template = template.chunk(output_chunks)

    inv_fields = inv_fields.drop_vars("time", errors="ignore")
    # includes regridded invariant field because ocean_fraction depends on
    # land fraction and temporally variable sea ice fraction
    # this will get written eagerly since it is not chunked
    template = template.update(inv_fields)
    template = template.update(ds_co2)

    return template, inv_fields


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
        help=(
            "Output grid specification according to ECMWF nomenclature. E.g. 'F90' for "
            f"1° Gaussian Grid. See more information at {GRID_DOCS_URL}"
        ),
    )
    parser.add_argument(
        "--output_time_chunksize",
        type=int,
        default=50,
        help="Number of times per output chunk.",
    )
    parser.add_argument(
        "--ncar_process_time_chunksize",
        type=int,
        default=10,
        help=(
            "Time chunk size for intermediate regridding step for all data from "
            "NCAR-sourced datasets (mean flux / surface analysis / invariant). Must "
            "be a multiple of 2 and divide evenly into output_time_chunksize."
        ),
    )
    parser.add_argument(
        "--reuse_template",
        action="store_true",
        help=(
            "Reuse the existing template at {TEMPLATE_PATH}. This can be helpful to "
            "speed up debugging but should not be used for production runs."
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

    # desired start/end of output dataset, inclusive
    start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.datetime.strptime(args.end_time, "%Y-%m-%dT%H:%M:%S")

    # to properly align mean flux with the output time, these requirements are necessary
    assert start_time.hour == 0 or start_time.hour == 12
    assert end_time.hour == 6 or end_time.hour == 18

    # mean flux forecast initial times are at 6Z and 18Z, and we label by the
    # end of the time-averaging periods
    start_time_mean_flux = start_time - datetime.timedelta(hours=6)
    end_time_mean_flux = end_time - datetime.timedelta(hours=12)

    regular_time_slice = {"time": slice(start_time, end_time, TIME_STEP)}
    forecast_time_slice = {
        "forecast_initial_time": slice(start_time_mean_flux, end_time_mean_flux)
    }
    sel_indices = {
        WIND_TEMP: regular_time_slice,
        SURFACE: regular_time_slice,
        SURFACE_REANALYSIS: regular_time_slice,
        MOISTURE: regular_time_slice,
        INVARIANT: dict(time="1979-01-01T00:00:00"),
        SURFACE_ANALYSIS_LATLON: regular_time_slice,
        MEAN_FLUX: forecast_time_slice,
        GOOGLE_LATLON: regular_time_slice,
    }

    msg = (
        "ncar_process_time_chunksize must be a multiple of 2, "
        f"got {args.ncar_process_time_chunksize}"
    )
    assert args.ncar_process_time_chunksize % 2 == 0, msg
    msg = (
        "output_time_chunksize must be a multiple of ncar_process_time_chunksize, "
        f"got {args.output_time_chunksize} and {args.ncar_process_time_chunksize}"
    )
    assert args.output_time_chunksize % args.ncar_process_time_chunksize == 0, msg
    output_chunks = {"time": args.output_time_chunksize}
    ncar_process_chunks = {"time": args.ncar_process_time_chunksize}

    logging.info("Opening datasets")
    ds_native = open_native_datasets(sel_indices)
    ds_meanflux = open_meanflux_dataset(sel_indices)
    ds_quarter_degree_sfc, ds_quarter_degree_inv = open_quarter_degree_datasets(
        sel_indices
    )
    ds_google_latlon = open_google_latlon_dataset(sel_indices)
    ds_co2 = open_co2_dataset(start_time, end_time)
    logging.info("Getting vertical coordinate")
    ds_akbk = _get_vertical_coordinate(ds_native, "t", args.output_layer_indices)

    logging.info("Generating template")
    template, ds_pt25deg_inv_regridded = _make_template(
        ds_native,
        ds_meanflux,
        ds_quarter_degree_sfc,
        ds_quarter_degree_inv,
        ds_google_latlon,
        ds_co2,
        ds_akbk,
        output_chunks,
        args.reuse_template,
        args.output_grid,
        args.output_layer_indices,
    )

    logging.info("Template finished generating. Starting pipeline.")
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        (
            p
            | xbeam.DatasetToChunks(ds_meanflux, chunks={"forecast_initial_time": 1})
            | beam.MapTuple(split_and_average_over_forecast_hour)
            | xbeam.ConsolidateChunks(ncar_process_chunks)
            | beam.MapTuple(
                process_quarter_degree_data_mean_flux, output_grid=args.output_grid
            )
            | "mean_flux_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_chunks)
            | "mean_flux_to_zarr"
            >> xbeam.ChunksToZarr(args.output_path, template, output_chunks)
        )

        (
            p
            | "qd_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_quarter_degree_sfc, chunks=ncar_process_chunks)
            | beam.MapTuple(
                process_quarter_degree_data_sfc_an,
                invariant_ds=ds_pt25deg_inv_regridded,
                output_grid=args.output_grid,
            )
            | "qd_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_chunks)
            | "qd_to_zarr"
            >> xbeam.ChunksToZarr(args.output_path, template, output_chunks)
        )

        (
            p
            | "pl_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_google_latlon, chunks=ncar_process_chunks)
            | beam.MapTuple(process_pressure_level_data)
            | "pl_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_chunks)
            | "pl_to_zarr"
            >> xbeam.ChunksToZarr(args.output_path, template, output_chunks)
        )

        (
            p
            | "native_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_native, chunks={"time": 1})
            | beam.MapTuple(
                process_native_data,
                output_grid=args.output_grid,
                output_layer_indices=args.output_layer_indices,
            )
            | "native_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_chunks)
            | "native_to_zarr"
            >> xbeam.ChunksToZarr(args.output_path, template, output_chunks)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    main()
