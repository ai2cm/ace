import argparse
import datetime
import logging
import os

import apache_beam as beam
import metview
import pandas as pd
import xarray as xr
import xarray_beam as xbeam
from apache_beam.options.pipeline_options import PipelineOptions

GRID_DOCS_URL = "https://confluence.ecmwf.int/display/OIFS/4.3+OpenIFS%3A+Horizontal+Resolution+and+Configurations"  # noqa: E501
DEFAULT_OUTPUT_GRID = "F90"  # 1° regular Gaussian grid. See GRID_DOCS_URL linked above.
TIME_STEP = 6  # in same units as resolution of time coordinate of data (i.e. hours)
GRAVITY = 9.80665  # value used in metview according to https://metview.readthedocs.io/en/latest/metview/macro/functions/fieldset.html#id0 # noqa: E501

URL_GOOGLE_ARCO_ERA5 = "gs://gcp-public-data-arco-era5"
URL_GOOGLE_LATLON = f"{URL_GOOGLE_ARCO_ERA5}/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
GOOGLE_LATLON = "google_latlon"

URLS = {
    GOOGLE_LATLON: URL_GOOGLE_LATLON,
}

VARIABLE_NAMES = {
    GOOGLE_LATLON: [
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
    ],
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


OUTPUT_PRESSURE_LEVELS = [
    1000,
    925,
    850,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    50,
]

RENAME_Q_PRES = {f"specific_humidity_{p}": f"Q{p}" for p in OUTPUT_PRESSURE_LEVELS}
RENAME_T_PRES = {f"temperature_{p}": f"TMP{p}" for p in OUTPUT_PRESSURE_LEVELS}
RENAME_U_PRES = {f"u_component_of_wind_{p}": f"UGRD{p}" for p in OUTPUT_PRESSURE_LEVELS}
RENAME_V_PRES = {f"v_component_of_wind_{p}": f"VGRD{p}" for p in OUTPUT_PRESSURE_LEVELS}
RENAME_Z_PRES = {f"geopotential_{p}": f"h{p}" for p in OUTPUT_PRESSURE_LEVELS}
RENAME_PRESSURE_LEVEL = {
    **RENAME_Q_PRES,
    **RENAME_T_PRES,
    **RENAME_U_PRES,
    **RENAME_V_PRES,
    **RENAME_Z_PRES,
}


def _open_zarr(key, sel_indices):
    ds = xr.open_zarr(URLS[key], chunks=None)
    ds = ds[VARIABLE_NAMES[key]]
    # xarray does not raise an error if selecting beyond bounds of time coord
    # so we manually check here that all desired data is available.
    dims = list(sel_indices[key])
    for dim in dims:
        desired_start = sel_indices[key][dim].start
        desired_stop = sel_indices[key][dim].stop
        ds_start = pd.Timestamp(ds[dim].min().values.item())
        ds_stop = pd.Timestamp(ds[dim].max().values.item())
        assert desired_start >= ds_start, f"{key} dataset {dim} start out of bounds"
        assert desired_stop <= ds_stop, f"{key} dataset {dim} stop out of bounds"
    ds = ds.sel(**sel_indices[key])
    return ds


def open_google_latlon_dataset(indices) -> xr.Dataset:
    ds = _open_zarr(GOOGLE_LATLON, indices)
    return ds


def _to_dataarray(fs: metview.Fieldset, name: str) -> xr.DataArray:
    return fs.to_dataset()[name].load()


def _delete_fs(fs: metview.Fieldset):
    # manually delete the temporary grib file that MetView creates
    path = fs.url()
    if os.path.exists(path):
        os.remove(path)
    del fs


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
            for pressure in OUTPUT_PRESSURE_LEVELS:
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
    return regridded


def process_pressure_level_data(key, ds, output_grid=DEFAULT_OUTPUT_GRID):
    output = _process_pressure_level_data(ds, output_grid)
    # coordinates will be written by template, so drop here to avoid possible conflicts
    output = output.drop_vars(["latitude", "longitude", "time"], errors="ignore")
    new_key = key.replace(
        offsets={"time": key.offsets["time"], "latitude": 0, "longitude": 0},
        vars=frozenset(output.keys()),
    )
    return new_key, output


def _regrid_quarter_degree(ds, output_grid):
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


def _make_template(
    ds_google_latlon,
    output_chunks,
    output_grid,
):
    """Here we (mostly) lazily process the data to make a reference zarr store
    for the output. This function mirrors what the pipeline does."""

    desired_time = ds_google_latlon.time.drop_vars(
        SCALAR_COORDS_TO_DROP, errors="ignore"
    )
    ds_google_latlon_regridded_time_sample = _process_pressure_level_data(
        ds_google_latlon.isel(time=0), output_grid
    ).squeeze()

    ds_google_latlon_regridded_time_sample = (
        ds_google_latlon_regridded_time_sample.chunk({"latitude": -1, "longitude": -1})
    )

    # manually expand time dim to include full time coordinate
    template = xbeam.make_template(ds_google_latlon_regridded_time_sample)
    template = template.expand_dims(dim={"time": desired_time}, axis=0)
    template = template.chunk(output_chunks)

    return template


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
        default=20,
        help="Number of times per output chunk.",
    )
    parser.add_argument(
        "--process_time_chunksize",
        type=int,
        default=10,
        help=(
            "Time chunk size for intermediate regridding step. Must divide evenly into "
            "output_time_chunksize."
        ),
    )
    return parser


def main():
    parser = _get_parser()
    args, pipeline_args = parser.parse_known_args()

    # desired start/end of output dataset, inclusive
    start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.datetime.strptime(args.end_time, "%Y-%m-%dT%H:%M:%S")

    regular_time_slice = {"time": slice(start_time, end_time, TIME_STEP)}
    sel_indices = {
        GOOGLE_LATLON: regular_time_slice,
    }

    msg = (
        "output_time_chunksize must be a multiple of process_time_chunksize, "
        f"got {args.output_time_chunksize} and {args.process_time_chunksize}"
    )
    assert args.output_time_chunksize % args.process_time_chunksize == 0, msg
    output_chunks = {"time": args.output_time_chunksize}
    process_chunks = {"time": args.process_time_chunksize}

    logging.info("Opening datasets")
    ds_google_latlon = open_google_latlon_dataset(sel_indices)

    logging.info("Generating template")
    template = _make_template(
        ds_google_latlon,
        output_chunks,
        args.output_grid,
    )

    logging.info("Template finished generating. Starting pipeline.")
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        (
            p
            | "pl_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_google_latlon, chunks=process_chunks)
            | beam.MapTuple(process_pressure_level_data)
            | "pl_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_chunks)
            | "pl_to_zarr"
            >> xbeam.ChunksToZarr(args.output_path, template, output_chunks)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    main()
