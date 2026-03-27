"""Post-process AIMIP inference results into CMIP6-compliant netCDF files.

Reads raw ACE output netCDFs from GCS, applies CF-convention transformations,
and uploads the results to GCS and/or DKRZ object storage.
"""

import dataclasses
import datetime
import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Callable, Literal

import cftime
import click
import fsspec
import numpy as np
import s3fs
import xarray as xr
import yaml
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Path / encoding constants ---

AIMIP_RAW_RESULTS_DIR = (
    "gs://vcm-ml-intermediate/2025-11-25-ace-aimip-inference-results"
)
AIMIP_PROCESSED_RESULTS_DIR = (
    "gs://vcm-ml-intermediate/2025-11-30-ace-aimip-processed-uploaded-data"
)
FILENAME_TEMPLATE = "{varname}_{table_id}_{source}_{experiment_id}_{variant_label}_{grid_label}_{time_range}.nc"  # noqa: E501
END_DATE = (2025, 1, 1)
TIME_ENCODING = "seconds since 1970-01-01"
CALENDAR = "proleptic_gregorian"
OUTPUT_VERSION = "v20251130"
LOCAL_DIR = "/tmp/aimip-ace/"
MODEL_SOURCE_NAME = "ACE2-ERA5"
DKRZ_REMOTE_PREFIX = f"ai-mip/Ai2/{MODEL_SOURCE_NAME}"

DEFAULT_SIMULATIONS_FILE = Path(__file__).parent / "simulations.yaml"
DEFAULT_FILES_FILE = Path(__file__).parent / "files.yaml"


# --- Configuration dataclasses ---


@dataclasses.dataclass
class SimulationConfig:
    name: str
    experiment_id: str
    realization_index: int


@dataclasses.dataclass
class FileConfig:
    varname: str
    table_id: str
    time_range: str
    grid_label: str
    standard_name: str
    long_name: str
    units: str


def load_simulations(path: Path) -> list[SimulationConfig]:
    with open(path) as f:
        items = yaml.safe_load(f)
    return [SimulationConfig(**item) for item in items]


def load_files(path: Path) -> list[FileConfig]:
    with open(path) as f:
        items = yaml.safe_load(f)
    return [FileConfig(**item) for item in items]


# --- Helper functions ---


def load_file(filepath: str, **xr_open_kwargs) -> xr.Dataset:
    with fsspec.open(filepath, "rb") as f:
        dataset = xr.open_dataset(f, **xr_open_kwargs).load()
    return dataset


def assign_global_attrs(
    ds: xr.Dataset,
    now: str,
    experiment_id: Literal["aimip", "aimip-p2k", "aimip-p4k"],
    frequency: str,
    grid: str,
    grid_label: str,
    realization_index: int,
    source_id: str,
    table_id: str,
    title: str,
    tracking_id: str,
    variable_id: str,
    variant_label: str,
) -> xr.Dataset:
    global_attributes = {
        "activity_id": "AIMIP",
        "Conventions": "CF-1.7 CMIP-6.2",
        "creation_date": now,
        "data_specs_version": "",
        "experiment": f"{experiment_id.upper()}: AI AGCM intercomparison patterned after AMIP",  # noqa: E501
        "experiment_id": experiment_id,
        "external_variables": "areacella",
        "forcing_index": "1",
        "frequency": frequency,
        "further_info_url": "https://github.com/ai2cm/ace",
        "grid": grid,
        "grid_label": grid_label,
        "initialization_index": "1",
        "institution": "Allen Institute for Artificial Intelligence",
        "institution_id": "Ai2",
        "license": (
            "CMIP6 model data produced by Allen Institute for Artificial Intelligence is licensed "  # noqa: E501
            "under a Creative Commons CC BY 4.0 license (https://creativecommons.org/licenses/). "  # noqa: E501
            "Further information about this data, including some limitations, can be found via the "  # noqa: E501
            "further_info_url (recorded as a global attribute in this file). "
        ),
        "mip_era": "CMIP6/CMIP7",
        "nominal_resolution": "1x1 degree",
        "physics_index": "1",
        "product": "model_output",
        "realization_index": f"{realization_index:d}",
        "realm": "atmos",
        "references": (
            "Watt-Meyer, et al., 2025: ACE2: accurately learning subseasonal to decadal atmospheric "  # noqa: E501
            "variability and forced responses. npj Clim Atmos Sci 8, 205 (2025). "
            "https://doi.org/10.1038/s41612-025-01090-0"
        ),
        "source": "ACE2-ERA5: ACE (Ai2 climate emulator) version 2 trained on ERA5",
        "source_id": source_id,
        "source_type": "AGCM",
        "sub_experiment": "none",
        "sub_experiment_id": "none",
        "table_id": table_id,
        "title": title,
        "tracking_id": tracking_id,
        "variable_id": variable_id,
        "variant_label": variant_label,
    }
    return ds.assign_attrs(global_attributes)


def assign_variable_attrs(
    ds: xr.Dataset,
    variable_name: str,
    standard_name: str,
    long_name: str,
    units: str,
) -> xr.Dataset:
    variable_attrs = {
        "standard_name": standard_name,
        "long_name": long_name,
        "units": units,
        "cell_methods": "area: time: mean",
        "cell_measures": "area: areacella",
    }
    assert variable_name in ds.data_vars, "variable in dataset"
    ds[variable_name] = ds[variable_name].assign_attrs(variable_attrs)
    return ds


def assign_coord_attrs(
    ds: xr.Dataset,
    lat_name: str = "lat",
    lon_name: str = "lon",
    time_name: str = "time",
) -> xr.Dataset:
    assert lat_name in ds.coords, "latitude in dataset coords"
    latitude_attrs = {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "bounds": "lat_bnds",
        "axis": "Y",
    }
    ds[lat_name] = ds[lat_name].assign_attrs(latitude_attrs)
    assert lon_name in ds.coords, "longitude in dataset coords"
    longitude_attrs = {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "bounds": "lon_bnds",
        "axis": "X",
    }
    ds[lon_name] = ds[lon_name].assign_attrs(longitude_attrs)
    assert time_name in ds.coords, "time in dataset coords"
    time_attrs = {
        "standard_name": "time",
        "long_name": "time",
        "bounds": "time_bnds",
        "axis": "T",
    }
    ds[time_name] = ds[time_name].assign_attrs(time_attrs)
    return ds


def stack_vertical_dimension(
    ds: xr.Dataset,
    vertical_dim_name: str,
    output_varname: str,
    level_pattern: str,
    standard_name: str,
    long_name: str,
    units: str,
) -> xr.Dataset:
    """Take a dataset of 2D variables with vertical coordinate in their name,
    and stack them as a dataarray along a vertical dimension."""
    output_ds = xr.Dataset().assign_attrs(ds.attrs)
    vertical_arrays = []
    for varname in sorted(list(ds.data_vars)):
        level_match = re.search(level_pattern, varname)
        if level_match is not None:
            vertical_arrays.append((float(level_match[0]), ds[varname]))
    if len(vertical_arrays) > 0:
        sorted_vertical_arrays = sorted(vertical_arrays, key=lambda x: x[0])
        output_da = xr.concat(
            [
                array.expand_dims({vertical_dim_name: [level]})
                for level, array in sorted_vertical_arrays
            ],
            dim=vertical_dim_name,
        )
    elif len(list(ds.data_vars)) == 1:
        output_da = ds[list(ds.data_vars)[0]]
    else:
        raise ValueError("Improper dataset")
    output_ds[output_varname] = output_da
    return output_ds


def assign_pressure_level_attrs(
    ds: xr.Dataset, pressure_level_name: str = "plev"
) -> xr.Dataset:
    if pressure_level_name in ds.coords:
        pressure_level_attrs = {
            "standard_name": "air_pressure",
            "long_name": "pressure",
            "units": "Pa",
            "axis": "Z",
            "positive": "down",
        }
        updated_pressure_level_coord = ds.coords[pressure_level_name] * 100  # hPa to Pa
        updated_pressure_level_coord = updated_pressure_level_coord.assign_attrs(
            pressure_level_attrs
        )
        return ds.assign_coords({pressure_level_name: updated_pressure_level_coord})
    else:
        return ds


def assign_model_layer_attrs(
    ds: xr.Dataset, model_layer_name: str = "model_layer"
) -> xr.Dataset:
    if model_layer_name in ds.coords:
        model_layer_attrs = {
            "axis": "Z",
            "positive": "down",
            "bounds": f"{model_layer_name}_bnds",
        }
        updated_model_layer_coord = ds.coords[model_layer_name].assign_attrs(
            model_layer_attrs
        )
        return ds.assign_coords({model_layer_name: updated_model_layer_coord})
    else:
        return ds


def simplify_time_sample_dims(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.squeeze()
    ds = ds.assign_coords({"time": ds["valid_time"]}).drop_vars(
        ["valid_time", "init_time"]
    )
    return ds


def monthly_data_time_coord(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.where(ds.counts > 0).dropna(dim="time")
    ds = ds.drop_vars("counts")
    month_starts = ds["time"].values - datetime.timedelta(days=14)
    return ds.assign_coords({"time": month_starts})


def daily_data_time_coord(ds: xr.Dataset) -> xr.Dataset:
    day_starts = ds["time"].values - datetime.timedelta(hours=9)
    return ds.assign_coords({"time": day_starts})


def nn_bounds(
    cell_centers_array: np.ndarray,
    start_value: float,
    end_value: float,
) -> np.ndarray:
    """Compute 1D cell bounds from centers based on nearest neighbor."""
    bounds = (cell_centers_array[:-1] + cell_centers_array[1:]) / 2
    bounds = np.concat(
        [
            np.concat([np.array([start_value]), bounds])[..., None],
            np.concat([bounds, np.array([end_value])])[..., None],
        ],
        axis=1,
    )
    return bounds


def time_bounds(
    start_array: np.ndarray,
    end_time: tuple[int, ...] = END_DATE,
) -> np.ndarray:
    """Compute 1D time bounds from period starts."""
    datetime_type = type(start_array[0])
    end_datetime = datetime_type(*end_time)
    end_array = np.concat([start_array[1:], np.array([end_datetime])])
    return np.concat([start_array[..., None], end_array[..., None]], axis=1)


def ace_layer_bounds(model_layer_name: str, bounds_name: str) -> xr.DataArray:
    """Compute pressure bounds of ACE model layers in reference atmosphere,
    i.e., surface pressure of 1000 hPa. Based on Table 2 of Watt-Meyer et al., 2023,
    https://arxiv.org/pdf/2310.02074, and
    https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions.
    """
    interface_pressures_pa = {
        0: 0.0,
        1: 5119.90,
        2: 14419.11,
        3: 25316.35,
        4: 40436.21,
        5: 59435.82,
        6: 76944.76,
        7: 90450.19,
        8: 100000.0,
    }
    layer_bounds = [
        [interface_pressures_pa[i_layer], interface_pressures_pa[i_layer + 1]]
        for i_layer in range(8)
    ]
    return xr.DataArray(
        data=layer_bounds,
        dims=[model_layer_name, bounds_name],
        coords={model_layer_name: list(range(8))},
        attrs={"units": "Pa"},
    )


def add_coord_bounds(
    ds: xr.Dataset,
    lat_name: str = "lat",
    lon_name: str = "lon",
    time_name: str = "time",
    bounds_name: str = "bnds",
    end_date: tuple[int, ...] = END_DATE,
    model_layer_name: str = "model_layer",
) -> xr.Dataset:
    ds[f"{lat_name}_{bounds_name}"] = xr.DataArray(
        nn_bounds(ds[lat_name].values, -90.0, 90.0),
        dims=[lat_name, bounds_name],
    )
    ds[f"{lon_name}_{bounds_name}"] = xr.DataArray(
        nn_bounds(ds[lon_name].values, 0.0, 360.0),
        dims=[lon_name, bounds_name],
    )
    ds[f"{time_name}_{bounds_name}"] = xr.DataArray(
        time_bounds(ds[time_name].values, end_date),
        dims=[time_name, bounds_name],
    )
    if model_layer_name in ds.coords:
        ds[f"{model_layer_name}_{bounds_name}"] = ace_layer_bounds(
            model_layer_name=model_layer_name,
            bounds_name=bounds_name,
        )
    return ds


def cftime_to_datenum(
    cftime_array: np.ndarray,
    time_encoding: str,
    calendar: str,
) -> np.ndarray:
    return cftime.date2num(cftime_array, units=time_encoding, calendar=calendar)


def encode_time_coords(
    ds: xr.Dataset,
    time_name: str = "time",
    time_bounds_name: str = "time_bnds",
    time_encoding: str = TIME_ENCODING,
    calendar: str = CALENDAR,
) -> xr.Dataset:
    assert time_name in ds.coords, "time name in coords"
    ds = ds.assign_coords(
        {time_name: cftime_to_datenum(ds[time_name], time_encoding, calendar)}
    )
    ds[time_name].attrs.update({"units": time_encoding, "calendar": calendar})
    assert time_bounds_name in ds.data_vars, "time bounds in dataset"
    ds[time_bounds_name].values[:] = cftime_to_datenum(
        ds[time_bounds_name], time_encoding, calendar
    )
    ds[time_bounds_name].attrs.update({"units": time_encoding, "calendar": calendar})
    return ds


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    cmd = ["gsutil", "-m", "cp", "-r", local_path, gcs_path]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def upload_to_dkrz(local_path: str, dkrz_path: str, fs_ace: s3fs.S3FileSystem) -> None:
    fs_ace.put(local_path, dkrz_path, recursive=True)


# --- CLI ---


@click.command()
@click.option(
    "--raw-results-dir",
    default=AIMIP_RAW_RESULTS_DIR,
    show_default=True,
    help="GCS directory containing raw inference results.",
)
@click.option(
    "--processed-results-dir",
    default=AIMIP_PROCESSED_RESULTS_DIR,
    show_default=True,
    help="GCS destination for processed results.",
)
@click.option(
    "--local-dir",
    default=LOCAL_DIR,
    show_default=True,
    help="Local scratch directory for intermediate files.",
)
@click.option(
    "--output-version",
    default=OUTPUT_VERSION,
    show_default=True,
    help="Version string included in output directory paths.",
)
@click.option(
    "--simulation",
    default=None,
    help="Process only this simulation name. Default processes all simulations.",
)
@click.option(
    "--simulations-file",
    default=str(DEFAULT_SIMULATIONS_FILE),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML file listing SimulationConfig entries.",
)
@click.option(
    "--files-file",
    default=str(DEFAULT_FILES_FILE),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML file listing FileConfig entries.",
)
@click.option("--skip-gcs-upload", is_flag=True, help="Skip uploading results to GCS.")
@click.option(
    "--skip-dkrz-upload", is_flag=True, help="Skip uploading results to DKRZ."
)
def postprocess(
    raw_results_dir: str,
    processed_results_dir: str,
    local_dir: str,
    output_version: str,
    simulation: str | None,
    simulations_file: Path,
    files_file: Path,
    skip_gcs_upload: bool,
    skip_dkrz_upload: bool,
) -> None:
    """Post-process AIMIP inference results into CMIP6-compliant netCDF files."""
    load_dotenv()

    simulations = load_simulations(simulations_file)
    files = load_files(files_file)

    if simulation is not None:
        simulations = [s for s in simulations if s.name == simulation]
        if not simulations:
            raise click.BadParameter(
                f"Simulation '{simulation}' not found in {simulations_file}."
            )

    if not skip_dkrz_upload:
        fs_ace = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": "https://s3.eu-dkrz-1.dkrz.cloud"},
            key=os.environ["ACE_KEY"],
            secret=os.environ["ACE_SECRET"],
        )
    else:
        fs_ace = None

    for sim in simulations:
        variant_label = f"r{sim.realization_index}i1p1f1"
        raw_simulation_directory = os.path.join(raw_results_dir, sim.name)
        local_processed_simulation_dir = os.path.join(
            local_dir, sim.experiment_id, variant_label
        )
        logger.info("Processing data for simulation %s...", sim.name)

        for fc in files:
            assign_vertical_coord_attrs: Callable[..., xr.Dataset]
            if fc.grid_label == "gr":
                vertical_coordinate = "plev"
                grid_description = (
                    "horizontal: native atmosphere N360 regular Gaussian grid (360x180 lonxlat); "  # noqa: E501
                    "vertical: air pressure, regridded from native via ML"
                )
                assign_vertical_coord_attrs = assign_pressure_level_attrs
            elif fc.grid_label == "gn":
                vertical_coordinate = "model_layer"
                grid_description = (
                    "horizontal: native atmosphere N360 regular Gaussian grid (360x180 lonxlat); "  # noqa: E501
                    "vertical: native atmosphere coarsened layer-means on hybrid sigma-pressure coordinate"  # noqa: E501
                )
                assign_vertical_coord_attrs = assign_model_layer_attrs
            else:
                raise ValueError(f"Invalid grid_label: {fc.grid_label}")

            if fc.table_id == "Amon":
                frequency = "mon"
                standardize_time_coord = monthly_data_time_coord
            elif fc.table_id == "day":
                frequency = "day"
                standardize_time_coord = daily_data_time_coord
            else:
                raise ValueError(f"Invalid table_id: {fc.table_id}")

            now = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            tracking_id = f"hdl:21.14100/{str(uuid.uuid3(uuid.NAMESPACE_DNS, now))}"
            filename = FILENAME_TEMPLATE.format(
                varname=fc.varname,
                table_id=fc.table_id,
                source=MODEL_SOURCE_NAME,
                experiment_id=sim.experiment_id,
                variant_label=variant_label,
                grid_label=fc.grid_label,
                time_range=fc.time_range,
            )
            filepath = os.path.join(raw_simulation_directory, filename)
            logger.info("...loading %s.", filepath)
            result_ds = load_file(
                filepath,
                decode_timedelta=False,
                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            )
            processed_file = (
                result_ds.pipe(simplify_time_sample_dims)
                .pipe(standardize_time_coord)
                .pipe(
                    stack_vertical_dimension,
                    vertical_coordinate,
                    fc.varname,
                    r"[0-9]+$",
                    fc.standard_name,
                    fc.long_name,
                    fc.units,
                )
                .pipe(add_coord_bounds)
                .pipe(encode_time_coords)
                .pipe(assign_coord_attrs)
                .pipe(assign_vertical_coord_attrs)
                .pipe(
                    assign_variable_attrs,
                    fc.varname,
                    fc.standard_name,
                    fc.long_name,
                    fc.units,
                )
                .pipe(
                    assign_global_attrs,
                    now=now,
                    experiment_id=sim.experiment_id,
                    frequency=frequency,
                    grid=grid_description,
                    grid_label=fc.grid_label,
                    realization_index=sim.realization_index,
                    source_id=MODEL_SOURCE_NAME,
                    table_id=fc.table_id,
                    title=filename,
                    tracking_id=tracking_id,
                    variable_id=fc.varname,
                    variant_label=variant_label,
                )
            )
            local_processed_output_dir = os.path.join(
                local_processed_simulation_dir,
                fc.table_id,
                fc.varname,
                fc.grid_label,
                output_version,
            )
            os.makedirs(local_processed_output_dir, exist_ok=True)
            local_processed_output_filepath = os.path.join(
                local_processed_output_dir, filename
            )
            processed_file.to_netcdf(local_processed_output_filepath)

        if not skip_gcs_upload:
            logger.info("Uploading data to GCS for simulation %s.", sim.name)
            upload_to_gcs(local_dir, processed_results_dir)

        if not skip_dkrz_upload:
            logger.info("Uploading data to DKRZ for simulation %s.", sim.name)
            upload_to_dkrz(local_dir, DKRZ_REMOTE_PREFIX, fs_ace)  # type: ignore[arg-type]

        logger.info("Deleting local data for simulation %s.", sim.name)
        shutil.rmtree(local_dir)


if __name__ == "__main__":
    postprocess()
