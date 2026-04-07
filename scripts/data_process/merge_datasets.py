# This script is used to append to an existing dataset
import dataclasses
import logging
import os
import sys
import time
from typing import Dict, Mapping, MutableMapping, Sequence

import click
import dacite
import xarray as xr
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compute_dataset import (
    DatasetComputationConfig,
    DatasetConfig,
    clear_compressors_encoding,
)


@dataclasses.dataclass
class DatasetAppendConfig:
    variable_sources: Mapping[str, Sequence[str]]
    renaming: Mapping[str, str] = dataclasses.field(default_factory=dict)
    start_date: str | None = None
    end_date: str | None = None

    @classmethod
    def from_file(cls, path: str) -> "DatasetAppendConfig":
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )


def get_dataset_urls(
    config: DatasetAppendConfig, run_directory: str
) -> MutableMapping[str, str]:
    return {k: os.path.join(run_directory, k) for k in config.variable_sources}


def open_datasets(
    config: DatasetAppendConfig, urls: MutableMapping[str, str]
) -> xr.Dataset:
    datasets = []
    for store, names in config.variable_sources.items():
        url = urls[store]
        ds = xr.open_zarr(url, decode_timedelta=True)[names]
        datasets.append(ds)
    return xr.merge(datasets, compat="equals")


def get_dataset_attrs(store_path: str) -> Dict:
    ds = xr.open_zarr(store_path)
    return ds.attrs


def get_variables_to_append(
    run_directory: str,
    append_store: str,
    dataset_config: DatasetComputationConfig,
    append_config: DatasetAppendConfig,
) -> xr.Dataset:
    urls = get_dataset_urls(append_config, run_directory)

    ds_temp = open_datasets(append_config, urls)

    ds = xr.Dataset()

    output_ds = xr.open_zarr(append_store)
    for variable in ds_temp.keys():
        if variable in output_ds:
            if variable in append_config.renaming:
                logging.warning(
                    f"Variable {variable} already exists in {append_store} and "
                    f"is being renamed to {append_config.renaming[variable]}."
                )
            else:
                logging.info(
                    f"Variable {variable} already exists in {append_store}, skipping."
                )
                continue
        ds[variable] = ds_temp[variable].sel(
            time=slice(append_config.start_date, append_config.end_date)
        )
    return ds


def get_merged_ds(
    run_directory: str,
    append_store: str,
    dataset_config: DatasetComputationConfig,
    append_config: DatasetAppendConfig,
) -> xr.Dataset:
    urls = get_dataset_urls(append_config, run_directory)

    ds_temp = open_datasets(append_config, urls)

    ds = xr.Dataset()

    output_ds = xr.open_zarr(append_store)
    output_ds = output_ds.assign_coords(
        latitude=ds_temp.latitude,
        longitude=ds_temp.longitude,
    )

    for variable in ds_temp.keys():
        ds[variable] = ds_temp[variable].sel(
            time=slice(append_config.start_date, append_config.end_date)
        )
    for variable in output_ds.data_vars:
        if variable == "surface_temperature":
            ds["sea_surface_temperature"] = output_ds[variable].sel(
                time=slice(append_config.start_date, append_config.end_date)
            )
        else:
            if variable not in ds:
                if output_ds[variable].dims == ("time", "latitude", "longitude"):
                    ds[variable] = output_ds[variable].sel(
                        time=slice(append_config.start_date, append_config.end_date)
                    )
                else:
                    ds[variable] = output_ds[variable]
    return ds


@click.command()
@click.option("--dataset-config", help="Path to dataset configuration YAML file.")
@click.option("--append-config", help="Path to append configuration YAML file.")
@click.option(
    "--run-directory",
    help="Path to reference run directory containing desired variables to be appended.",
)
@click.option(
    "--append-store", help="Path to an existing zarr store that will be appended."
)
@click.option("--debug", is_flag=True, help="Print metadata instead of writing output.")
@click.option(
    "--override-store-name",
    required=False,
    help="Override the location of where to save the store",
)
def main(
    dataset_config,
    append_config,
    run_directory,
    append_store,
    debug,
    override_store_name,
):
    import distributed
    import xpartition  # noqa: F401

    logging.basicConfig(level=logging.INFO)
    distributed.Client(n_workers=32)

    dataset_config = DatasetConfig.from_file(dataset_config).dataset_computation
    append_config = DatasetAppendConfig.from_file(append_config)

    logging.info(f"--run-directory is {run_directory}")
    logging.info(f"--append-store is {append_store}")

    xr.set_options(keep_attrs=True)

    standard_names = dataset_config.standard_names

    ds = get_merged_ds(
        run_directory,
        append_store,
        dataset_config,
        append_config,
    )

    if len(ds.data_vars) == 0:
        logging.info("No new variables to append.")
        return

    if dataset_config.sharding is None:
        inner_chunks = None
    else:
        inner_chunks = dataset_config.chunking.get_chunks(standard_names)

    attributes = get_dataset_attrs(append_store)
    variable_str = ", ".join(ds.keys())
    if "history" in attributes:
        attributes["history"] = (
            attributes["history"]
            + " "
            + "Dataset extended by full-model/scripts/data_process/append_dataset.py "
            f"script, adding the following variables: {variable_str}."
        )
    else:
        attributes["history"] = (
            "Dataset extended by full-model/scripts/data_process/append_dataset.py "
            f"script, adding the following variables: {variable_str}."
        )

    ds.attrs = attributes

    ds = ds.rename(append_config.renaming)

    ds = clear_compressors_encoding(ds)

    outer_chunks = dataset_config.sharding.get_chunks(standard_names)

    ds = ds.chunk(outer_chunks)

    logging.info(f"Append dataset size is {ds.nbytes / 1e9} GB")
    if debug:
        with xr.set_options(display_max_rows=500):
            logging.info(ds)
    else:
        if override_store_name is not None:
            append_store = override_store_name
            ds.partition.initialize_store(
                append_store, inner_chunks=inner_chunks, mode="w"
            )
        else:
            ds.partition.initialize_store(
                append_store, inner_chunks=inner_chunks, mode="a"
            )
        logging.info(
            f"Appending dataset to {append_store} with inner chunks {inner_chunks}"
        )
        for i in range(dataset_config.n_split):
            segment_number = f"{i + 1} / {dataset_config.n_split}"
            logging.info(f"Writing segment {segment_number}")
            segment_time = time.time()
            ds.partition.write(
                append_store,
                dataset_config.n_split,
                [dataset_config.standard_names.time_dim],
                i,
                collect_variable_writes=True,
            )
            segment_time = time.time() - segment_time
            logging.info(
                f"Appending Segment {segment_number} time: {segment_time:0.2f} seconds"
            )

        logging.info(f"Dataset appended to {append_store}")


if __name__ == "__main__":
    main()
