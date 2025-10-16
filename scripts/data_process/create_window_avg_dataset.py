import dataclasses
import os
import pdb
import sys
from datetime import datetime

import click
import dacite
import pandas as pd
import xarray as xr
import yaml
from time_utils import shift_timestamps_to_midpoint
from writer_utils import OutputWriterConfig


@dataclasses.dataclass
class InputDatasetConfig:
    zarr_path: str
    time_chunk_size: int
    names: list[str]
    merge_zarr: str | None = None

    def get_dataset(self) -> xr.Dataset:
        ds = xr.open_zarr(self.zarr_path, chunks={"time": self.time_chunk_size})
        if self.merge_zarr is not None:
            ds2 = xr.open_zarr(self.merge_zarr, chunks={"time": self.time_chunk_size})
            ds = xr.merge([ds, ds2])
        return ds[self.names]


@dataclasses.dataclass
class WindowAvgDatasetConfig:
    """
    Configuration for creating a window-averaged dataset.

    This class defines the parameters needed to generate a time-averaged
    dataset from a higher-frequency input dataset.

    Attributes:
        input_dataset: Configuration for the source atmosphere dataset.
        window_timedelta: The time interval for the resampling operation,
            e.g., '5D' for 5-day averages.
        output_name_prefix: Prefix for output dataset naming.
        output_directory: Directory where the output datasets will be created.
        first_timestamp: Optional start timestamp for time slicing,
            e.g. '2016-01-01T06:00:00'. The 'origin' used in resampling is
            one step (of size window_timedelta) earlier than first_timestamp.
        last_timestamp: Optional final timestamp in case of extra unneeded windows.
        shift_timestamps_to_avg_interval_midpoint: If True, shift time axis labels
            backwards by half the window size.
        output_writer: Configuration for dask and xpartition.
        version: An optional version string, e.g. a date '2025-01-01', which is
            prepended to output_name_prefix.
    """

    input_dataset: InputDatasetConfig
    window_timedelta: str
    output_name: str
    output_directory: str
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    shift_timestamps_to_avg_interval_midpoint: bool = False
    output_writer: OutputWriterConfig = dataclasses.field(
        default_factory=OutputWriterConfig
    )
    version: str | None = None

    @classmethod
    def from_file(cls, path: str) -> "WindowAvgDatasetConfig":
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )

    def compute_window_avg(self, ds: xr.Dataset) -> xr.Dataset:
        dt = pd.Timedelta(self.window_timedelta).to_pytimedelta()
        origin = ds.time.sel(time=self.first_timestamp).item() - dt
        return (
            ds.resample(
                time=self.window_timedelta, closed="right", label="right", origin=origin
            )
            .mean()
            .sel(time=slice(None, self.last_timestamp))
        )

    def shift_timestamps(self, ds: xr.Dataset):
        """
        Shifts the time coordinate to the midpoint of the averaging interval.

        This is useful for models where the timestamp represents the end of an
        averaging period. This method shifts it to the middle of that period.

        Args:
            ds: The input xarray.Dataset with a 'time' coordinate.

        Returns:
            A new xr.Dataset with shifted timestamps if the config flag
            is True, otherwise the original dataset.

        """
        if self.shift_timestamps_to_avg_interval_midpoint:
            ds = shift_timestamps_to_midpoint(ds, time_dim="time")
            ds["time"].attrs["long_name"] = "time, avg interval midpoint"
        return ds


@click.command()
@click.option("--yaml", help="Path to dataset configuration YAML file.")
@click.option(
    "--debug",
    is_flag=True,
    help="Print metadata instead of writing output.",
)
@click.option(
    "--subsample", is_flag=True, help="Subsample one year of the data before writing."
)
def main(
    yaml: str,
    debug: bool,
    subsample: bool,
):
    config = WindowAvgDatasetConfig.from_file(yaml)
    print(f"Input atmosphere dataset: {config.input_dataset.zarr_path}")

    config.output_writer.start_dask_client(debug)

    ds = config.input_dataset.get_dataset()
    ds = config.compute_window_avg(ds)
    ds = config.shift_timestamps(ds)

    if config.version is None:
        version = datetime.today().strftime("%Y-%m-%d")
    else:
        version = config.version

    output_store = os.path.join(
        config.output_directory,
        f"{version}-{config.output_name}.zarr",
    )

    if subsample:
        output_store = output_store.replace(".zarr", "-subsample.zarr")
        ds = ds.isel(time=slice(None, 73))

    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        config.output_writer.write(ds, output_store)

    config.output_writer.close_dask_client()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        pdb.post_mortem()  # Start the debugger
        raise  # Re-raise the exception to preserve the traceback
