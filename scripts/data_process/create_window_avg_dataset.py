import dataclasses
import logging
import pdb
import sys

import click
import dacite
import pandas as pd
import xarray as xr
import yaml
from time_utils import shift_timestamps_to_midpoint
from writer_utils import OutputWriterConfig


@dataclasses.dataclass
class WindowAvgInputDatasetConfig:
    zarr_path: str
    time_chunk_size: int

    def get_dataset(self) -> xr.Dataset:
        return xr.open_zarr(self.zarr_path, chunks={"time": self.time_chunk_size})


@dataclasses.dataclass
class WindowAvgDatasetConfig:
    """
    Configuration for creating a window-averaged dataset.

    This class defines the parameters needed to generate a time-averaged
    dataset from a higher-frequency input dataset.

    Attributes:
        window_timedelta: The time interval for the resampling operation,
            e.g., '120h' for 5-day averages. Must be specified in units that
            are "Tick-like" (h, m, s, ms, us).
        first_timestamp: Optional start timestamp for time slicing,
            e.g. '2016-01-01T06:00:00'. The 'origin' used in resampling is
            one step (of size window_timedelta) earlier than first_timestamp.
        last_timestamp: Optional final timestamp in case of extra unneeded windows.
            Used after window averaging but before shifting timestamps.
        shift_timestamps_to_avg_interval_midpoint: If True, shift time axis labels
            backwards by half the window size.
        time_dim: Name of the time dimension.
        subset_names: Optional list of data variable names to subset the input dataset.
    """

    window_timedelta: str
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    shift_timestamps_to_avg_interval_midpoint: bool = False
    time_dim: str = "time"
    subset_names: list[str] | None = None

    def _compute_window_avg(self, ds: xr.Dataset) -> xr.Dataset:
        if self.subset_names is not None:
            ds = ds[self.subset_names]

        # Split dataset into time-varying and time-invariant variables
        time_varying_vars = [
            var for var in ds.data_vars if self.time_dim in ds[var].dims
        ]
        time_invariant_vars = [
            var for var in ds.data_vars if self.time_dim not in ds[var].dims
        ]

        if len(time_varying_vars) == 0:
            raise ValueError("There are no time-varying variables in the dataset.")

        ds_time_varying = ds[time_varying_vars]
        dt = pd.Timedelta(self.window_timedelta).to_pytimedelta()
        origin = ds_time_varying.time.sel(time=self.first_timestamp).item() - dt
        ds_time_varying_avg = (
            ds_time_varying.resample(
                time=self.window_timedelta, closed="right", label="right", origin=origin
            )
            .mean()
            .sel(time=slice(None, self.last_timestamp))
        )

        # Merge with time-invariant variables
        if time_invariant_vars:
            ds_time_invariant = ds[time_invariant_vars]
            ds = xr.merge([ds_time_varying_avg, ds_time_invariant])
        else:
            ds = ds_time_varying_avg

        logging.info(
            f"After _compute_window_avg time coord:\n {str(ds[self.time_dim])}"
        )
        return ds

    def _shift_timestamps(self, ds: xr.Dataset):
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
            logging.info(
                f"After _shift_timestamps time coord:\n {str(ds[self.time_dim])}"
            )
        return ds

    def get_window_avg(self, ds: xr.Dataset) -> xr.Dataset:
        logging.info(f"Input time coord:\n {str(ds[self.time_dim])}")
        ds = self._compute_window_avg(ds)
        ds = self._shift_timestamps(ds)
        return ds


@dataclasses.dataclass
class CreateWindowAvgDatasetConfig:
    """
    Top-level runner config.

    Attributes:
        input_dataset: Configuration for the source atmosphere dataset.
        window_avg: Configuration for creating the window-averaged dataset.
        output_zarr_path: Full path to the output zarr store.
        output_writer: Configuration for dask and xpartition.
    """

    input_dataset: WindowAvgInputDatasetConfig
    window_avg: WindowAvgDatasetConfig
    output_zarr_path: str
    output_writer: OutputWriterConfig = dataclasses.field(
        default_factory=OutputWriterConfig
    )

    def __post_init__(self):
        logging.info(f"Input dataset: {self.input_dataset.zarr_path}")

    @classmethod
    def from_file(cls, path: str) -> "CreateWindowAvgDatasetConfig":
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )


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
    logging.basicConfig(level=logging.INFO)

    config = CreateWindowAvgDatasetConfig.from_file(yaml)

    config.output_writer.start_dask_client(debug)
    ds = config.input_dataset.get_dataset()
    ds = config.window_avg.get_window_avg(ds)

    output_store = config.output_zarr_path
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
