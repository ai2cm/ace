"""
Create a new netcdf file with the specified variables repeated n times in time.

Example usage:

```
python compute_repeating_forcing.py \
    --input_dir /net/nfs/climate/data/2023-08-11-vertically-resolved-1deg-fme-ensemble-dataset/validation/ic_0011 \
    --output_dir /tmp/repeating_climate/ \
    --n_times 2 \
    -f DSWRFtoa \
    -f surface_temperature \
    -f ocean_fraction \
    -f land_fraction \
    -f sea_ice_fraction
```
"""  # noqa: E501

from glob import glob
from pathlib import Path
from typing import Dict

import click
import numpy as np
import xarray as xr


def get_time_invariant_vars(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    arrays = {}
    for var in ds.data_vars:
        if "time" not in ds[var].dims:
            arrays[str(var)] = ds[var]
    return arrays


@click.command()
@click.option(
    "--n_times",
    type=int,
    help=(
        "Total number of repeats, e.g. 2-times means that the original forcing "
        "is there plus and additional repeat."
    ),
    required=True,
)
@click.option(
    "--input_dir",
    type=Path,
    help=("Directory containing netcdf files, e.g. from a single ensemble member"),
    required=True,
)
@click.option(
    "--output_dir",
    type=Path,
    help="Where to store the output netcdf files.",
    required=True,
)
@click.option(
    "-f",
    "--repeat_variables",
    help="Variables to repeat (usually forcing and prescribed variables)",
    default=["surface_temperature"],
    multiple=True,
)
@click.option("--nc-format", default="NETCDF3_64BIT", help="netCDF file format")
def main(n_times: int, input_dir: Path, output_dir: Path, repeat_variables, nc_format):
    click.echo("Loading data...", nl=False)
    files = sorted(glob(str(input_dir / "*.nc")))
    arrays = get_time_invariant_vars(xr.open_dataset(files[0], decode_timedelta=False))
    ds = xr.open_mfdataset(files)
    click.echo("done")

    click.echo("Detected time-invariant variables:")
    for name in arrays:
        click.echo(f'"{name}": {arrays[name].shape}')

    # initial condition
    ic_vars = []
    for var in ds.data_vars:
        if var not in repeat_variables and var not in arrays:
            ic_vars.append(str(var))
            arrays[str(var)] = (
                ds[var]
                .isel(time=0)
                .expand_dims("initial_condition", axis=0)
                .copy(deep=True)
            )

    time_axis = 0
    for var in repeat_variables:
        click.echo(f'Repeating "{var}"')
        arrays[var] = xr.DataArray(  # type: ignore
            np.repeat(ds[var].data, n_times, axis=time_axis), dims=ds[var].dims
        )

    dt: int = (ds.time[1] - ds.time[0]).values.astype("timedelta64[h]").astype(int)
    time_coord = xr.date_range(
        ds.time.item(0),
        periods=len(ds.time) * n_times,
        freq=f"{dt}h",
        calendar=ds.time.dt.calendar,
        use_cftime=True,
    )

    coords = {**ds.coords, "time": time_coord, "initial_condition": [0]}
    monthly_ds = xr.Dataset(arrays, coords=coords).resample(time="MS")
    bytes = 0.0
    history = (
        f"Created by running "
        f"`full_model/project/data_process/compute_repeating_forcing.py --input_dir "
        f"{input_dir} -f {' -f '.join(repeat_variables)}`"
    )
    for i, (label, data) in enumerate(monthly_ds):
        if i != 0:
            data = data.drop_vars(ic_vars)
        bytes += data.nbytes
        click.echo(
            f"Processing month {i + 1}/{len(monthly_ds)} ({(bytes / 1e9):.2f} GB)\r",
            nl=False,
        )
        # use these options to enable opening data with netCDF4.MFDataset
        outpath = (output_dir / label.strftime("%Y%m%d%H")).with_suffix(".nc")
        data.attrs["history"] = history
        data.to_netcdf(outpath, unlimited_dims=["time"], format=nc_format)

    click.echo("")


if __name__ == "__main__":
    main()  # type: ignore
