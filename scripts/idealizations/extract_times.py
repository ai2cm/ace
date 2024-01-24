from typing import Optional, Sequence

import click
import xarray
from utils import Grid, roundtrip, validate_file_path


# fmt: off
@click.command()
@click.option(
    "--input_file", "-i", required=True, help="Path to the raw data"
)  # TODO(gideond) use click.File
@click.option(
    "--output_file", "-o", required=True, help="Path to the output file"
)
@click.option(
    "--smooth", "-m", is_flag=True, required=False, default=False, type=bool, help="Whether to smooth the data using an SHT roundtrip."  # noqa: E501
)
@click.option(
    "--grid", "-g", required=False, default=Grid.EQUIANGULAR, type=Grid, help="Grid type for the SHT."  # noqa: E501
)
@click.option(
    "--index", "-x", required=True, multiple=True, type=int, help="Time index of the data to extract. Can be passed multiple times, e.g. -x 0 -x 1 yields [0,1]."  # noqa: E501
)
@click.option(
    "--repeat", "-r", required=False, default=1, type=int, help="Number of times to repeat the extracted data, e.g. -r 2 yields [0,1,0,1].",  # noqa: E501
)
@click.option(
    "--final_index", "-z", required=False, default=None, type=int, help="Final time index of the data to extract, e.g. -x 0 -x 1 -r 2 -z 42 yields [0,1,0,1,42].",  # noqa: E501
)
def main(
    input_file: str,
    output_file: str,
    smooth: bool,
    index: Sequence[int],
    repeat: int = 1,
    final_index: Optional[int] = None,
    grid: Grid = Grid.EQUIANGULAR,
) -> None:
    """Extracts a training pair from the raw data"""
    validate_file_path(input_file, ".nc")
    validate_file_path(output_file, ".nc")

    indices = list(list(index) * repeat)
    if final_index is not None:
        indices.append(final_index)

    ds = xarray.open_dataset(input_file)
    ds_subset = ds.isel(time=indices)

    nlat, nlon = len(ds_subset["grid_yt"]), len(ds_subset["grid_xt"])  # hard coded
    if smooth:
        ds_subset = roundtrip(ds_subset, nlat, nlon, grid)

    ds_subset.to_netcdf(output_file, unlimited_dims=["time"], format="NETCDF4_CLASSIC")
    print(f"Training pair extracted from {input_file} and saved to {output_file}")


if __name__ == "__main__":
    main()
