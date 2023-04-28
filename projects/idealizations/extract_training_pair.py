import click
import xarray


def validate_file_path(file_path: str, extension: str) -> None:
    """Validates the file path"""
    if not file_path.endswith(extension):
        raise ValueError("File path must be a netCDF file")


@click.command()
@click.option("--input_file", "-i", required=True, help="Path to the raw data")
@click.option("--output_file", "-o", required=True, help="Path to the output file")
@click.option(
    "--time_index",
    "-t",
    required=True,
    type=int,
    help="Time index t of the training pair, (t, t+1)",
)
def main(input_file: str, output_file: str, time_index: int) -> None:
    """Extracts a training pair from the raw data"""
    validate_file_path(input_file, ".nc")
    validate_file_path(output_file, ".nc")

    raw_data = xarray.open_dataset(input_file)
    time_slice = slice(time_index, time_index + 2)
    training_pair = raw_data.isel(time=time_slice)
    training_pair.to_netcdf(
        output_file, unlimited_dims=["time"], format="NETCDF4_CLASSIC"
    )
    print(f"Training pair extracted from {input_file} and saved to {output_file}")


if __name__ == "__main__":
    main()
