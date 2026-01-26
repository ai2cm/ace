# copy_and_rename_variable_inplace.py
import logging

import click
import xarray as xr

logging.basicConfig(level=logging.INFO)


def copy_and_rename_variable_inplace(
    store_path: str,
    variable_name: str,
    new_name: str,
    debug: bool = False,
) -> None:
    """
    Copy an existing variable in a dataset and rename it in-place.

    Parameters
    ----------
    store_path : str
        Path to the zarr store.
    variable_name : str
        Name of the variable to copy.
    new_name : str
        New name for the copied variable.
    debug : bool
        If True, print dataset info instead of writing.
    """
    # Open dataset
    ds = xr.open_zarr(store_path, decode_timedelta=True)
    logging.info(f"Opened dataset with variables: {list(ds.data_vars)}")

    if variable_name not in ds:
        raise ValueError(f"Variable {variable_name} not found in {store_path}")

    if new_name in ds:
        raise ValueError(f"Variable {new_name} already exists in {store_path}")

    # Copy the variable
    ds[new_name] = ds[variable_name]

    logging.info(f"Variable {variable_name} copied to {new_name}")

    if debug:
        print(ds)
    else:
        # Write back to the same store
        ds.to_zarr(store_path, mode="a")
        logging.info(f"Dataset updated in-place at {store_path}")


@click.command()
@click.option("--store-path", help="Path to the Zarr store.", required=True)
@click.option("--variable-name", help="Variable to copy.", required=True)
@click.option("--new-name", help="New variable name.", required=True)
@click.option("--debug", is_flag=True, help="Print dataset instead of writing.")
def main(store_path, variable_name, new_name, debug):
    copy_and_rename_variable_inplace(
        store_path=store_path,
        variable_name=variable_name,
        new_name=new_name,
        debug=debug,
    )


if __name__ == "__main__":
    main()
