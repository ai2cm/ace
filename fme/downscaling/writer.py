from typing import Dict, List, Mapping

import cftime
import fsspec
import gcsfs
import netCDF4
import numpy as np
import zarr


def _encode_cftime_times(times, units="hours since 2000-01-01", calendar="julian"):
    return netCDF4.date2num(times, units=units, calendar=calendar)


def insert_into_zarr(
    path: str, data: Mapping[str, np.ndarray], insert_slices: Mapping[int, slice]
):
    root = zarr.open(path, mode="a")
    for var_name, var_data in data.items():
        n_dims = len(var_data.shape)
        insert_slices_tuple = tuple(
            [
                insert_slices.get(dim_index, slice(None, None))
                for dim_index in range(n_dims)
            ]
        )
        root[var_name][insert_slices_tuple] = var_data


def initialize_zarr(
    path: str,
    vars: List[str],
    dim_sizes: tuple,
    chunks: Mapping[str, int],
    coords: Dict[str, np.ndarray],
    dim_names: tuple,
    dtype="f4",
):
    """
    Initialize a Zarr group with the specified dimensions and chunk sizes.
    """
    root = zarr.open(path, mode="a")
    chunks_tuple = tuple(
        [chunks.get(dim, dim_sizes[d]) for d, dim in enumerate(dim_names)]
    )
    for var in vars:
        root.create_array(
            name=var,
            shape=dim_sizes,
            chunks=chunks_tuple,
            dtype=dtype,
            dimension_names=dim_names,
        )

    if "time" in coords:
        times = coords.pop("time")
        if isinstance(times[0], cftime.datetime):
            encoded_times = _encode_cftime_times(
                times, units="hours since 2000-01-01", calendar="julian"
            )
        else:
            encoded_times = times

        time_ds = root.create_array(
            name="time",
            shape=(len(times),),
            dtype=dtype,
            dimension_names=["time"],
        )
        time_ds[:] = encoded_times
        time_ds.attrs["units"] = "hours since 2000-01-01"
        time_ds.attrs["calendar"] = "julian"

    for key, values in coords.items():
        coord_ds = root.create_array(
            name=key,
            shape=values.shape,
            dtype=dtype,
            dimension_names=[key],
        )
        coord_ds[:] = values
    zarr.consolidate_metadata(root.store)


class ZarrWriter:
    def __init__(
        self,
        path: str,
        dims: tuple,
        coords: Dict[str, np.ndarray],
        data_vars: List[str],
        chunks: Dict[str, int] | None = None,
    ):
        """
        Initialize the ZarrWriter with the specified parameters.

        path: Zarr store is saved to this path
        dims: Order of data dimensions.
        coords: Dict mapping dimension names to their coordinates array.
            Note that dimensions
            that are often without coordinates (ex. sample) should be provided as an
            array of integer coordinates.
        data_vars: Variables to write
        chunks: Optional mapping of dimension name to chunk size. If a dimension is not
            in this mapping, the chunk size will be the same as the dimension size.
        """
        self.path = path
        self.dims = dims
        self.coords = coords
        self.data_vars = data_vars
        self.chunks = chunks or {}

        self._store_initialized = False

        self._verify_path_empty()

        for coord, coord_arr in coords.items():
            if len(coord_arr.shape) != 1:
                raise ValueError(
                    f"{coord} must be 1D array. Found shape {coord_arr.shape}"
                )

        for dim in self.dims:
            if dim not in self.coords:
                raise ValueError(
                    f"Missing coordinate for dimension {dim}. "
                    "For dimensionless axes (ex. sample), an integer array "
                    "should be provided."
                )

    def _verify_path_empty(self):
        if self.path.startswith("gs://"):
            fs = gcsfs.GCSFileSystem()
        else:
            fs = fsspec.filesystem("file")

        if fs.exists(self.path):
            raise FileExistsError(
                f"Zarr store {self.path} already exists. "
                "Please delete first before writing to this path, "
                "or write to a different path."
            )

    def record_batch(
        self, data: Mapping[str, np.ndarray], position_slices: Mapping[str, slice]
    ):
        """
        Writes a batch of data into the zarr store.
        data: Mapping of variable name to data array.
        position_slices: Mapping of dimension name to the slice
            along that dimension axis to insert data.
        """
        if not self._store_initialized:
            self.create_zarr_store(example_data=data)
        indexed_position_slices = {
            self.dims.index(dim): position_slices[dim] for dim in position_slices.keys()
        }
        write_data = {v: data[v] for v in self.data_vars}
        insert_into_zarr(self.path, write_data, indexed_position_slices)

    def create_zarr_store(
        self,
        example_data: Mapping[str, np.ndarray],
    ):
        data_dtype = example_data[self.data_vars[0]].dtype
        dim_sizes = tuple([len(self.coords[dim]) for dim in self.dims])
        initialize_zarr(
            path=self.path,
            vars=self.data_vars,
            dim_sizes=dim_sizes,
            chunks=self.chunks,
            coords=self.coords,
            dim_names=self.dims,
            dtype=data_dtype,
        )
        self._store_initialized = True
