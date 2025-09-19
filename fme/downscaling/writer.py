import logging
from collections.abc import Mapping

import cftime
import fsspec
import gcsfs
import netCDF4
import numpy as np
import xarray as xr
import zarr

from fme.core.distributed import Distributed

logger = logging.getLogger(__name__)


def _encode_cftime_times(times, units="hours since 2000-01-01", calendar="julian"):
    return netCDF4.date2num(times, units=units, calendar=calendar)


def _check_for_overwrite(arr, insert_slices_tuple):
    # Loads the slice of zarr array
    existing = arr[insert_slices_tuple]

    fill_value = arr.fill_value

    if fill_value is None:
        # If no fill_value, assume all entries are valid → forbid any overwrite
        if np.any(existing):
            raise RuntimeError(
                f"Attempting to overwrite existing values in {arr.name} "
                f"at slice {insert_slices_tuple}"
            )
    elif np.isnan(fill_value):
        # Special case for NaN fill value
        conflict_mask = ~np.isnan(existing)
        if np.any(conflict_mask):
            raise RuntimeError(
                f"Attempting to overwrite existing values in {arr.name} "
                f"at slice {insert_slices_tuple}"
            )
    else:
        conflict_mask = existing != fill_value
        if np.any(conflict_mask):
            raise RuntimeError(
                f"Attempting to overwrite existing values in {arr.name} "
                f"at slice {insert_slices_tuple}"
            )


def _check_data_size_fits_slice(data: np.ndarray, insert_slices: Mapping[int, slice]):
    for dim_index, slice in insert_slices.items():
        if data.shape[dim_index] != slice.stop - slice.start:
            raise RuntimeError(
                "Size of data to insert into zarr must match slice. "
                f"Attempted to insert data with size {data.shape[dim_index]} "
                f"into {slice} of dimension index {dim_index}."
            )


def insert_into_zarr(
    path: str,
    data: Mapping[str, np.ndarray],
    insert_slices: Mapping[int, slice],
    overwrite_check: bool = True,
):
    root = zarr.open(path, mode="a")
    for var_name, var_data in data.items():
        n_dims = len(var_data.shape)
        # Array data is not loaded until index or slice is referenced
        zarr_array = root[var_name]
        insert_slices_tuple = tuple(
            insert_slices.get(dim_index, slice(None, None))
            for dim_index in range(n_dims)
        )
        _check_data_size_fits_slice(var_data, insert_slices)
        if overwrite_check:
            _check_for_overwrite(zarr_array, insert_slices_tuple)
        zarr_array[insert_slices_tuple] = var_data


def initialize_zarr(
    path: str,
    vars: list[str],
    dim_sizes: tuple,
    chunks: Mapping[str, int],
    shards: Mapping[str, int] | None,
    coords: dict[str, np.ndarray],
    dim_names: tuple,
    dtype="f4",
    array_attributes: dict[str, dict[str, str]] | None = None,
    group_attributes: dict[str, str] | None = None,
):
    """
    Initialize a Zarr group with the specified dimensions and chunk sizes.
    """
    root = zarr.open(path, mode="a")
    root.update_attributes(group_attributes or {})
    chunks_tuple = tuple(
        [chunks.get(dim, dim_sizes[d]) for d, dim in enumerate(dim_names)]
    )
    if shards is not None:
        shards_tuple = tuple(
            [shards.get(dim, dim_sizes[d]) for d, dim in enumerate(dim_names)]
        )
        for c, s in zip(chunks_tuple, shards_tuple):
            if s % c != 0:
                raise ValueError(f"Chunk shape {c} is not divisible by shard shape {s}")
    else:
        shards_tuple = None
    array_attributes = array_attributes or {}
    for var in vars:
        root.create_array(
            name=var,
            shape=dim_sizes,
            chunks=chunks_tuple,
            shards=shards_tuple,
            dtype=dtype,
            dimension_names=dim_names,
            attributes=array_attributes.get(var, {}),
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
        coords: dict[str, np.ndarray],
        data_vars: list[str],
        chunks: dict[str, int] | None = None,
        shards: dict[str, int] | None = None,
        array_attributes: dict[str, dict[str, str]] | None = None,
        group_attributes: dict[str, str] | None = None,
        allow_existing: bool = False,
        overwrite_check: bool = True,
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
        shards: Optional mapping to store multiple chunks in a single storage object.
        array_attributes: Optional mapping of variable name to a dictionary of array
            attributes, e.g. units and long_name.
        group_attributes: Optional dictionary of attributes to add to the zarr group.
        allow_existing: If true, allow writing to a preexisting store.
        overwrite_check: If true, check when recording each batch that the slice of the
            existing store does not already contain data.
        """
        self.path = path
        self.dims = dims
        self.coords = coords
        self.data_vars = data_vars
        self.chunks = chunks or {}
        self.shards = shards or None
        self.array_attributes = array_attributes
        self.group_attributes = group_attributes
        self.dist = Distributed.get_instance()
        self.overwrite_check = overwrite_check
        if allow_existing:
            self._store_initialized = True if self._path_exists() else False
        else:
            if self._path_exists() is True:
                raise FileExistsError(
                    f"Zarr store {self.path} already exists. "
                    "Please delete first before writing to this path, "
                    "or write to a different path."
                )
            self._store_initialized = False

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

    @classmethod
    def from_existing_store(cls, path: str) -> "ZarrWriter":
        ds = xr.open_zarr(path)
        coords = {k: v.values for k, v in ds.coords.items()}
        dims_list = [ds[var].dims for var in ds.data_vars]
        array_attributes = {var: dict(ds[var].attrs) for var in ds.data_vars}
        group_attributes = dict(ds.attrs)
        if not all(dims == dims_list[0] for dims in dims_list):
            raise ValueError(
                "Data arrays must have same dim order when writing "
                "to existing zarr store with ZarrWriter."
            )
        return cls(
            path=path,
            dims=dims_list[0],
            coords=coords,
            chunks=ds.chunks,
            data_vars=list(ds.data_vars),
            array_attributes=array_attributes,
            group_attributes=group_attributes,
            allow_existing=True,
        )

    def _path_exists(self) -> bool:
        if self.path.startswith("gs://"):
            fs = gcsfs.GCSFileSystem()
        else:
            fs = fsspec.filesystem("file")

        if fs.exists(self.path):
            return True
        else:
            return False

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
        insert_into_zarr(
            self.path, write_data, indexed_position_slices, self.overwrite_check
        )

    def create_zarr_store(
        self,
        example_data: Mapping[str, np.ndarray],
    ):
        if self.dist.is_root():
            logger.debug(f"Rank {self.dist.rank}: Initializing zarr store")
            data_dtype = example_data[self.data_vars[0]].dtype
            dim_sizes = tuple([len(self.coords[dim]) for dim in self.dims])
            initialize_zarr(
                path=self.path,
                vars=self.data_vars,
                dim_sizes=dim_sizes,
                chunks=self.chunks,
                shards=self.shards,
                coords=self.coords,
                dim_names=self.dims,
                dtype=data_dtype,
                array_attributes=self.array_attributes,
                group_attributes=self.group_attributes,
            )
            self.dist.barrier()
            self._store_initialized = True
        else:
            logger.debug(
                f"Rank {self.dist.rank}: Waiting for zarr store to be initialized"
            )
            self.dist.barrier()
            self._store_initialized = True
