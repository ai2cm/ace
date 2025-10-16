import logging
from collections.abc import Mapping
from typing import Literal

import cftime
import fsspec
import numpy as np
import xarray as xr
import zarr

from fme.core.distributed import Distributed

logger = logging.getLogger(__name__)
DATETIME_ENCODING_UNITS = "microseconds since 1970-01-01"


def _encode_cftime_times(times, calendar="julian"):
    return cftime.date2num(times, units=DATETIME_ENCODING_UNITS, calendar=calendar)


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
    root = zarr.open_group(path, mode="r+")
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
    time_units: str = DATETIME_ENCODING_UNITS,
    time_calendar: str | None = "julian",
    nondim_coords: dict[str, xr.DataArray] | None = None,
    array_attributes: dict[str, dict[str, str]] | None = None,
    group_attributes: dict[str, str] | None = None,
    mode: str = "w-",
):
    """
    Initialize a Zarr group with the specified dimensions and chunk sizes.
    """
    root = zarr.open_group(path, mode=mode)
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

    if "time" in coords:
        times = coords.pop("time")
        time_ds = root.create_array(
            name="time",
            shape=(len(times),),
            dtype="int64",
            dimension_names=["time"],
        )
        if isinstance(times[0], cftime.datetime):
            time_coord_values = _encode_cftime_times(times, calendar=time_calendar)
            time_ds.attrs["units"] = DATETIME_ENCODING_UNITS
        else:
            time_coord_values = times
            time_ds.attrs["units"] = time_units

        if time_calendar:
            time_ds.attrs["calendar"] = time_calendar

        time_ds[:] = time_coord_values

    for key, values in coords.items():
        coord_ds = root.create_array(
            name=key, shape=values.shape, dtype=dtype, dimension_names=[key]
        )
        coord_ds[:] = values

    if nondim_coords is not None:
        for key, da in nondim_coords.items():
            nondim_coord_ds = root.create_array(
                name=key,
                shape=da.shape,
                dtype=da.dtype,
                dimension_names=list(da.dims),
            )
            nondim_coord_ds[:] = da.values
            nondim_coord_ds.attrs.update(da.attrs)

    array_attributes = array_attributes or {}

    written_coords = set()
    for var in vars:
        var_ds = root.create_array(
            name=var,
            shape=dim_sizes,
            chunks=chunks_tuple,
            shards=shards_tuple,
            dtype=dtype,
            dimension_names=dim_names,
            attributes=array_attributes.get(var, {}),
        )
        # Following xarray, only associate non-dimension coordinates
        # with a variable if the non-dimension coordinate's dimensions
        # are a subset of those of the variable:
        # https://github.com/pydata/xarray/blob/64704605a4912946d2835e54baa2905b8b4396a9/xarray/conventions.py#L670-L685
        associated_nondim_coords = [
            name
            for name, da in (nondim_coords or {}).items()
            if set(da.dims).issubset(dim_names)
        ]
        written_coords.update(associated_nondim_coords)
        var_ds.attrs["coordinates"] = " ".join(associated_nondim_coords)
    # Set a global "coordinates" attribute consisting of the non-dimension
    # coordinate names that were not appended to any variable-specific
    # "coordinates" attribute. This follows xarray's custom convention
    # for indicating this:
    # https://github.com/pydata/xarray/blob/64704605a4912946d2835e54baa2905b8b4396a9/xarray/conventions.py#L689-L740
    global_coords = set(nondim_coords or {}).difference(written_coords)
    if len(global_coords) > 0:
        root.attrs["coordinates"] = " ".join(sorted(global_coords))
    zarr.consolidate_metadata(root.store)


class ZarrWriter:
    def __init__(
        self,
        path: str,
        dims: tuple,
        coords: dict[str, np.ndarray],
        data_vars: list[str] | None = None,
        chunks: dict[str, int] | None = None,
        shards: dict[str, int] | None = None,
        array_attributes: dict[str, dict[str, str]] | None = None,
        group_attributes: dict[str, str] | None = None,
        mode: Literal["r+", "a", "w", "w-"] = "w-",
        overwrite_check: bool = True,
        time_units: str = DATETIME_ENCODING_UNITS,
        time_calendar: str | None = "julian",
        nondim_coords: dict[str, xr.DataArray] | None = None,
    ):
        """
        Initialize the ZarrWriter with the specified parameters.

        path: Zarr store is saved to this path
        dims: Order of data dimensions.
        coords: Dict mapping dimension names to their coordinates array.
            Note that dimensions
            that are often without coordinates (ex. sample) should be provided as an
            array of integer coordinates.
        data_vars: Variables to write. If None, all variables in data are saved.
        chunks: Optional mapping of dimension name to chunk size. If a dimension is not
            in this mapping, the chunk size will be the same as the dimension size.
        shards: Optional mapping to store multiple chunks in a single storage object.
        array_attributes: Optional mapping of variable name to a dictionary of array
            attributes, e.g. units and long_name.
        group_attributes: Optional dictionary of attributes to add to the zarr group.
        mode: Access mode used for the zarr.open_group call during store initialization.
            'r+' means read/write (must exist); 'a' means read/write (create if doesn’t
            exist); 'w' means create (overwrite if exists); 'w-' means create (fail if
            exists).
        overwrite_check: If true, check when recording each batch that the slice of the
            existing store does not already contain data.
        time_units: Units string for time coordinate. Defaults to
            "microseconds since 1970-01-01" if time coordinate is datetime and no units
            are provided.
        time_calendar: Calendar string for time coordinate if datetime
        nondim_coords: Optional mapping of coordinate name to DataArray for coordinates
            that are not associated with a dimension (ex. init_time, valid_time). Values
            are data arrays to allow for more freedom in these coords (e.g. can be
            multidimensional).
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
        self.time_units = time_units
        self.time_calendar = time_calendar
        self.nondim_coords = nondim_coords
        self.mode = mode
        if mode == "a" or mode == "r+":
            self._store_initialized = True if self._path_exists() else False
        else:
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

    def _path_exists(self) -> bool:
        fs = fsspec.url_to_fs(self.path)[0]

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
            self._create_zarr_store(example_data=data)
        indexed_position_slices = {
            self.dims.index(dim): position_slices[dim] for dim in position_slices.keys()
        }
        write_data = {v: data[v] for v in self.data_vars or data.keys()}
        insert_into_zarr(
            self.path, write_data, indexed_position_slices, self.overwrite_check
        )

    def _create_zarr_store(
        self,
        example_data: Mapping[str, np.ndarray],
    ):
        if self.dist.is_root():
            save_names = self.data_vars or list(example_data.keys())
            logger.debug(f"Rank {self.dist.rank}: Initializing zarr store")
            data_dtype = example_data[save_names[0]].dtype
            dim_sizes = tuple([len(self.coords[dim]) for dim in self.dims])
            initialize_zarr(
                path=self.path,
                vars=save_names,
                dim_sizes=dim_sizes,
                chunks=self.chunks,
                shards=self.shards,
                coords=self.coords,
                dim_names=self.dims,
                dtype=data_dtype,
                time_units=self.time_units,
                time_calendar=self.time_calendar,
                nondim_coords=self.nondim_coords,
                array_attributes=self.array_attributes,
                group_attributes=self.group_attributes,
                mode=self.mode,
            )
            self.dist.barrier()
            self._store_initialized = True
        else:
            logger.debug(
                f"Rank {self.dist.rank}: Waiting for zarr store to be initialized"
            )
            self.dist.barrier()
            self._store_initialized = True
