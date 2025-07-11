import datetime
import json
import logging
import os
import re
import warnings
from collections import namedtuple
from functools import lru_cache
from urllib.parse import urlparse

import fsspec
import numpy as np
import torch
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

from fme.core.coordinates import (
    DepthCoordinate,
    HorizontalCoordinates,
    HybridSigmaPressureCoordinate,
    NullVerticalCoordinate,
    VerticalCoordinate,
)
from fme.core.dataset.properties import DatasetProperties
from fme.core.mask_provider import MaskProvider
from fme.core.stacker import Stacker
from fme.core.typing_ import TensorDict

from .config import XarrayDataConfig
from .data_typing import VariableMetadata
from .utils import (
    as_broadcasted_tensor,
    get_horizontal_coordinates,
    get_nonspacetime_dimensions,
    load_series_data,
    load_series_data_zarr_async,
)

SLICE_NONE = slice(None)
logger = logging.getLogger(__name__)

VariableNames = namedtuple(
    "VariableNames",
    (
        "time_dependent_names",
        "time_invariant_names",
        "static_derived_names",
    ),
)


def _get_vertical_coordinate(
    ds: xr.Dataset, dtype: torch.dtype | None
) -> VerticalCoordinate:
    """
    Get vertical coordinate from a dataset.

    If the dataset contains variables named `ak_N` and `bk_N` where
    `N` is the level number, then a hybrid sigma-pressure coordinate
    will be returned. If the dataset contains variables named
    `idepth_N` then a depth coordinate will be returned. If neither thing
    is true, a hybrid sigma-pressure coordinate of lenght 0 is returned.

    Args:
        ds: Dataset to get vertical coordinates from.
        dtype: Data type of the returned tensors. If None, the dtype is not
            changed from the original in ds.
    """
    ak_mapping = {
        int(v[3:]): torch.as_tensor(ds[v].values)
        for v in ds.variables
        if v.startswith("ak_")
    }
    bk_mapping = {
        int(v[3:]): torch.as_tensor(ds[v].values)
        for v in ds.variables
        if v.startswith("bk_")
    }
    ak_list = [ak_mapping[k] for k in sorted(ak_mapping.keys())]
    bk_list = [bk_mapping[k] for k in sorted(bk_mapping.keys())]

    idepth_mapping = {
        int(v[7:]): torch.as_tensor(ds[v].values)
        for v in ds.variables
        if v.startswith("idepth_")
    }
    idepth_list = [idepth_mapping[k] for k in sorted(idepth_mapping.keys())]

    if len(ak_list) > 0 and len(bk_list) > 0 and len(idepth_list) > 0:
        raise ValueError(
            "Dataset contains both hybrid sigma-pressure and depth coordinates. "
            "Can only provide one, or else the vertical coordinate is ambiguous."
        )

    coordinate: VerticalCoordinate
    surface_mask = None
    if len(idepth_list) > 0:
        if "mask_0" in ds.data_vars:
            mask_layers = {
                name: torch.as_tensor(ds[name].values, dtype=dtype)
                for name in ds.data_vars
                if re.match(r"mask_(\d+)$", name)
            }
            for name in mask_layers:
                if "time" in ds[name].dims:
                    raise ValueError("The ocean mask must by time-independent.")
            stacker = Stacker({"mask": ["mask_"]})
            mask = stacker("mask", mask_layers)
            if "surface_mask" in ds.data_vars:
                if "time" in ds["surface_mask"].dims:
                    raise ValueError("The surface mask must be time-independent.")
                surface_mask = torch.as_tensor(ds["surface_mask"].values, dtype=dtype)
        else:
            logger.warning(
                "Dataset does not contain a mask. Providing a DepthCoordinate with "
                "mask set to 1 at all layers."
            )
            mask = torch.ones(len(idepth_list) - 1, dtype=dtype)
        coordinate = DepthCoordinate(
            torch.as_tensor(idepth_list, dtype=dtype), mask, surface_mask
        )
    elif len(ak_list) > 0 and len(bk_list) > 0:
        coordinate = HybridSigmaPressureCoordinate(
            ak=torch.as_tensor(ak_list, dtype=dtype),
            bk=torch.as_tensor(bk_list, dtype=dtype),
        )
    else:
        logger.warning("Dataset does not contain a vertical coordinate.")
        coordinate = NullVerticalCoordinate()

    return coordinate


def _get_raw_times(paths: list[str], engine: str) -> list[np.ndarray]:
    times = []
    for path in paths:
        with _open_xr_dataset(path, engine=engine) as ds:
            times.append(ds.time.values)
    return times


def _repeat_and_increment_time(
    raw_times: list[np.ndarray], n_repeats: int, timestep: datetime.timedelta
) -> list[np.ndarray]:
    """Repeats and increments a collection of arrays of evenly spaced times."""
    n_timesteps = sum(len(times) for times in raw_times)
    timespan = timestep * n_timesteps

    repeated_and_incremented_time = []
    for repeats in range(n_repeats):
        increment = repeats * timespan
        for time in raw_times:
            incremented_time = time + increment
            repeated_and_incremented_time.append(incremented_time)
    return repeated_and_incremented_time


def _get_cumulative_timesteps(time: list[np.ndarray]) -> np.ndarray:
    """Returns a list of cumulative timesteps for each item in a time coordinate."""
    num_timesteps_per_file = [0]
    for time_coord in time:
        num_timesteps_per_file.append(len(time_coord))
    return np.array(num_timesteps_per_file).cumsum()


def _get_file_local_index(index: int, start_indices: np.ndarray) -> tuple[int, int]:
    """
    Return a tuple of the index of the file containing the time point at `index`
    and the index of the time point within that file.
    """
    file_index = np.searchsorted(start_indices, index, side="right") - 1
    time_index = index - start_indices[file_index]
    return int(file_index), time_index


class StaticDerivedData:
    names = ("x", "y", "z")
    metadata = {
        "x": VariableMetadata(units="", long_name="Euclidean x-coordinate"),
        "y": VariableMetadata(units="", long_name="Euclidean y-coordinate"),
        "z": VariableMetadata(units="", long_name="Euclidean z-coordinate"),
    }

    def __init__(self, coordinates: HorizontalCoordinates):
        self._coords = coordinates
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._z: torch.Tensor | None = None

    def _get_xyz(self) -> TensorDict:
        if self._x is None or self._y is None or self._z is None:
            coords = self._coords
            x, y, z = coords.xyz

            self._x = torch.as_tensor(x)
            self._y = torch.as_tensor(y)
            self._z = torch.as_tensor(z)

        return {"x": self._x, "y": self._y, "z": self._z}

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._get_xyz()[name]


def _get_protocol(path):
    return urlparse(str(path)).scheme


def _get_fs(path):
    protocol = _get_protocol(path)
    if not protocol:
        protocol = "file"
    proto_kw = _get_fs_protocol_kwargs(path)
    fs = fsspec.filesystem(protocol, **proto_kw)

    return fs


def _preserve_protocol(original_path, glob_paths):
    protocol = _get_protocol(str(original_path))
    if protocol:
        glob_paths = [f"{protocol}://{path}" for path in glob_paths]
    return glob_paths


def _get_fs_protocol_kwargs(path):
    protocol = _get_protocol(path)
    kwargs = {}
    if protocol == "gs":
        # https://gcsfs.readthedocs.io/en/latest/api.html#gcsfs.core.GCSFileSystem
        key_json = os.environ.get("FSSPEC_GS_KEY_JSON", None)
        key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)

        if key_json is not None:
            token = json.loads(key_json)
        elif key_file is not None:
            token = key_file
        else:
            logger.warning(
                "GCS currently expects user credentials authenticated using"
                " `gcloud auth application-default login`. This is not recommended for "
                "production use."
            )
            token = "google_default"
        kwargs["token"] = token
    elif protocol == "s3":
        # https://s3fs.readthedocs.io/en/latest/#s3-compatible-storage
        env_vars = [
            "FSSPEC_S3_KEY",
            "FSSPEC_S3_SECRET",
            "FSSPEC_S3_ENDPOINT_URL",
        ]
        for v in env_vars:
            if v not in os.environ:
                warnings.warn(
                    f"An S3 path was specified but environment variable {v} "
                    "was not found. This may cause authentication issues if not "
                    "set and no other defaults are present. See "
                    "https://s3fs.readthedocs.io/en/latest/#s3-compatible-storage"
                    " for details."
                )

    return kwargs


def _open_xr_dataset(path: str, *args, **kwargs):
    # need the path to get protocol specific arguments for the backend
    protocol_kw = _get_fs_protocol_kwargs(path)
    if protocol_kw:
        kwargs.update({"storage_options": protocol_kw})

    return xr.open_dataset(
        path,
        *args,
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
        mask_and_scale=False,
        cache=False,
        chunks=None,
        **kwargs,
    )


_open_xr_dataset_lru = lru_cache()(_open_xr_dataset)


def _open_file_fh_cached(path, **kwargs):
    protocol = _get_protocol(path)
    if protocol:
        # add an LRU cache for remote zarrs
        return _open_xr_dataset_lru(
            path,
            **kwargs,
        )
    # netcdf4 and h5engine have a filehandle LRU cache in xarray
    # https://github.com/pydata/xarray/blob/cd3ab8d5580eeb3639d38e1e884d2d9838ef6aa1/xarray/backends/file_manager.py#L54 # noqa: E501
    return _open_xr_dataset(
        path,
        **kwargs,
    )


def get_raw_paths(path, file_pattern):
    fs = _get_fs(path)
    glob_paths = sorted(fs.glob(os.path.join(path, file_pattern)))
    raw_paths = _preserve_protocol(path, glob_paths)
    return raw_paths


def _get_mask_provider(ds: xr.Dataset, dtype: torch.dtype | None) -> MaskProvider:
    """
    Get mask provider from a dataset.

    If the dataset contains static variables that start with the string "mask_" or a
    variable named "surface_mask", then these variables will be used to instantiate
    a MaskProvider object. Otherwise, an empty MaskProvider is returned.

    Args:
        ds: Dataset to get vertical coordinates from.
        dtype: Data type of the returned tensors. If None, the dtype is not
            changed from the original in ds.
    """
    masks: dict[str, torch.Tensor] = {
        name: torch.as_tensor(ds[name].values, dtype=dtype)
        for name in ds.data_vars
        if "mask_" in name
    }
    for name in masks:
        if "time" in ds[name].dims:
            raise ValueError("Masks must be time-independent.")
    mask_provider = MaskProvider(masks)
    logging.info(f"Initialized {mask_provider}.")
    return mask_provider


class XarrayDataset(torch.utils.data.Dataset):
    """Load data from a directory of files matching a pattern using xarray. The
    number of contiguous timesteps to load for each sample is specified by the
    n_timesteps argument.

    For example, if the file(s) have the time coordinate
    (t0, t1, t2, t3, t4) and n_timesteps=3, then this dataset will
    provide three samples: (t0, t1, t2), (t1, t2, t3), and (t2, t3, t4).
    """

    def __init__(self, config: XarrayDataConfig, names: list[str], n_timesteps: int):
        self._horizontal_coordinates: HorizontalCoordinates
        self._names = names
        self.path = config.data_path
        self.file_pattern = config.file_pattern
        self.engine = config.engine
        self.dtype = config.torch_dtype
        self.spatial_dimensions = config.spatial_dimensions
        self.fill_nans = config.fill_nans
        self.subset_config = config.subset
        self._raw_paths = get_raw_paths(self.path, self.file_pattern)
        if len(self._raw_paths) == 0:
            raise ValueError(
                f"No files found matching '{self.path}/{self.file_pattern}'."
            )
        self.full_paths = self._raw_paths * config.n_repeats
        self.sample_n_times = n_timesteps
        self._get_files_stats(config.n_repeats, config.infer_timestep)
        first_dataset = xr.open_dataset(
            self.full_paths[0],
            decode_times=False,
            decode_timedelta=False,
            engine=self.engine,
            chunks=None,
        )
        self._mask_provider = _get_mask_provider(first_dataset, self.dtype)
        (
            self._horizontal_coordinates,
            self._static_derived_data,
            _loaded_horizontal_dims,
        ) = self.configure_horizontal_coordinates(first_dataset)
        (
            self._time_dependent_names,
            self._time_invariant_names,
            self._static_derived_names,
        ) = self._group_variable_names_by_time_type()

        self._vertical_coordinate = _get_vertical_coordinate(first_dataset, self.dtype)
        self.overwrite = config.overwrite

        self._nonspacetime_dims = get_nonspacetime_dimensions(
            first_dataset, _loaded_horizontal_dims
        )
        self._shape_excluding_time = [
            first_dataset.sizes[dim]
            for dim in (self._nonspacetime_dims + _loaded_horizontal_dims)
        ]
        self._loaded_dims = ["time"] + self._nonspacetime_dims + _loaded_horizontal_dims
        self.isel = {
            dim: v if isinstance(v, int) else v.slice for dim, v in config.isel.items()
        }
        self._isel_tuple = tuple(
            [self.isel.get(dim, SLICE_NONE) for dim in self._loaded_dims[1:]]
        )
        self._check_isel_dimensions(first_dataset.sizes)

    def _check_isel_dimensions(self, data_dim_sizes):
        # Horizontal dimensions are not currently supported, as the current isel code
        # does not adjust HorizonalCoordinates to match selection.
        if "time" in self.isel:
            raise ValueError("isel cannot be used to select time. Use subset instead.")

        for dim, selection in self.isel.items():
            if dim not in self._nonspacetime_dims:
                raise ValueError(
                    f"isel dimension {dim} must be a non-spacetime dimension "
                    f"of the dataset ({self._nonspacetime_dims})."
                )
            max_isel_index = (
                (selection.start or 0) if isinstance(selection, slice) else selection
            )
            if max_isel_index >= data_dim_sizes[dim]:
                raise ValueError(
                    f"isel index {max_isel_index} is out of bounds for dimension "
                    f"{dim} with size {data_dim_sizes[dim]}."
                )

    @property
    def _shape_excluding_time_after_selection(self):
        final_shape = []
        for orig_size, sel in zip(self._shape_excluding_time, self._isel_tuple):
            # if selecting a single index, dimension is squeezed
            # so it is not included in the final shape
            if isinstance(sel, slice):
                if sel.start is None and sel.stop is None and sel.step is None:
                    final_shape.append(orig_size)
                else:
                    final_shape.append(len(range(*sel.indices(orig_size))))
        return final_shape

    @property
    def dims(self) -> list[str]:
        # Final dimensions of returned data after dims that are selected
        # with a single index are dropped
        final_dims = ["time"]
        for dim, sel in zip(self._loaded_dims[1:], self._isel_tuple):
            if isinstance(sel, slice):
                final_dims.append(dim)
        return final_dims

    @property
    def properties(self) -> DatasetProperties:
        return DatasetProperties(
            self._variable_metadata,
            self._vertical_coordinate,
            self._horizontal_coordinates,
            self._mask_provider,
            self.timestep,
            self._is_remote,
        )

    @property
    def _is_remote(self) -> bool:
        protocol = _get_protocol(str(self.path))
        if not protocol or protocol == "file":
            return False
        return True

    @property
    def all_times(self) -> xr.CFTimeIndex:
        """Time index of all available times in the data."""
        return self._all_times

    def _get_variable_metadata(self, ds):
        result = {}
        for name in self._names:
            if name in StaticDerivedData.names:
                result[name] = StaticDerivedData.metadata[name]
            elif hasattr(ds[name], "units") and hasattr(ds[name], "long_name"):
                result[name] = VariableMetadata(
                    units=ds[name].units,
                    long_name=ds[name].long_name,
                )
        self._variable_metadata = result

    def _get_files_stats(self, n_repeats: int, infer_timestep: bool):
        logging.info(f"Opening data at {os.path.join(self.path, self.file_pattern)}")
        raw_times = _get_raw_times(self._raw_paths, engine=self.engine)

        self._timestep: datetime.timedelta | None
        if infer_timestep:
            self._timestep = _get_timestep(np.concatenate(raw_times))
            time_coord = _repeat_and_increment_time(raw_times, n_repeats, self.timestep)
        else:
            self._timestep = None
            time_coord = raw_times

        cum_num_timesteps = _get_cumulative_timesteps(time_coord)
        self.start_indices = cum_num_timesteps[:-1]
        self.total_timesteps = cum_num_timesteps[-1]
        self._n_initial_conditions = self.total_timesteps - self.sample_n_times + 1
        self._sample_start_times = xr.CFTimeIndex(
            np.concatenate(time_coord)[: self._n_initial_conditions]
        )
        self._all_times = xr.CFTimeIndex(np.concatenate(time_coord))

        del cum_num_timesteps, time_coord

        ds = self._open_file(0)
        self._get_variable_metadata(ds)

        logging.info(f"Found {self._n_initial_conditions} samples.")

    def _group_variable_names_by_time_type(self) -> VariableNames:
        """Returns lists of time-dependent variable names, time-independent
        variable names, and variables which are only present as an initial
        condition.
        """
        (
            time_dependent_names,
            time_invariant_names,
            static_derived_names,
        ) = ([], [], [])
        # Don't use open_mfdataset here, because it will give time-invariant
        # fields a time dimension. We assume that all fields are present in the
        # netcdf file corresponding to the first chunk of time.
        with _open_xr_dataset(self.full_paths[0], engine=self.engine) as ds:
            for name in self._names:
                if name in StaticDerivedData.names:
                    static_derived_names.append(name)
                else:
                    try:
                        da = ds[name]
                    except KeyError:
                        raise ValueError(
                            f"Required variable not found in dataset: {name}."
                        )
                    else:
                        dims = da.dims
                        if "time" in dims:
                            time_dependent_names.append(name)
                        else:
                            time_invariant_names.append(name)
            logging.info(
                f"The required variables have been found in the dataset: {self._names}."
            )

        return VariableNames(
            time_dependent_names,
            time_invariant_names,
            static_derived_names,
        )

    def configure_horizontal_coordinates(
        self, first_dataset
    ) -> tuple[HorizontalCoordinates, StaticDerivedData, list[str]]:
        horizontal_coordinates: HorizontalCoordinates
        static_derived_data: StaticDerivedData

        horizontal_coordinates, dim_names = get_horizontal_coordinates(
            first_dataset, self.spatial_dimensions, self.dtype
        )
        static_derived_data = StaticDerivedData(horizontal_coordinates)

        coords_sizes = {
            coord_name: len(coord)
            for coord_name, coord in horizontal_coordinates.coords.items()
        }
        logging.info(f"Horizontal coordinate sizes are {coords_sizes}.")
        return horizontal_coordinates, static_derived_data, dim_names

    @property
    def timestep(self) -> datetime.timedelta:
        if self._timestep is None:
            raise ValueError(
                "Timestep was not inferred in the data loader. Note "
                "XarrayDataConfig.infer_timestep must be set to True for this "
                "to occur."
            )
        else:
            return self._timestep

    def __len__(self):
        return self._n_initial_conditions

    def _open_file(self, idx):
        logger.debug(f"Opening file {self.full_paths[idx]}")
        return _open_file_fh_cached(self.full_paths[idx], engine=self.engine)

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        """Return cftime index corresponding to start time of each sample."""
        return self._sample_start_times

    def __getitem__(self, idx: int) -> tuple[TensorDict, xr.DataArray]:
        """Return a sample of data spanning the timesteps
        [idx, idx + self.sample_n_times).

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of a sample's data (i.e. a mapping from names to torch.Tensors) and
            its corresponding time coordinate.
        """
        time_slice = slice(idx, idx + self.sample_n_times)
        return self.get_sample_by_time_slice(time_slice)

    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> tuple[TensorDict, xr.DataArray]:
        input_file_idx, input_local_idx = _get_file_local_index(
            time_slice.start, self.start_indices
        )
        output_file_idx, output_local_idx = _get_file_local_index(
            time_slice.stop - 1, self.start_indices
        )

        # get the sequence of observations
        arrays: dict[str, list[torch.Tensor]] = {}
        idxs = range(input_file_idx, output_file_idx + 1)
        total_steps = 0
        for i, file_idx in enumerate(idxs):
            start = input_local_idx if i == 0 else 0
            if i == len(idxs) - 1:
                stop = output_local_idx
            else:
                stop = (
                    self.start_indices[file_idx + 1] - self.start_indices[file_idx] - 1
                )

            n_steps = stop - start + 1
            shape = [n_steps] + self._shape_excluding_time_after_selection
            total_steps += n_steps
            if self.engine == "zarr":
                tensor_dict = load_series_data_zarr_async(
                    idx=start,
                    n_steps=n_steps,
                    path=self.full_paths[file_idx],
                    names=self._time_dependent_names,
                    final_dims=self.dims,
                    final_shape=shape,
                    fill_nans=self.fill_nans,
                    nontime_selection=self._isel_tuple,
                )
            else:
                ds = self._open_file(file_idx)
                ds = ds.isel(**self.isel)
                tensor_dict = load_series_data(
                    idx=start,
                    n_steps=n_steps,
                    ds=ds,
                    names=self._time_dependent_names,
                    final_dims=self.dims,
                    final_shape=shape,
                    fill_nans=self.fill_nans,
                )
                ds.close()
                del ds
            for n in self._time_dependent_names:
                arrays.setdefault(n, []).append(tensor_dict[n])

        tensors: TensorDict = {}
        for n, tensor_list in arrays.items():
            tensors[n] = torch.cat(tensor_list)
        del arrays

        # load time-invariant variables from first dataset
        if len(self._time_invariant_names) > 0:
            ds = self._open_file(idxs[0])
            ds = ds.isel(**self.isel)
            shape = [total_steps] + self._shape_excluding_time_after_selection
            for name in self._time_invariant_names:
                variable = ds[name].variable
                if self.fill_nans is not None:
                    variable = variable.fillna(self.fill_nans.value)
                tensors[name] = as_broadcasted_tensor(variable, self.dims, shape)
            ds.close()
            del ds

        # load static derived variables
        for name in self._static_derived_names:
            tensor = self._static_derived_data[name]
            horizontal_dims = [1] * tensor.ndim
            tensors[name] = tensor.repeat((total_steps, *horizontal_dims))

        # cast to desired dtype
        tensors = {k: v.to(dtype=self.dtype) for k, v in tensors.items()}

        # Apply field overwrites
        tensors = self.overwrite.apply(tensors)

        # Create a DataArray of times to return corresponding to the slice that
        # is valid even when n_repeats > 1.
        time = xr.DataArray(self.all_times[time_slice].values, dims=["time"])

        return tensors, time


def _get_timestep(time: np.ndarray) -> datetime.timedelta:
    """Computes the timestep of an array of a time coordinate array.

    Raises an error if the times are not separated by a positive constant
    interval, or if the array has one or fewer times.
    """
    assert len(time.shape) == 1, "times must be a 1D array"

    if len(time) > 1:
        timesteps = np.diff(time)
        timestep = timesteps[0]

        if not (timestep > datetime.timedelta(days=0)):
            raise ValueError("Timestep of data must be greater than zero.")

        if not np.all(timesteps == timestep):
            raise ValueError("Time coordinate does not have a uniform timestep.")

        return timestep
    else:
        raise ValueError(
            "Time coordinate does not have enough times to infer a timestep."
        )
