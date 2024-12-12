import datetime
import json
import logging
import os
import warnings
from collections import namedtuple
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

import fsspec
import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import (
    HEALPixCoordinates,
    HorizontalCoordinates,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.device import get_device
from fme.core.typing_ import Slice, TensorDict

from .config import RepeatedInterval, TimeSlice, XarrayDataConfig
from .data_typing import Dataset, VariableMetadata
from .requirements import DataRequirements
from .utils import (
    as_broadcasted_tensor,
    get_horizontal_dimensions,
    infer_horizontal_dimension_names,
    load_series_data,
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


def _get_vertical_coordinates(
    ds: xr.Dataset, dtype: Optional[torch.dtype]
) -> HybridSigmaPressureCoordinate:
    """
    Get hybrid sigma-pressure coordinates from a dataset.

    Assumes that the dataset contains variables named `ak_N` and `bk_N` where
    `N` is the level number. The returned tensors are sorted by level number.

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

    if len(ak_list) == 0 or len(bk_list) == 0:
        logger.warning("Dataset does not contain ak and bk coordinates.")
        return HybridSigmaPressureCoordinate(
            ak=torch.tensor([]),
            bk=torch.tensor([]),
        )

    return HybridSigmaPressureCoordinate(
        ak=torch.as_tensor(ak_list, dtype=dtype),
        bk=torch.as_tensor(bk_list, dtype=dtype),
    )


def get_raw_times(paths: List[str], engine: str) -> List[np.ndarray]:
    times = []
    for path in paths:
        with _open_xr_dataset(path, engine=engine) as ds:
            times.append(ds.time.values)
    return times


def repeat_and_increment_time(
    raw_times: List[np.ndarray], n_repeats: int, timestep: datetime.timedelta
) -> List[np.ndarray]:
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


def get_cumulative_timesteps(time: List[np.ndarray]) -> np.ndarray:
    """Returns a list of cumulative timesteps for each item in a time coordinate."""
    num_timesteps_per_file = [0]
    for time_coord in time:
        num_timesteps_per_file.append(len(time_coord))
    return np.array(num_timesteps_per_file).cumsum()


def get_file_local_index(index: int, start_indices: np.ndarray) -> Tuple[int, int]:
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
        self._x: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None
        self._z: Optional[torch.Tensor] = None

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
        use_cftime=True,
        mask_and_scale=False,
        cache=False,
        chunks={},
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


class DatasetProperties:
    def __init__(
        self,
        variable_metadata: Mapping[str, VariableMetadata],
        vertical_coordinate: HybridSigmaPressureCoordinate,
        horizontal_coordinates: HorizontalCoordinates,
        timestep: datetime.timedelta,
        is_remote: bool,
    ):
        self.variable_metadata = variable_metadata
        self.vertical_coordinate = vertical_coordinate
        self.horizontal_coordinates = horizontal_coordinates
        self.timestep = timestep
        self.is_remote = is_remote

    def to_device(self) -> "DatasetProperties":
        device = get_device()
        return DatasetProperties(
            self.variable_metadata,
            self.vertical_coordinate.to(device),
            self.horizontal_coordinates.to(device),
            self.timestep,
            self.is_remote,
        )

    def update(self, other: "DatasetProperties"):
        self.is_remote = self.is_remote or other.is_remote
        if self.timestep != other.timestep:
            raise ValueError("Inconsistent timesteps between datasets")
        if self.variable_metadata != other.variable_metadata:
            raise ValueError("Inconsistent metadata between datasets")
        if self.vertical_coordinate != other.vertical_coordinate:
            raise ValueError("Inconsistent vertical coordinates between datasets")
        if self.horizontal_coordinates != other.horizontal_coordinates:
            raise ValueError("Inconsistent horizontal coordinates between datasets")


def get_xarray_dataset(
    config: XarrayDataConfig, requirements: DataRequirements
) -> Tuple["Dataset", DatasetProperties]:
    dataset = XarrayDataset(config, requirements)
    properties = dataset.properties
    index_slice = as_index_selection(config.subset, dataset)
    dataset = dataset.subset(index_slice)
    return dataset, properties


class XarrayDataset(Dataset):
    """Load data from a directory of files matching a pattern using xarray. The
    number of contiguous timesteps to load for each sample is specified by
    requirements.n_timesteps.

    For example, if the file(s) have the time coordinate
    (t0, t1, t2, t3, t4) and requirements.n_timesteps=3, then this dataset will
    provide three samples: (t0, t1, t2), (t1, t2, t3), and (t2, t3, t4).
    """

    def __init__(
        self,
        config: XarrayDataConfig,
        requirements: DataRequirements,
    ):
        self._horizontal_coordinates: HorizontalCoordinates
        self.renamed_variables = config.renamed_variables or {}
        self._names = self._get_names_to_load(requirements.names)
        self.path = config.data_path
        self.file_pattern = config.file_pattern
        self.engine = config.engine
        self.dtype = config.torch_dtype
        self.spatial_dimensions = config.spatial_dimensions
        self.fill_nans = config.fill_nans
        fs = _get_fs(self.path)
        glob_paths = sorted(fs.glob(os.path.join(self.path, config.file_pattern)))
        self._raw_paths = _preserve_protocol(self.path, glob_paths)
        if len(self._raw_paths) == 0:
            raise ValueError(
                f"No files found matching '{self.path}/{self.file_pattern}'."
            )
        self.full_paths = self._raw_paths * config.n_repeats
        self.n_steps = requirements.n_timesteps  # one input, n_steps - 1 outputs
        self._get_files_stats(config.n_repeats, config.infer_timestep)
        first_dataset = xr.open_dataset(
            self.full_paths[0],
            decode_times=False,
            engine=self.engine,
        )
        (
            self._horizontal_coordinates,
            self._static_derived_data,
        ) = self.configure_horizontal_coordinates(first_dataset)
        (
            self._time_dependent_names,
            self._time_invariant_names,
            self._static_derived_names,
        ) = self._group_variable_names_by_time_type()
        self._vertical_coordinates = _get_vertical_coordinates(
            first_dataset, self.dtype
        )
        self.overwrite = config.overwrite

    @property
    def properties(self) -> DatasetProperties:
        return DatasetProperties(
            self._variable_metadata,
            self._vertical_coordinates,
            self._horizontal_coordinates,
            self.timestep,
            self.is_remote,
        )

    def _get_names_to_load(self, names: List[str]) -> List[str]:
        # requirements.names from stepper config refer to the final set of
        # variables after any renaming occurs. This returns the set of names
        # to load from data before renaming.
        inverted_renaming = {v: k for k, v in self.renamed_variables.items()}
        return [inverted_renaming.get(n, n) for n in names]

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._horizontal_coordinates

    @property
    def is_remote(self) -> bool:
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
        raw_times = get_raw_times(self._raw_paths, engine=self.engine)

        self._timestep: Optional[datetime.timedelta]
        if infer_timestep:
            self._timestep = get_timestep(np.concatenate(raw_times))
            time_coord = repeat_and_increment_time(raw_times, n_repeats, self.timestep)
        else:
            self._timestep = None
            time_coord = raw_times

        cum_num_timesteps = get_cumulative_timesteps(time_coord)
        self.start_indices = cum_num_timesteps[:-1]
        self.total_timesteps = cum_num_timesteps[-1]
        self._n_initial_conditions = self.total_timesteps - self.n_steps + 1
        self._sample_start_time = xr.CFTimeIndex(
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
    ) -> Tuple[HorizontalCoordinates, StaticDerivedData]:
        horizontal_coordinates: HorizontalCoordinates
        static_derived_data: StaticDerivedData
        dims = get_horizontal_dimensions(first_dataset, self.dtype)

        if self.spatial_dimensions == "latlon":
            lons = dims[0]
            lats = dims[1]
            names = infer_horizontal_dimension_names(first_dataset)
            lon_name = names[0]
            lat_name = names[1]
            horizontal_coordinates = LatLonCoordinates(
                lon=lons,
                lat=lats,
                loaded_lat_name=lat_name,
                loaded_lon_name=lon_name,
            )
            static_derived_data = StaticDerivedData(horizontal_coordinates)
        elif self.spatial_dimensions == "healpix":
            face = dims[0]
            height = dims[1]
            width = dims[2]
            horizontal_coordinates = HEALPixCoordinates(
                face=face,
                height=height,
                width=width,
            )
            static_derived_data = StaticDerivedData(horizontal_coordinates)
        else:
            raise ValueError(
                f"unexpected config.spatial_dimensions {self.spatial_dimensions},"
                " should be one of 'latlon' or 'healpix'"
            )
        coords_sizes = {
            coord_name: len(coord)
            for coord_name, coord in horizontal_coordinates.coords.items()
        }
        logging.info(f"Horizontal coordinate sizes are {coords_sizes}.")
        return horizontal_coordinates, static_derived_data

    @property
    def variable_metadata(self) -> Mapping[str, VariableMetadata]:
        return self._variable_metadata

    @property
    def vertical_coordinate(self) -> HybridSigmaPressureCoordinate:
        return self._vertical_coordinates

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
    def sample_start_time(self) -> xr.CFTimeIndex:
        """Return cftime index corresponding to start time of each sample."""
        return self._sample_start_time

    def __getitem__(self, idx: int) -> Tuple[TensorDict, xr.DataArray]:
        """Return a sample of data spanning the timesteps [idx, idx + self.n_steps).

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of a sample's data (i.e. a mapping from names to torch.Tensors) and
            its corresponding time coordinate.
        """
        time_slice = slice(idx, idx + self.n_steps)
        return self.get_sample_by_time_slice(time_slice)

    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> Tuple[TensorDict, xr.DataArray]:
        input_file_idx, input_local_idx = get_file_local_index(
            time_slice.start, self.start_indices
        )
        output_file_idx, output_local_idx = get_file_local_index(
            time_slice.stop - 1, self.start_indices
        )

        # get the sequence of observations
        arrays: Dict[str, List[torch.Tensor]] = {}
        idxs = range(input_file_idx, output_file_idx + 1)
        total_steps = 0
        for i, file_idx in enumerate(idxs):
            ds = self._open_file(file_idx)
            if self.fill_nans is not None:
                ds = ds.fillna(self.fill_nans.value)
            start = input_local_idx if i == 0 else 0
            stop = output_local_idx if i == len(idxs) - 1 else len(ds["time"]) - 1
            n_steps = stop - start + 1
            total_steps += n_steps
            tensor_dict = load_series_data(
                idx=start,
                n_steps=n_steps,
                ds=ds,
                names=self._time_dependent_names,
                time_dim="time",
                spatial_dim_names=self._horizontal_coordinates.loaded_dims,
            )
            for n in self._time_dependent_names:
                arrays.setdefault(n, []).append(tensor_dict[n])
            ds.close()
            del ds

        tensors: TensorDict = {}
        for n, tensor_list in arrays.items():
            tensors[n] = torch.cat(tensor_list)
        del arrays

        # load time-invariant variables from first dataset
        if len(self._time_invariant_names) > 0:
            ds = self._open_file(idxs[0])
            dims = ["time"] + self._horizontal_coordinates.loaded_dims
            shape = [total_steps] + [ds.sizes[dim] for dim in dims[1:]]
            for name in self._time_invariant_names:
                variable = ds[name].variable
                tensors[name] = as_broadcasted_tensor(variable, dims, shape)
            ds.close()
            del ds

        # load static derived variables
        for name in self._static_derived_names:
            tensor = self._static_derived_data[name]
            tensors[name] = tensor.repeat((total_steps, 1, 1))

        # cast to desired dtype
        tensors = {k: v.to(dtype=self.dtype) for k, v in tensors.items()}

        # apply renaming
        for original_name, new_name in self.renamed_variables.items():
            tensors[new_name] = tensors.pop(original_name)

        # Apply field overwrites
        tensors = self.overwrite.apply(tensors)

        # Create a DataArray of times to return corresponding to the slice that
        # is valid even when n_repeats > 1.
        time = xr.DataArray(self.all_times[time_slice].values, dims=["time"])

        return tensors, time

    def subset(self, subset: Union[slice, torch.Tensor]) -> Dataset:
        """Returns a subset of the dataset and propagates other properties."""
        indices = range(len(self))[subset]
        logging.info(f"Subsetting dataset samples according to {subset}.")
        subsetted_dataset = torch.utils.data.Subset(self, indices)
        return subsetted_dataset


def as_index_selection(
    subset: Union[Slice, TimeSlice, RepeatedInterval], dataset: XarrayDataset
) -> Union[slice, np.ndarray]:
    """Converts a subset defined either as a Slice or TimeSlice into an index slice
    based on time coordinate in provided dataset.
    """
    if isinstance(subset, Slice):
        index_selection = subset.slice
    elif isinstance(subset, TimeSlice):
        index_selection = subset.slice(dataset.sample_start_time)
    elif isinstance(subset, RepeatedInterval):
        try:
            index_selection = subset.get_boolean_mask(len(dataset), dataset.timestep)
        except ValueError as e:
            raise ValueError(f"Error when applying RepeatedInterval to dataset: {e}")
    else:
        raise TypeError(f"subset must be Slice or TimeSlice, got {type(subset)}")
    return index_selection


def get_timestep(time: np.ndarray) -> datetime.timedelta:
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
