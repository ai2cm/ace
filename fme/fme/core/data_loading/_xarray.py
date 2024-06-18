import datetime
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

import fme
from fme.core import metrics
from fme.core.typing_ import TensorDict
from fme.core.winds import lon_lat_to_xyz

from .config import Slice, TimeSlice, XarrayDataConfig
from .data_typing import (
    Dataset,
    HorizontalCoordinates,
    SigmaCoordinates,
    VariableMetadata,
)
from .requirements import DataRequirements
from .utils import (
    as_broadcasted_tensor,
    get_lons_and_lats,
    get_times,
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


def subset_dataset(dataset: Dataset, subset: slice) -> Dataset:
    """Returns a subset of the dataset and propagates other properties."""
    indices = range(len(dataset))[subset]
    logging.info(f"Subsetting dataset samples according to {subset}.")
    subsetted_dataset = torch.utils.data.Subset(dataset, indices)
    subsetted_dataset.metadata = dataset.metadata
    subsetted_dataset.area_weights = dataset.area_weights
    subsetted_dataset.sigma_coordinates = dataset.sigma_coordinates
    subsetted_dataset.horizontal_coordinates = dataset.horizontal_coordinates
    subsetted_dataset.timestep = dataset.timestep
    subsetted_dataset.is_remote = dataset.is_remote
    return subsetted_dataset


def get_sigma_coordinates(ds: xr.Dataset) -> SigmaCoordinates:
    """
    Get sigma coordinates from a dataset.

    Assumes that the dataset contains variables named `ak_N` and `bk_N` where
    `N` is the level number. The returned tensors are sorted by level number.

    Args:
        ds: Dataset to get sigma coordinates from.
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
        return SigmaCoordinates(
            ak=torch.tensor([], device=fme.get_device()),
            bk=torch.tensor([], device=fme.get_device()),
        )

    if len(ak_list) != len(bk_list):
        raise ValueError(
            "Expected same number of ak and bk coordinates, "
            f"got {len(ak_list)} and {len(bk_list)}."
        )

    return SigmaCoordinates(
        ak=torch.as_tensor(ak_list, device=fme.get_device(), dtype=torch.float),
        bk=torch.as_tensor(bk_list, device=fme.get_device(), dtype=torch.float),
    )


def get_raw_times(paths: List[str], engine: str) -> List[np.ndarray]:
    times = []
    for path in paths:
        with _open_xr_dataset(path, engine=engine) as ds:
            times.append(ds.time.values)
    return times


def repeat_and_increment_times(
    raw_times: List[np.ndarray], n_repeats: int, timestep: datetime.timedelta
) -> List[np.ndarray]:
    """Repeats and increments a collection of arrays of evenly spaced times."""
    n_timesteps = sum(len(times) for times in raw_times)
    timespan = timestep * n_timesteps

    repeated_and_incremented_times = []
    for repeats in range(n_repeats):
        increment = repeats * timespan
        for times in raw_times:
            incremented_times = times + increment
            repeated_and_incremented_times.append(incremented_times)
    return repeated_and_incremented_times


def get_cumulative_timesteps(times: List[np.ndarray]) -> np.ndarray:
    """Returns a list of cumulative timesteps for each item in times."""
    num_timesteps_per_file = [0]
    for time_coord in times:
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

    def __init__(self, lons, lats):
        """
        Args:
            lons: 1D array of longitudes.
            lats: 1D array of latitudes.
        """
        self._lats = lats
        self._lons = lons
        self._x: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None
        self._z: Optional[torch.Tensor] = None

    def _get_xyz(self) -> TensorDict:
        if self._x is None or self._y is None or self._z is None:
            lats, lons = np.broadcast_arrays(self._lats[:, None], self._lons[None, :])
            x, y, z = lon_lat_to_xyz(lons, lats)
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
        # If necessary switch this to look for environment var credentials
        # https://gcsfs.readthedocs.io/en/latest/api.html#gcsfs.core.GCSFileSystem
        logger.warning(
            "GCS currently expects user credentials authenticated using"
            " `gcloud auth application-default login`. This is not recommended for "
            "production use."
        )
        kwargs["token"] = "google_default"
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
        fn = _open_xr_dataset_lru
    else:
        # netcdf4 and h5engine have a filehandle LRU cache in xarray
        # https://github.com/pydata/xarray/blob/cd3ab8d5580eeb3639d38e1e884d2d9838ef6aa1/xarray/backends/file_manager.py#L54 # noqa: E501
        fn = _open_xr_dataset

    return fn(
        path,
        **kwargs,
    )


class XarrayDataset(Dataset):
    """Load data from a directory of files matching a pattern using xarray. The
    number of contiguous timesteps to load for each sample is specified by
    requirements.n_timesteps.

    For example, if the file(s) have the time coordinate
    (t0, t1, t2, t3, t4) and requirements.n_timesteps=3, then this dataset will
    provide three samples: (t0, t1, t2), (t1, t2, t3), and (t2, t3, t4)."""

    def __init__(
        self,
        config: XarrayDataConfig,
        requirements: DataRequirements,
    ):
        self.names = requirements.names
        self.path = config.data_path
        self.file_pattern = config.file_pattern
        self.engine = "netcdf4" if config.engine is None else config.engine
        self._default_file_pattern_check()
        fs = _get_fs(self.path)
        glob_paths = sorted(fs.glob(os.path.join(self.path, config.file_pattern)))
        self._raw_paths = _preserve_protocol(self.path, glob_paths)
        if len(self._raw_paths) == 0:
            raise ValueError(
                f"No files found matching '{self.path}/{config.file_pattern}'."
            )
        self.full_paths = self._raw_paths * config.n_repeats
        self.n_steps = requirements.n_timesteps  # one input, n_steps - 1 outputs
        self._get_files_stats(config.n_repeats, config.infer_timestep)
        first_dataset = xr.open_dataset(
            self.full_paths[0],
            decode_times=False,
            engine=self.engine,
        )
        lons, lats = get_lons_and_lats(first_dataset)
        self.lon_dim, self.lat_dim = infer_horizontal_dimension_names(first_dataset)
        self._static_derived_data = StaticDerivedData(lons, lats)
        (
            self.time_dependent_names,
            self.time_invariant_names,
            self.static_derived_names,
        ) = self._group_variable_names_by_time_type()
        self._area_weights = metrics.spherical_area_weights(lats, len(lons))
        self._sigma_coordinates = get_sigma_coordinates(first_dataset)
        self._horizontal_coordinates = HorizontalCoordinates(
            lat=torch.as_tensor(lats, device=fme.get_device()),
            lon=torch.as_tensor(lons, device=fme.get_device()),
        )

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
    def available_times(self) -> xr.CFTimeIndex:
        """Time index of all available times in the data"""
        return self._available_times

    def _default_file_pattern_check(self):
        if self.engine == "zarr" and self.file_pattern == "*.nc":
            raise ValueError(
                "The file pattern is set to the default NetCDF file pattern *.nc "
                "but the engine is specified as 'zarr'. Please set "
                "`XarrayDataConfig.file_pattern` to match the zarr filename."
            )

    def _get_metadata(self, ds):
        result = {}
        for name in self.names:
            if name in StaticDerivedData.names:
                result[name] = StaticDerivedData.metadata[name]
            elif hasattr(ds[name], "units") and hasattr(ds[name], "long_name"):
                result[name] = VariableMetadata(
                    units=ds[name].units,
                    long_name=ds[name].long_name,
                )
        self._metadata = result

    def _get_files_stats(self, n_repeats: int, infer_timestep: bool):
        logging.info(f"Opening data at {os.path.join(self.path, self.file_pattern)}")
        raw_times = get_raw_times(self._raw_paths, engine=self.engine)

        self._timestep: Optional[datetime.timedelta]
        if infer_timestep:
            self._timestep = get_timestep(np.concatenate(raw_times))
            time_coords = repeat_and_increment_times(
                raw_times, n_repeats, self.timestep
            )
        else:
            self._timestep = None
            time_coords = raw_times

        cum_num_timesteps = get_cumulative_timesteps(time_coords)
        self.start_indices = cum_num_timesteps[:-1]
        self.total_timesteps = cum_num_timesteps[-1]
        self._n_initial_conditions = self.total_timesteps - self.n_steps + 1
        self._time_index = xr.CFTimeIndex(
            np.concatenate(time_coords)[: self._n_initial_conditions]
        )
        self._available_times = xr.CFTimeIndex(np.concatenate(raw_times))

        del cum_num_timesteps, time_coords

        ds = self._open_file(0)
        self._get_metadata(ds)

        for i in range(len(self.names)):
            if self.names[i] in ds.variables:
                img_shape = ds[self.names[i]].shape[-2:]
                break
        else:
            raise ValueError(
                f"None of the requested variables {self.names} are present "
                f"in the dataset."
            )
        logging.info(f"Found {self._n_initial_conditions} samples.")
        logging.info(f"Image shape is {img_shape[0]} x {img_shape[1]}.")
        logging.info(f"Following variables are available: {list(ds.variables)}.")

    def _group_variable_names_by_time_type(self) -> VariableNames:
        """Returns lists of time-dependent variable names, time-independent
        variable names, and variables which are only present as an initial
        condition."""
        (
            time_dependent_names,
            time_invariant_names,
            static_derived_names,
        ) = ([], [], [])
        # Don't use open_mfdataset here, because it will give time-invariant
        # fields a time dimension. We assume that all fields are present in the
        # netcdf file corresponding to the first chunk of time.
        with _open_xr_dataset(self.full_paths[0], engine=self.engine) as ds:
            for name in self.names:
                if name in StaticDerivedData.names:
                    static_derived_names.append(name)
                else:
                    dims = ds[name].dims
                    if "time" in dims:
                        time_dependent_names.append(name)
                    else:
                        time_invariant_names.append(name)
        return VariableNames(
            time_dependent_names,
            time_invariant_names,
            static_derived_names,
        )

    @property
    def area_weights(self) -> torch.Tensor:
        return self._area_weights

    @property
    def metadata(self) -> Mapping[str, VariableMetadata]:
        return self._metadata

    @property
    def sigma_coordinates(self) -> SigmaCoordinates:
        return self._sigma_coordinates

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
    def time_index(self) -> xr.CFTimeIndex:
        """Return cftime index corresponding to start time of each sample."""
        return self._time_index

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
        times_segments: List[xr.DataArray] = []
        idxs = range(input_file_idx, output_file_idx + 1)
        total_steps = 0
        for i, file_idx in enumerate(idxs):
            ds = self._open_file(file_idx)
            start = input_local_idx if i == 0 else 0
            stop = output_local_idx if i == len(idxs) - 1 else len(ds["time"]) - 1
            n_steps = stop - start + 1
            total_steps += n_steps
            tensor_dict = load_series_data(
                start,
                n_steps,
                ds,
                self.time_dependent_names,
                time_dim="time",
                lon_dim=self.lon_dim,
                lat_dim=self.lat_dim,
            )
            for n in self.time_dependent_names:
                arrays.setdefault(n, []).append(tensor_dict[n])
            times_segments.append(get_times(ds, start, n_steps))
            del ds

        tensors: TensorDict = {}
        for n, tensor_list in arrays.items():
            tensors[n] = torch.cat(tensor_list)
        del arrays
        times: xr.DataArray = xr.concat(times_segments, dim="time")

        # load time-invariant variables from first dataset
        ds = self._open_file(idxs[0])
        dims = ("time", self.lat_dim, self.lon_dim)
        shape = (total_steps, ds.sizes[self.lat_dim], ds.sizes[self.lon_dim])
        for name in self.time_invariant_names:
            variable = ds[name].variable
            tensors[name] = as_broadcasted_tensor(variable, dims, shape)

        # load static derived variables
        for name in self.static_derived_names:
            tensor = self._static_derived_data[name]
            tensors[name] = tensor.repeat((total_steps, 1, 1))

        return tensors, times


def as_index_slice(subset: Union[Slice, TimeSlice], dataset: XarrayDataset) -> slice:
    """Converts a subset defined either as a Slice or TimeSlice into an index slice
    based on time coordinate in provided dataset."""
    if isinstance(subset, Slice):
        index_slice = subset.slice
    elif isinstance(subset, TimeSlice):
        index_slice = subset.slice(dataset.time_index)
    else:
        raise TypeError(f"subset must be Slice or TimeSlice, got {type(subset)}")
    return index_slice


def get_timestep(times: np.ndarray) -> datetime.timedelta:
    """Computes the timestep of an array of times.

    Raises an error if the times are not separated by a positive constant
    interval, or if the array has one or fewer times.
    """
    assert len(times.shape) == 1, "times must be a 1D array"

    if len(times) > 1:
        timesteps = np.diff(times)
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
