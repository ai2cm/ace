import dataclasses
import logging
import os
from collections import namedtuple
from glob import glob
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import xarray as xr

import fme
from fme.core import metrics
from fme.core.typing_ import TensorDict
from fme.core.winds import lon_lat_to_xyz

from .config import XarrayDataConfig
from .data_typing import (
    Dataset,
    HorizontalCoordinates,
    SigmaCoordinates,
    VariableMetadata,
)
from .requirements import DataRequirements
from .utils import get_lons_and_lats, get_times, load_series_data

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
    subsetted_dataset = torch.utils.data.Subset(dataset, indices)
    subsetted_dataset.metadata = dataset.metadata
    subsetted_dataset.area_weights = dataset.area_weights
    subsetted_dataset.sigma_coordinates = dataset.sigma_coordinates
    subsetted_dataset.horizontal_coordinates = dataset.horizontal_coordinates
    return subsetted_dataset


def get_datasets_at_path(
    params: XarrayDataConfig, requirements: DataRequirements, subset: slice
) -> List[Dataset]:
    paths = sorted([str(d) for d in Path(params.data_path).iterdir() if d.is_dir()])
    if len(paths) == 0:
        raise ValueError(
            f"No directories found in {params.data_path}. "
            "Check path and whether you meant to use 'ensemble_xarray' data_type."
        )
    datasets, metadatas, sigma_coords = [], [], []
    for path in paths:
        params_curr_member = dataclasses.replace(params, data_path=path)
        dataset = XarrayDataset(params_curr_member, requirements)

        dataset = subset_dataset(dataset, subset)
        datasets.append(dataset)
        metadatas.append(dataset.metadata)
        sigma_coords.append(dataset.sigma_coordinates)

    if not _all_same(metadatas):
        raise ValueError("Metadata for each ensemble member should be the same.")

    ak, bk = list(
        zip(*[(s.ak.cpu().numpy(), s.bk.cpu().numpy()) for s in sigma_coords])
    )
    if not (_all_same(ak, cmp=np.allclose) and _all_same(bk, cmp=np.allclose)):
        raise ValueError(
            "Sigma coordinates for each ensemble member should be the same."
        )
    return datasets


def _all_same(iterable, cmp=lambda x, y: x == y):
    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration:
        return True
    return all(cmp(first, rest) for rest in it)


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
        raise ValueError("Dataset does not contain ak and bk sigma coordinates.")

    if len(ak_list) != len(bk_list):
        raise ValueError(
            "Expected same number of ak and bk coordinates, "
            f"got {len(ak_list)} and {len(bk_list)}."
        )

    return SigmaCoordinates(
        ak=torch.as_tensor(ak_list, device=fme.get_device(), dtype=torch.float),
        bk=torch.as_tensor(bk_list, device=fme.get_device(), dtype=torch.float),
    )


def get_cumulative_timesteps(paths: List[str]) -> np.ndarray:
    """Returns a list of cumulative timesteps for each file in paths."""
    num_timesteps_per_file = [0]
    for path in paths:
        with xr.open_dataset(path, use_cftime=True) as ds:
            num_timesteps_per_file.append(len(ds.time))
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


class XarrayDataset(Dataset):
    """Load data from a directory of time-ordered netCDF files using xarray. The
    number of contiguous timesteps to load for each sample is specified by
    requirements.n_timesteps.

    For example, if the concatenated netCDF files have the time coordinate
    (t0, t1, t2, t3, t4) and requirements.n_timesteps=3, then this dataset will
    provide three samples: (t0, t1, t2), (t1, t2, t3), and (t2, t3, t4)."""

    def __init__(
        self,
        config: XarrayDataConfig,
        requirements: DataRequirements,
    ):
        self.names = requirements.names
        self.path = config.data_path
        self.engine = "netcdf4" if config.engine is None else config.engine
        # assume that filenames include time ordering
        self.full_paths = sorted(glob(os.path.join(self.path, "*.nc")))
        if len(self.full_paths) == 0:
            raise ValueError(f"No netCDF files found in '{self.path}'.")
        self.full_paths *= config.n_repeats
        self.n_steps = requirements.n_timesteps  # one input, n_steps - 1 outputs
        self._get_files_stats()
        first_dataset = xr.open_dataset(
            self.full_paths[0],
            decode_times=False,
        )
        lons, lats = get_lons_and_lats(first_dataset)
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

    def _get_files_stats(self):
        logging.info(f"Opening data at {os.path.join(self.path, '*.nc')}")
        cum_num_timesteps = get_cumulative_timesteps(self.full_paths)
        self.start_indices = cum_num_timesteps[:-1]
        self.total_timesteps = cum_num_timesteps[-1]
        self._n_initial_conditions = self.total_timesteps - self.n_steps + 1
        del cum_num_timesteps

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
        with xr.open_dataset(self.full_paths[0]) as ds:
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

    def __len__(self):
        return self._n_initial_conditions

    def _open_file(self, idx):
        return xr.open_dataset(
            self.full_paths[idx],
            engine=self.engine,
            use_cftime=True,
            cache=False,
            mask_and_scale=False,
        )

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
                start, n_steps, ds, self.time_dependent_names
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
        for name in self.time_invariant_names:
            tensor = torch.as_tensor(ds[name].values)
            tensors[name] = tensor.repeat((total_steps, 1, 1))

        # load static derived variables
        for name in self.static_derived_names:
            tensor = self._static_derived_data[name]
            tensors[name] = tensor.repeat((total_steps, 1, 1))

        return tensors, times
