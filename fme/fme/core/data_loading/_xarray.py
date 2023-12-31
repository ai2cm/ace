import logging
import os
from collections import namedtuple
from glob import glob
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import xarray as xr

import fme
from fme.core import metrics

from .params import DataLoaderParams
from .requirements import DataRequirements
from .typing import Dataset, HorizontalCoordinates, SigmaCoordinates, VariableMetadata
from .utils import apply_slice, get_lons_and_lats, get_times, load_series_data

VariableNames = namedtuple(
    "VariableNames",
    ("time_dependent_names", "time_invariant_names", "initial_condition_names"),
)


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
        ak=torch.as_tensor(ak_list, device=fme.get_device()),
        bk=torch.as_tensor(bk_list, device=fme.get_device()),
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


class XarrayDataset(Dataset):
    """Handles dataloading over multiple netcdf files using the xarray library.
    Assumes that the netcdf filenames are time-ordered."""

    def __init__(
        self,
        params: DataLoaderParams,
        requirements: DataRequirements,
        window_time_slice: Optional[slice] = None,
    ):
        self.params = params
        self.names = requirements.names
        self.path = params.data_path
        self.n_workers = params.num_data_workers
        self.engine = "netcdf4" if params.engine is None else params.engine
        # assume that filenames include time ordering
        self.full_paths = sorted(glob(os.path.join(self.path, "*.nc")))
        self.full_paths *= self.params.n_repeats
        self.n_steps = requirements.n_timesteps  # one input, n_steps - 1 outputs
        self._get_files_stats()
        (
            self.time_dependent_names,
            self.time_invariant_names,
            self.initial_condition_names,
        ) = self._group_variable_names_by_time_type()
        self.window_time_slice = window_time_slice
        first_dataset = xr.open_dataset(
            self.full_paths[0],
            decode_times=False,
        )
        lons, lats = get_lons_and_lats(first_dataset)
        self._area_weights = metrics.spherical_area_weights(lats, len(lons))
        self._sigma_coordinates = get_sigma_coordinates(first_dataset)
        self._horizontal_coordinates = HorizontalCoordinates(
            lat=torch.as_tensor(lats, device=fme.get_device()),
            lon=torch.as_tensor(lons, device=fme.get_device()),
        )

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._horizontal_coordinates

    def _get_samples_and_initial_conditions(self, ds):
        if "initial_condition" in ds.dims:
            n_candidate_ics = 1
        else:
            n_candidate_ics = self.total_timesteps - self.n_steps + 1
        self._initial_conditions = list(range(n_candidate_ics))[
            self.params.window_starts.slice
        ]
        self._n_initial_conditions = len(self._initial_conditions)
        if self.params.n_samples is not None:
            if self.params.n_samples > self._n_initial_conditions:
                raise ValueError(
                    f"Requested {self.params.n_samples} samples, but only "
                    f"{self._n_initial_conditions} initial conditions "
                    "are available."
                )
            self._n_initial_conditions = self.params.n_samples

    def _get_metadata(self, ds):
        result = {}
        for name in self.names:
            if hasattr(ds[name], "units") and hasattr(ds[name], "long_name"):
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
        del cum_num_timesteps

        ds = self._open_file(0)
        self._get_samples_and_initial_conditions(ds)
        self._get_metadata(ds)

        img_shape = ds[self.names[0]].shape[1:]
        logging.info(f"Found {self._n_initial_conditions} samples.")
        logging.info(f"Image shape is {img_shape[0]} x {img_shape[1]}.")
        logging.info(f"Following variables are available: {list(ds.variables)}.")

    def _group_variable_names_by_time_type(self) -> VariableNames:
        """Returns lists of time-dependent variable names, time-independent
        variable names, and variables which are only present as an initial
        condition."""
        time_dependent_names, time_invariant_names, initial_condition_names = [], [], []
        # Don't use open_mfdataset here, because it will give time-invariant
        # fields a time dimension. We assume that all fields are present in the
        # netcdf file corresponding to the first chunk of time.
        with xr.open_dataset(self.full_paths[0]) as ds:
            for name in self.names:
                dims = ds[name].dims
                if "time" in dims:
                    time_dependent_names.append(name)
                elif "initial_condition" in dims:
                    initial_condition_names.append(name)
                else:
                    time_invariant_names.append(name)
        return VariableNames(
            time_dependent_names, time_invariant_names, initial_condition_names
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

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], xr.DataArray]:
        """Open a time-ordered subset of the files which contain the input with
        global index idx and its outputs. Get a starting index in the first file
        (input_local_idx) and a final index in the last file (output_local_idx),
        returning the time-ordered sequence of observations from input_local_idx
        to output_local_idx (inclusive).

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of a sample's data (a mapping from names to data, for use in
                training and inference) and its corresponding time coordinates.
        """
        # transform sample index to timestep index
        idx = self._initial_conditions[idx]
        if self.window_time_slice is not None:
            time_slice = apply_slice(
                outer_slice=slice(idx, idx + self.n_steps),
                inner_slice=self.window_time_slice,
            )
        else:
            time_slice = slice(idx, idx + self.n_steps)

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

        tensors: Dict[str, torch.Tensor] = {}
        for n, tensor_list in arrays.items():
            tensors[n] = torch.cat(tensor_list)
        del arrays
        times: xr.DataArray = xr.concat(times_segments, dim="time")

        # load time-invariant variables from first dataset
        ds = self._open_file(idxs[0])
        for name in self.time_invariant_names:
            tensor = torch.as_tensor(ds[name].values)
            tensors[name] = tensor.repeat((total_steps, 1, 1))

        for name in self.initial_condition_names:
            tensors[name] = torch.as_tensor(ds[name].values)

        return tensors, times
