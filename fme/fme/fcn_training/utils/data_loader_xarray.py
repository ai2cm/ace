import os
import torch
import logging
import xarray as xr
from glob import glob
from typing import Mapping, Tuple
from torch.utils.data import Dataset
from .data_loader_params import DataLoaderParams
from .data_requirements import DataRequirements
from .data_loader_fv3gfs import VariableMetadata
from .data_utils import load_series_data

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def get_file_local_index(
    global_idx: int,
    observation_times: xr.CFTimeIndex,
    file_start_times: xr.CFTimeIndex,
) -> Tuple[int, int]:
    """Takes a global_idx which identifies the position of an observation in the
    time-ordered union of all observations across some number of monthly files.
    Returns a tuple where the first element gives the index of the file where
    the observation is found (assuming files are time ordered), and the second
    element gives the file-local index of the observation in that file.

    """
    observation_time = observation_times[global_idx]
    file_idx = file_start_times.get_indexer([observation_time], method="ffill").item()
    file_start_time = file_start_times[file_idx]
    local_idx = global_idx - observation_times.get_indexer([file_start_time]).item()
    return file_idx, local_idx


class XarrayDataset(Dataset):
    def __init__(self, params: DataLoaderParams, requirements: DataRequirements):
        self.params = params
        self.in_names = requirements.in_names
        self.out_names = requirements.out_names
        self.names = requirements.names
        self.n_in_channels = len(self.in_names)
        self.n_out_channels = len(self.out_names)
        self.path = params.data_path
        # assume that filenames include time ordering
        self.full_paths = sorted(glob(os.path.join(self.path, "*.nc")))
        self.datasets = [None for _ in range(len(self.full_paths))]
        self.n_steps = requirements.n_timesteps  # one input, n_steps - 1 outputs
        self._get_files_stats()
        if params.n_samples is not None:
            self.n_samples_total = params.n_samples

    def _get_metadata(self, ds):
        result = {}
        for name in self.names:
            if hasattr(ds.variables[name], "units") and hasattr(
                ds.variables[name], "long_name"
            ):
                result[name] = VariableMetadata(
                    units=ds.variables[name].units,
                    long_name=ds.variables[name].long_name,
                )
        self._metadata = result

    def _get_files_stats(self):
        logging.info(f"Opening data at {os.path.join(self.path, '*.nc')}")
        file_start_times = []
        for path in self.full_paths:
            with xr.open_dataset(path, use_cftime=True) as ds:
                file_start_times.append(ds["time"][0].item())
        self.file_start_times = xr.CFTimeIndex(file_start_times)
        with xr.open_mfdataset(self.full_paths, use_cftime=True) as ds:
            self.observation_times = ds.get_index("time")
            # minus (n_steps - 1) since don't have outputs for those steps
            self.n_samples_total = len(self.observation_times) - self.n_steps + 1
            self._get_metadata(ds)
            img_shape = ds[self.names[0]].shape[1:]
            logging.info(f"Found {self.n_samples_total} samples.")
            logging.info(f"Image shape is {img_shape[0]} x {img_shape[1]}.")
            logging.info(f"Following variables are available: {list(ds.variables)}.")

    @property
    def metadata(self) -> Mapping[str, VariableMetadata]:
        return self._metadata

    @property
    def ds(self) -> xr.Dataset:
        return xr.open_mfdataset(
            self.full_paths,
            data_vars="minimal",
            coords="minimal",
            decode_times=False,
        )

    def __len__(self):
        return self.n_samples_total

    def _open_file(self, idx):
        if self.datasets[idx] is None:
            self.datasets[idx] = xr.open_dataset(
                self.full_paths[idx],
                decode_times=False,
                cache=False,
                mask_and_scale=False,
            )

    def __getitem__(self, idx):
        """Open a time-ordered subset of the files which contain the input with
        global index idx and its outputs. Get a starting index in the first file
        (input_local_idx) and a final index in the last file (output_local_idx),
        returning the time-ordered sequence of observations from input_local_idx
        to output_local_idx (inclusive).

        """
        input_file_idx, input_local_idx = get_file_local_index(
            idx, self.observation_times, self.file_start_times
        )
        output_file_idx, output_local_idx = get_file_local_index(
            idx + self.n_steps - 1, self.observation_times, self.file_start_times
        )

        # open the subset of files
        for file_idx in range(input_file_idx, output_file_idx + 1):
            self._open_file(file_idx)

        # get the sequence of observations
        arrays = {}
        datasets = self.datasets[input_file_idx : output_file_idx + 1]
        for i, ds in enumerate(datasets):
            start = input_local_idx if i == 0 else 0
            stop = output_local_idx if i == len(datasets) - 1 else len(ds["time"]) - 1
            n_steps = stop - start + 1
            tensor_dict = load_series_data(start, n_steps, ds, self.names)
            for n in self.names:
                arrays.setdefault(n, []).append(tensor_dict[n])
        for n, tensor_list in arrays.items():
            arrays[n] = torch.cat(tensor_list)
        return arrays
