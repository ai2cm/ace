import os
from fme.core import metrics
import torch
import logging
import xarray as xr
from glob import glob
from typing import Mapping, Optional, Tuple
import fme
from .data_typing import Dataset, HorizontalCoordinates, VariableMetadata
from .data_loader_params import DataLoaderParams
from .data_requirements import DataRequirements
from .data_loader_netcdf4 import get_sigma_coordinates
from .data_utils import apply_slice, load_series_data, get_lons_and_lats


def get_file_local_index(
    global_idx: int,
    observation_times: xr.CFTimeIndex,
    file_start_times: xr.CFTimeIndex,
) -> Tuple[int, int]:
    """Takes a global_idx which identifies the position of an observation in the
    time-ordered union of all observations across some number of files. Returns
    a tuple where the first element gives the index of the file where the
    observation is found (assuming files are time ordered), and the second
    element gives the file-local index of the observation in that file.

    """
    observation_time = observation_times[global_idx]
    file_idx = file_start_times.get_indexer([observation_time], method="ffill").item()
    file_start_time = file_start_times[file_idx]
    local_idx = global_idx - observation_times.get_indexer([file_start_time]).item()
    return file_idx, local_idx


class XarrayDataset(Dataset):
    def __init__(
        self,
        params: DataLoaderParams,
        requirements: DataRequirements,
        window_time_slice: Optional[slice] = None,
    ):
        self.params = params
        self.in_names = requirements.in_names
        self.out_names = requirements.out_names
        self.names = requirements.names
        self.n_in_channels = len(self.in_names)
        self.n_out_channels = len(self.out_names)
        self.path = params.data_path
        self.n_workers = params.num_data_workers
        self.engine = "netcdf4" if params.engine is None else params.engine
        # assume that filenames include time ordering
        self.full_paths = sorted(glob(os.path.join(self.path, "*.nc")))
        self.n_steps = requirements.n_timesteps  # one input, n_steps - 1 outputs
        self._get_files_stats()
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
            if self.params.n_samples is not None:
                if self.params.n_samples > self.n_samples_total:
                    raise ValueError(
                        f"Requested {self.params.n_samples} samples, but only "
                        f"{self.n_samples_total} are available."
                    )
                self.n_samples_total = self.params.n_samples
            self._get_metadata(ds)
            img_shape = ds[self.names[0]].shape[1:]
            logging.info(f"Found {self.n_samples_total} samples.")
            logging.info(f"Image shape is {img_shape[0]} x {img_shape[1]}.")
            logging.info(f"Following variables are available: {list(ds.variables)}.")

    @property
    def area_weights(self) -> torch.Tensor:
        return self._area_weights

    @property
    def metadata(self) -> Mapping[str, VariableMetadata]:
        return self._metadata

    @property
    def sigma_coordinates(self):
        return self._sigma_coordinates

    def __len__(self):
        return self.n_samples_total

    def _open_file(self, idx):
        return xr.open_dataset(
            self.full_paths[idx],
            engine=self.engine,
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
        if self.window_time_slice is not None:
            time_slice = apply_slice(
                outer_slice=slice(idx, idx + self.n_steps),
                inner_slice=self.window_time_slice,
            )
        else:
            time_slice = slice(idx, idx + self.n_steps)

        input_file_idx, input_local_idx = get_file_local_index(
            time_slice.start, self.observation_times, self.file_start_times
        )
        output_file_idx, output_local_idx = get_file_local_index(
            time_slice.stop - 1,
            self.observation_times,
            self.file_start_times,
        )

        # get the sequence of observations
        arrays = {}
        idxs = range(input_file_idx, output_file_idx + 1)
        for i, file_idx in enumerate(idxs):
            ds = self._open_file(file_idx)
            start = input_local_idx if i == 0 else 0
            stop = output_local_idx if i == len(idxs) - 1 else len(ds["time"]) - 1
            n_steps = stop - start + 1
            tensor_dict = load_series_data(start, n_steps, ds, self.names)
            for n in self.names:
                arrays.setdefault(n, []).append(tensor_dict[n])
            del ds
        for n, tensor_list in arrays.items():
            arrays[n] = torch.cat(tensor_list)
        return arrays
