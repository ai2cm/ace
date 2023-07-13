from collections import namedtuple
import logging
import os
from typing import List, Mapping
from torch.utils.data import Dataset
import torch
import netCDF4
from .data_loader_params import DataLoaderParams
from .data_requirements import DataRequirements
import numpy as np


def load_series_data(idx: int, n_steps: int, ds: netCDF4.MFDataset, names: List[str]):
    # flip the lat dimension so that it is increasing
    arrays = {
        n: torch.as_tensor(
            np.flip(ds.variables[n][idx : idx + n_steps, :, :], axis=-2).copy()
        )
        for n in names
    }
    return arrays


VariableMetadata = namedtuple("VariableMetadata", ["units", "long_name"])


class FV3GFSDataset(Dataset):
    def __init__(self, params: DataLoaderParams, requirements: DataRequirements):
        self.params = params
        self.in_names = requirements.in_names
        self.out_names = requirements.out_names
        self.names = requirements.names
        self.n_in_channels = len(self.in_names)
        self.n_out_channels = len(self.out_names)
        self.path = params.data_path
        self.full_path = os.path.join(self.path, "*.nc")
        self.n_steps = requirements.n_timesteps  # one input, one output timestep
        logging.info(f"Opening data at {self.full_path}")
        self.ds = netCDF4.MFDataset(self.full_path)
        self.ds.set_auto_mask(False)
        self.metadata = self._compute_metadata()  # Note: requires `self.ds`
        self._log_files_stats()
        self.n_samples_total = len(self.ds.variables["time"][:]) - self.n_steps + 1
        if params.n_samples is not None:
            if params.n_samples > self.n_samples_total:
                raise ValueError(
                    f"Requested {params.n_samples} samples, but only "
                    f"{self.n_samples_total} are available."
                )
            self.n_samples_total = params.n_samples
        logging.info(f"Using {self.n_samples_total} samples.")

    def _log_files_stats(self):
        if "grid_xt" in self.ds.variables:
            img_shape_y = len(self.ds.variables["grid_yt"][:])
            img_shape_x = len(self.ds.variables["grid_xt"][:])
        else:
            img_shape_y = len(self.ds.variables["lat"][:])
            img_shape_x = len(self.ds.variables["lon"][:])
        logging.info(f"Image shape is {img_shape_x} x {img_shape_y}.")
        logging.info(f"Following variables are available: {list(self.ds.variables)}.")

    def _compute_metadata(self) -> Mapping[str, VariableMetadata]:
        result = {}
        for name in self.names:
            if hasattr(self.ds.variables[name], "units") and hasattr(
                self.ds.variables[name], "long_name"
            ):
                result[name] = VariableMetadata(
                    units=self.ds.variables[name].units,
                    long_name=self.ds.variables[name].long_name,
                )
        return result

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, idx):
        return load_series_data(
            idx=idx, n_steps=self.n_steps, ds=self.ds, names=self.names
        )
