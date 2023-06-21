import logging
import os
from typing import List
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


class FV3GFSDataset(Dataset):
    def __init__(
        self, params: DataLoaderParams, path: str, requirements: DataRequirements
    ):
        self.params = params
        self._check_for_not_implemented_features()
        self.in_names = requirements.in_names
        self.out_names = requirements.out_names
        self.names = requirements.names
        self.n_in_channels = len(self.in_names)
        self.n_out_channels = len(self.out_names)
        self.path = path
        self.full_path = os.path.join(path, "*.nc")
        self.dt = params.dt
        self.n_steps = 2  # one input, one output timestep
        self._get_files_stats()

    def _check_for_not_implemented_features(self):
        if self.params.dt != 1:
            raise NotImplementedError("step size must be 1 for FV3GFSDataset")

    def _get_files_stats(self):
        logging.info(f"Opening data at {self.full_path}")
        self.ds = netCDF4.MFDataset(self.full_path)
        self.ds.set_auto_mask(False)
        # minus one since don't have an output for the last step
        self.n_samples_total = len(self.ds.variables["time"][:]) - self.n_steps + 1
        # provided ERA5 dataloader gets the "wrong" x/y convention (x is lat, y is lon)
        # so we follow that convention here for consistency
        if "grid_xt" in self.ds.variables:
            self.img_shape_x = len(self.ds.variables["grid_yt"][:])
            self.img_shape_y = len(self.ds.variables["grid_xt"][:])
        else:
            self.img_shape_x = len(self.ds.variables["lat"][:])
            self.img_shape_y = len(self.ds.variables["lon"][:])
        logging.info(f"Found {self.n_samples_total} samples.")
        logging.info(f"Image shape is {self.img_shape_x} x {self.img_shape_y}.")
        logging.info(f"Following variables are available: {list(self.ds.variables)}.")

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, idx):
        return load_series_data(
            idx=idx, n_steps=self.n_steps, ds=self.ds, names=self.names
        )
