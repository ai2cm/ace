import logging
import os
from typing import MutableMapping
import numpy as np
from torch.utils.data import Dataset
import netCDF4
from utils.constants import CHANNEL_NAMES
from utils.img_utils import reshape_fields

# conversion from 'standard' names defined in utils/constants.py to those
# in FV3GFS output netCDFs
FV3GFS_NAMES = {
    "u10": "UGRD10m",
    "v10": "VGRD10m",
    "t2m": "TMP2m",
    "sp": "PRESsfc",
    "msl": "PRMSL",
    "t850": "TMP850",
    "u1000": "UGRD1000",
    "v1000": "VGRD1000",
    "z1000": "h1000",
    "u850": "UGRD850",
    "v850": "VGRD850",
    "z850": "h850",
    "u500": "UGRD500",
    "v500": "VGRD500",
    "z500": "h500",
    "t500": "TMP500",
    "z50": "h50",
    "rh500": "RH500",
    "rh850": "RH850",
    "tcwv": "TCWV",
}


class FV3GFSDataset(Dataset):
    def __init__(self, params: MutableMapping, path: str, train: bool):
        self.params = params
        self._check_for_not_implemented_features()
        self.path = path
        self.full_path = os.path.join(path, "*.nc")
        self.train = train
        self.dt = params.dt
        self.n_history = params.n_history
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.in_names = [FV3GFS_NAMES[CHANNEL_NAMES[c]] for c in self.in_channels]
        self.out_names = [FV3GFS_NAMES[CHANNEL_NAMES[c]] for c in self.out_channels]
        self.crop_size_x = params.crop_size_x
        self.crop_size_y = params.crop_size_y
        self.roll = params.roll
        self.two_step_training = params.two_step_training
        self.orography = params.orography
        self.precip = True if "precip" in params else False
        self.add_noise = params.add_noise if train else False
        self.normalize = params.normalize if "normalize" in params else True
        self._get_files_stats()
        self._load_stats_data()

    def _check_for_not_implemented_features(self):
        if self.params.dt != 1:
            raise NotImplementedError("step size must be 1 for FV3GFSDataset")
        if self.params.n_history != 0:
            msg = "non-zero n_history is not implemented for FV3GFSDataset"
            raise NotImplementedError(msg)
        if self.params.crop_size_x is not None or self.params.crop_size_y is not None:
            msg = "non-null crop_size_x or crop_size_y is not implemented for FV3GFSDataset"
            raise NotImplementedError(msg)
        if self.params.roll:
            raise NotImplementedError("roll=True not implemented for FV3GFSDataset")
        if self.params.two_step_training:
            msg = "two_step_training not implemented for FV3GFSDataset"
            raise NotImplementedError(msg)
        if self.params.orography:
            msg = "training w/ orography not implemented for FV3GFSDataset"
            raise NotImplementedError(msg)
        if "precip" in self.params:
            msg = "precip training not implemented for FV3GFSDataset"
            raise NotImplementedError(msg)
        if self.params.add_grid:
            raise NotImplementedError("add_grid not implemented for FV3GFSDataset")

    def _get_files_stats(self):
        logging.info(f"Opening data at {self.full_path}")
        self.ds = netCDF4.MFDataset(self.full_path)
        self.ds.set_auto_mask(False)
        # minus one since don't have an output for the last step
        self.n_samples_total = len(self.ds.variables["time"][:]) - 1
        # provided ERA5 dataloader gets the "wrong" x/y convention (x is lat, y is lon)
        # so we follow that convention here for consistency
        self.img_shape_x = len(self.ds.variables["grid_yt"][:])
        self.img_shape_y = len(self.ds.variables["grid_xt"][:])
        logging.info(f"Found {self.n_samples_total} samples.")
        logging.info(f"Image shape is {self.img_shape_x} x {self.img_shape_y}.")
        logging.info(f"Following variables are available: {list(self.ds.variables)}.")

    def _load_stats_data(self):
        logging.info(f"Opening global mean stats data at {self.params.global_means_path}")
        in_, out_ = load_arrays_from_netcdf(self.params.global_means_path, self.in_names, self.out_names)
        self.in_means = in_.reshape((1, self.n_in_channels, 1, 1))
        self.out_means = out_.reshape((1, self.n_out_channels, 1, 1))

        logging.info(f"Opening stddev stats data at {self.params.global_stds_path}")
        in_, out_ = load_arrays_from_netcdf(self.params.global_stds_path, self.in_names, self.out_names)
        self.in_stds = in_.reshape((1, self.n_in_channels, 1, 1))
        self.out_stds = out_.reshape((1, self.n_out_channels, 1, 1))

        # just used for multistep validation
        logging.info(f"Opening time mean stats data at {self.params.time_means_path}")
        in_, out_ = load_arrays_from_netcdf(self.params.time_means_path, self.in_names, self.out_names)
        self.out_time_means = np.expand_dims(out_, 0)

    @property
    def data_array(self):
        fv3gfs_names = [FV3GFS_NAMES[v] for v in CHANNEL_NAMES]
        arrays = [np.expand_dims(self.ds.variables[v][:], 1) for v in fv3gfs_names]
        return np.concatenate(arrays, axis=1)

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, idx):
        in_arrays = [self.ds.variables[c][idx : idx + 1, :, :] for c in self.in_names]
        out_arrays = [
            self.ds.variables[c][idx + 1 : idx + 2, :, :] for c in self.out_names
        ]
        in_array = np.concatenate(in_arrays, axis=0)
        out_array = np.concatenate(out_arrays, axis=0)
        in_tensor = reshape_fields(
            in_array,
            "inp",
            self.crop_size_x,
            self.crop_size_y,
            0,
            0,
            self.params,
            self.roll,
            self.train,
            self.in_means,
            self.in_stds,
            self.normalize,
            self.orography,
            self.add_noise,
        )
        out_tensor = reshape_fields(
            out_array,
            "tar",
            self.crop_size_x,
            self.crop_size_y,
            0,
            0,
            self.params,
            self.roll,
            self.train,
            self.out_means,
            self.out_stds,
            self.normalize,
            self.orography,
        )
        return in_tensor, out_tensor

def load_arrays_from_netcdf(path, in_names, out_names):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    in_array = np.array([ds.variables[c][:] for c in in_names])
    out_array = np.array([ds.variables[c][:] for c in out_names])
    ds.close()
    return in_array, out_array
