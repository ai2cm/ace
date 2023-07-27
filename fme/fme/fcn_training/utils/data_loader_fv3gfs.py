import logging
import os
from collections import namedtuple
from typing import Mapping, Optional, Tuple

import netCDF4
import numpy as np
from torch.utils.data import Dataset

from fme.core import metrics

from .data_loader_params import DataLoaderParams
from .data_requirements import DataRequirements
from .data_utils import load_series_data

VariableMetadata = namedtuple("VariableMetadata", ["units", "long_name"])


class FV3GFSDataset(Dataset):
    def __init__(
        self,
        params: DataLoaderParams,
        requirements: DataRequirements,
        window_time_slice: Optional[slice] = None,
    ):
        """
        Args:
            params: Parameters for the data loader.
            requirements: Data requirements for the model.
            window_time_slice: Time slice within each window to use for the data loader,
                if given the loader will only return data from this time slice.
                By default it will return the full windows.
        """
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
        self.metadata = self._compute_variable_metadata()  # Note: requires `self.ds`
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
        self.window_time_slice = window_time_slice
        self.lats, self.lons = np.array(self.ds["grid_yt"]), np.array(
            self.ds["grid_xt"]
        )
        self.area_weights = metrics.spherical_area_weights(self.lats, len(self.lons))

        try:
            self.lats, self.lons = np.array(self.ds.variables["grid_yt"]), np.array(
                self.ds.variables["grid_xt"]
            )
        except AttributeError:
            raise ValueError(
                (
                    "Dataset does not contain grid_yt and"
                    "grid_xt variables which define the spatial grid."
                )
            )

        self.area_weights = metrics.spherical_area_weights(self.lats, len(self.lons))
        ak, bk = self._get_sigma_coordinates()
        self.sigma_coordinates = dict(ak=ak, bk=bk)

    def _log_files_stats(self):
        if "grid_xt" in self.ds.variables:
            img_shape_y = len(self.ds.variables["grid_yt"][:])
            img_shape_x = len(self.ds.variables["grid_xt"][:])
        else:
            img_shape_y = len(self.ds.variables["lat"][:])
            img_shape_x = len(self.ds.variables["lon"][:])
        logging.info(f"Image shape is {img_shape_x} x {img_shape_y}.")
        logging.info(f"Following variables are available: {list(self.ds.variables)}.")

    def _compute_variable_metadata(self) -> Mapping[str, VariableMetadata]:
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

    def _get_sigma_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        ak = sorted([v for v in self.ds.variables if v.startswith("ak")])
        bk = sorted([v for v in self.ds.variables if v.startswith("bk")])

        if len(ak) == 0 or len(bk) == 0:
            raise ValueError("Dataset does not contain ak and bk sigma coordinates.")

        if len(ak) != len(bk):
            raise ValueError(
                "Dataset contains different number of ak and bk coordinates."
            )

        if len(ak) > 10:
            raise NotImplementedError(
                "Sigma coordinate names must be parsed to support more than 10 levels."
            )

        ak = np.array([self.ds[k][:] for k in ak])
        bk = np.array([self.ds[k][:] for k in bk])
        return ak, bk

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, idx):
        return load_series_data(
            idx=idx,
            n_steps=self.n_steps,
            ds=self.ds,
            names=self.names,
            window_time_slice=self.window_time_slice,
        )
