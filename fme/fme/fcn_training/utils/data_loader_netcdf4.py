import logging
import os
from typing import Mapping, Optional

import netCDF4
from torch.utils.data import Dataset
import torch
import xarray as xr

from fme.core import metrics

from .data_loader_params import DataLoaderParams
from .data_requirements import DataRequirements
from .data_utils import load_series_data, get_lons_and_lats
from .data_typing import SigmaCoordinates, VariableMetadata
from fme.core.device import get_device


class NetCDF4Dataset(Dataset):
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
        self.lons, self.lats = get_lons_and_lats(self.ds)
        self._log_files_stats()
        self.area_weights = metrics.spherical_area_weights(self.lats, len(self.lons))
        self.sigma_coordinates = get_sigma_coordinates(self.ds)

    def _log_files_stats(self):
        img_shape_y = len(self.lats)
        img_shape_x = len(self.lons)
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


def get_sigma_coordinates(ds) -> SigmaCoordinates:
    """
    Get sigma coordinates from a dataset.

    Assumes that the dataset contains variables named `ak_N` and `bk_N` where
    `N` is the level number. The returned tensors are sorted by level number.

    Args:
        ds: Dataset to get sigma coordinates from.
    """
    if isinstance(ds, xr.Dataset):
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
    else:  # netCDF4 dataset
        ak_mapping = {
            int(v[3:]): torch.as_tensor(ds.variables[v][:])
            for v in ds.variables
            if v.startswith("ak_")
        }
        bk_mapping = {
            int(v[3:]): torch.as_tensor(ds.variables[v][:])
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
        ak=torch.as_tensor(ak_list, device=get_device()),
        bk=torch.as_tensor(bk_list, device=get_device()),
    )
