import netCDF4 as nc
import numpy as np
import torch
from torch import Tensor
import os

from modulus.internal.utils.graphcast.graph_utils import deg2rad


class StaticData:
    def __init__(
        self,
        static_dataset_path: str,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
    ) -> None:
        self.lsm_path = os.path.join(static_dataset_path, "land_sea_mask.nc")
        self.geop_path = os.path.join(static_dataset_path, "geopotential.nc")
        self.lat = latitudes
        self.lon = longitudes

    def get_lsm(self) -> Tensor:
        ds = nc.Dataset(self.lsm_path)
        lsm = np.expand_dims(ds["lsm"], axis=0)
        return torch.tensor(lsm, dtype=torch.float32)

    def get_geop(self) -> Tensor:
        ds = nc.Dataset(self.geop_path)
        geop = np.expand_dims(ds["z"], axis=0)
        geop = (geop - geop.mean()) / geop.std()
        return torch.tensor(geop, dtype=torch.float32)

    def get_lat_lon(self) -> Tensor:
        # cos latitudes
        cos_lat = torch.cos(deg2rad(self.lat))
        cos_lat = cos_lat.view(1, 1, self.lat.size(0), 1)
        cos_lat_mg = cos_lat.expand(1, 1, self.lat.size(0), self.lon.size(0))

        # sin longitudes
        sin_lon = torch.sin(deg2rad(self.lon))
        sin_lon = sin_lon.view(1, 1, 1, self.lon.size(0))
        sin_lon_mg = sin_lon.expand(1, 1, self.lat.size(0), self.lon.size(0))

        # cos longitudes
        cos_lon = torch.cos(deg2rad(self.lon))
        cos_lon = cos_lon.view(1, 1, 1, self.lon.size(0))
        cos_lon_mg = cos_lon.expand(1, 1, self.lat.size(0), self.lon.size(0))

        outvar = torch.cat((cos_lat_mg, sin_lon_mg, cos_lon_mg), dim=1)
        return outvar

    def get(self) -> Tensor:
        lsm = self.get_lsm()
        geop = self.get_geop()
        lat_lon = self.get_lat_lon()
        return torch.concat((lsm, geop, lat_lon), dim=1)
