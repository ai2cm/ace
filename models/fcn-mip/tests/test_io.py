from fcn_mip import netcdf
import numpy as np
import netCDF4 as nc
from fcn_mip.schema import Grid
from fcn_mip.weather_events import Window, Diagnostic
import torch


def test_initialize_netcdf(tmp_path):
    domain = Window(
        name="TestAverage",
        lat_min=-15,
        lat_max=15,
        diagnostics=[Diagnostic(type="zonal", function="mean", channels=["tcwv"])],
    )
    lat = np.array([-20, 0, 20])
    lon = np.array([0, 1, 2])
    n_ensemble = 1
    path = tmp_path / "a.nc"
    with nc.Dataset(path.as_posix(), "w") as ncfile:
        netcdf.initialize_netcdf(
            ncfile,
            [domain],
            Grid("720x1440"),
            lat,
            lon,
            n_ensemble,
            torch.device(type="cpu"),
        )
