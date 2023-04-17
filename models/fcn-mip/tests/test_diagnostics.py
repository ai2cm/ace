from fcn_mip import netcdf
import numpy as np
import netCDF4 as nc
from fcn_mip import schema, weather_events
import torch
import pytest
import pathlib
import config


@pytest.mark.parametrize("cls", config.TEST_DIAGNOSTICS)
def test_diagnostic(cls: str, tmp_path: pathlib.Path):
    domain = weather_events.Window(
        name="Test",
        lat_min=-15,
        lat_max=15,
        diagnostics=[
            weather_events.Diagnostic(type=cls, function="", channels=["tcwv"])
        ],
    )
    lat = np.array([-20, 0, 20])
    lon = np.array([0, 1, 2])
    n_ensemble = 2
    path = tmp_path / "a.nc"
    weather_event = weather_events.read("EastCoast")
    with nc.Dataset(path.as_posix(), "w") as ncfile:
        total_diagnostics = netcdf.initialize_netcdf(
            ncfile,
            [domain],
            schema.Grid.grid_720x1440,
            lat,
            lon,
            n_ensemble,
            torch.device(type="cpu"),
        )[0]

        for diagnostic in total_diagnostics:
            print(ncfile)
            print(ncfile["Test"])
            nlat = len(ncfile["Test"]["lat"][:])
            nlon = len(ncfile["Test"]["lon"][:])
            data = torch.randn((n_ensemble, 1, nlat, nlon))
            time_index = 0
            batch_id = 0
            batch_size = n_ensemble
            diagnostic.update(data, time_index, batch_id, batch_size)

            # TODO Fix input data issues with crps, skill
            if not (cls in ["crps", "skill"]):
                diagnostic.finalize(
                    np.array([time_index]), weather_event, schema.ChannelSet.var34
                )

        if cls == "skill":
            assert "tcwv" in ncfile["Test"]["skill"].variables
        elif cls == "raw":
            assert "tcwv" in ncfile["Test"].variables
        else:
            assert "tcwv" in ncfile["Test"][cls].variables


@pytest.mark.parametrize("cls", ["absolute", "zonal", "meridional"])
@pytest.mark.parametrize("func", ["mean", "var"])
def test_diagnostic_functions(cls: str, func: str, tmp_path: pathlib.Path):
    domain = weather_events.Window(
        name="Test",
        lat_min=-15,
        lat_max=15,
        diagnostics=[
            weather_events.Diagnostic(type=cls, function=func, channels=["tcwv"])
        ],
    )
    lat = np.array([-20, 0, 20])
    lon = np.array([0, 1, 2])
    n_ensemble = 2
    path = tmp_path / "a.nc"
    with nc.Dataset(path.as_posix(), "w") as ncfile:
        total_diagnostics = netcdf.initialize_netcdf(
            ncfile,
            [domain],
            schema.Grid.grid_720x1440,
            lat,
            lon,
            n_ensemble,
            torch.device(type="cpu"),
        )[
            0
        ]  # Extract zeroth diagnostics since there is only 1 domain

        assert (cls + "_" + func) in ncfile["Test"].groups.keys()
        assert ("tcwv") in ncfile["Test"][cls + "_" + func].variables.keys()
        for diagnostic in total_diagnostics:
            print(diagnostic)
            nlat = len(ncfile["Test"]["lat"][:])
            nlon = len(ncfile["Test"]["lon"][:])
            data = torch.randn((n_ensemble, 1, nlat, nlon))
            time_index = 0
            batch_id = 0
            batch_size = n_ensemble
            diagnostic.update(data, time_index, batch_id, batch_size)
