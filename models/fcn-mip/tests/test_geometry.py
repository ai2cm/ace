from fcn_mip import geometry
from fcn_mip.weather_events import Window
import torch
import numpy as np

import pytest


@pytest.mark.xfail
def test_select_space():
    domain = Window(name="latitude", lat_min=-15, lat_max=15, diagnostics=[])
    lat = np.array([-20, 0, 20])
    lon = np.array([0, 1, 2])
    data = torch.ones((1, 1, len(lat), len(lon))).float()
    lat, lon, output = geometry.select_space(data, lat, lon, domain)
    assert tuple(output.shape[2:]) == (len(lat), len(lon))
    assert np.all(np.abs(lat) <= 15)


@pytest.mark.xfail
def test_get_bounds_window():
    domain = Window(name="latitude", lat_min=-15, lat_max=15, diagnostics=[])
    lat = np.array([-20, 0, 20])
    lon = np.array([0, 1, 2])
    lat_sl, _ = geometry.get_bounds_window(domain, lat, lon)
    assert lat[lat_sl].shape == (1,)
    assert lat[lat_sl][0] == 0
