import numpy as np
import pytest
import torch
import os
from fme.core.distributed import Distributed
from fme.core.coordinates import (
    LatLonCoordinates,
)

from fme.core.mask_provider import MaskProvider
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations


tmp_path="testdata"
def int():
    # Define the sizes
    torch.manual_seed(42)
    torch.set_printoptions(precision=12, sci_mode=False)  # Adjust precision as needed
    batch_size = 1  # Example batch size
    nlat = 180  # Example size for latitude
    nlon = 360  # Example size for longitude

    # Create the latitude tensor
    lat = torch.linspace(-90, 90, nlat)

    # Create the longitude tensor
    lon = torch.linspace(0, 360, nlon)

    input_tensor = torch.rand(batch_size, nlat, nlon)

    torch.save(input_tensor, os.path.join(tmp_path, "input.pt"))
    return lat, lon, nlat, nlon, batch_size

def test_lat_lon_ops_from_coords_wo_sp():
    os.environ['H_PARALLEL_SIZE'] = '1'
    lat, lon, nlat, nlon, batch_size = int()
    input_ = torch.load(os.path.join(tmp_path, "input.pt"))
    coords = LatLonCoordinates(lat=lat, lon=lon)
    gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())
    result = gridded_ops.area_weighted_mean(input_, name="T_0")
    torch.testing.assert_close(result, torch.tensor([0.501348972321]))

def test_lat_lon_ops_from_coords_w_sp():
    lat_host, lon_host, nlat, nlon, batch_size= int()
    input_ = torch.load(os.path.join(tmp_path, "input.pt"))
    os.environ['H_PARALLEL_SIZE'] = '2'
    os.environ['W_PARALLEL_SIZE'] = '2'
    dist = Distributed.get_instance()
    device=get_device()
    lat = lat_host.to(device)
    lon = lon_host.to(device)
    coords = LatLonCoordinates(lat=lat, lon=lon)
    gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())
    inp_local_host = (input_[:,*dist.get_local_slices((nlat,nlon))]).detach().clone()
    inp_local=inp_local_host.to(device)
    result_local = gridded_ops.area_weighted_mean(inp_local, name="T_0")
    result = dist.reduce_mean(result_local)
    torch.testing.assert_close(result.to("cpu"), torch.tensor([0.501348972321]))
