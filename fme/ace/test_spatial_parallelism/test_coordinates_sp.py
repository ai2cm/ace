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
    batch_size = 4  # Example batch size
    nlat = 180  # Example size for latitude
    nlon = 360  # Example size for longitude

    # Create the latitude tensor
    lat = torch.linspace(-90, 90, nlat)

    # Create the longitude tensor
    lon = torch.linspace(0, 360, nlon)

    input_tensor = torch.rand(batch_size, nlat, nlon)
    return lat, lon, nlat, nlon, batch_size, input_tensor

def test_lat_lon_ops_from_coords_wo_sp():
    os.environ['H_PARALLEL_SIZE'] = '1'
    lat, lon, nlat, nlon, batch_size, input_ = int()
    coords = LatLonCoordinates(lat=lat, lon=lon)
    gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())
    result = gridded_ops.area_weighted_mean(input_, name="T_0")
    print(result)

def test_lat_lon_ops_from_coords_w_sp():
    lat_host, lon_host, nlat, nlon, batch_size, input_= int()
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
    print("result_local",result_local)
    print("dist._distributed", dist._distributed)
    result = dist.reduce_mean(result_local)
    print("result", result)
    torch.testing.assert_close(result.to("cpu"), torch.tensor([0.501348972321, 0.500475645065, 0.500276744366, 0.497519612312]))
