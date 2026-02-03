import os

import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.mask_provider import MaskProvider


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


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires multi-GPU machine")
def test_lat_lon_ops_from_coords_w_sp():
    lat_host, lon_host, nlat, nlon, batch_size, input_ = int()
    # Compute reference result
    with Distributed.force_non_distributed():
        coords = LatLonCoordinates(lat=lat_host, lon=lon_host)
        gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())
        result_ref = gridded_ops.area_weighted_mean(input_, name="T_0")

    os.environ["H_PARALLEL_SIZE"] = "2"
    os.environ["W_PARALLEL_SIZE"] = "2"
    dist = Distributed.get_instance()
    device = get_device()
    lat = lat_host.to(device)
    lon = lon_host.to(device)
    coords = LatLonCoordinates(lat=lat, lon=lon)
    gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())
    inp_local_host = (input_[:, *dist.get_local_slices((nlat, nlon))]).detach().clone()
    inp_local = inp_local_host.to(device)
    result_local = gridded_ops.area_weighted_mean(inp_local, name="T_0")
    result = dist.reduce_mean(result_local)
    torch.testing.assert_close(result.to("cpu"), result_ref)
