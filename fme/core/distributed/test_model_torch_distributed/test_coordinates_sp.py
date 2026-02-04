import os

import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.mask_provider import MaskProvider


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multi-GPU machine")
def test_lat_lon_ops_from_coords_w_sp():
    # Define the sizes
    torch.manual_seed(42)
    batch_size = 4  # Example batch size
    nlat = 180  # Example size for latitude
    nlon = 360  # Example size for longitude

    # Create the latitude tensor
    lat_host = torch.linspace(-90, 90, nlat)

    # Create the longitude tensor
    lon_host = torch.linspace(0, 360, nlon)

    input_tensor = torch.rand(batch_size, nlat, nlon)

    # Compute reference result
    with Distributed.force_non_distributed():
        coords = LatLonCoordinates(lat=lat_host, lon=lon_host)
        gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())
        result_reference = gridded_ops.area_weighted_mean(input_tensor, name="T_0")

    os.environ["H_PARALLEL_SIZE"] = "2"
    os.environ["W_PARALLEL_SIZE"] = "1"
    dist = Distributed.get_instance()
    device = get_device()
    lat = lat_host.to(device)
    lon = lon_host.to(device)
    coords = LatLonCoordinates(lat=lat, lon=lon)
    gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())
    input_local_host = (
        (input_tensor[:, *dist.get_local_slices((nlat, nlon))]).detach().clone()
    )
    input_local = input_local_host.to(device)
    result_local = gridded_ops.area_weighted_mean(input_local, name="T_0")
    result = dist.reduce_mean(result_local)
    torch.testing.assert_close(result.to("cpu"), result_reference)
    # Set H_PARALLEL_SIZE back to 1.
    os.environ["H_PARALLEL_SIZE"] = "1"
