import logging

import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import WORLD_SIZE, requires_parallel
from fme.core.mask_provider import MaskProvider

logger = logging.getLogger(__name__)


@requires_parallel
@pytest.mark.parametrize(
    "h_parallel,w_parallel",
    [
        (2, 1),  # H-parallel split
        (1, 2),  # W-parallel split
        (3, 1),  # H-parallel split
        (2, 2),  # both
    ],
)
def test_lat_lon_ops_from_coords_w_spatial_parallelism(
    h_parallel, w_parallel, monkeypatch
):
    """Validate area-weighted mean under spatial parallelism.

    Parameterized over H/W decompositions; skips when the world size is not
    divisible by the requested decomposition. Compares the parallel result
    against a serial reference computed inside `Distributed.force_non_distributed()`.
    """

    if WORLD_SIZE % (h_parallel * w_parallel) != 0:
        pytest.skip(
            f"world_size={WORLD_SIZE} not divisible by "
            f"(h={h_parallel} * w={w_parallel})"
        )

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

    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))
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
    local_w_sum = gridded_ops.area_weighted_sum(input_local, name="T_0")
    local_weights_sum = gridded_ops.area_weighted_sum(
        torch.ones_like(input_local), name="T_0"
    )
    global_w_sum = dist.reduce_sum(local_w_sum)
    global_weights_sum = dist.reduce_sum(local_weights_sum)
    result = global_w_sum / global_weights_sum

    # Compute reference result
    with Distributed.force_non_distributed():
        coords = LatLonCoordinates(lat=lat_host, lon=lon_host)
        gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())
        result_reference = gridded_ops.area_weighted_mean(input_tensor, name="T_0")

    torch.testing.assert_close(result.to("cpu"), result_reference)
