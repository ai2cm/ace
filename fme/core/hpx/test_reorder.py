import pytest
import torch

from fme.core import get_device
from fme.core.hpx.reorder import get_reordering_ring_to_xy, get_reordering_xy_to_ring


def test_reorder_xy_to_ring_just_faces():
    device = get_device()
    reorder = get_reordering_xy_to_ring(nside=1, device=device)
    assert reorder.device == device
    expected_reorder = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        device=device,
    )
    assert torch.allclose(reorder, expected_reorder)


@pytest.mark.parametrize("nside", [1, 2, 4, 128])
def test_reorder_xy_to_ring(nside: int):
    device = get_device()
    reorder = get_reordering_xy_to_ring(nside=nside, device=device)
    assert reorder.shape == (12 * nside * nside,)
    unique_indices = set(reorder.tolist())
    assert len(unique_indices) == 12 * nside * nside


def test_reorder_ring_to_xy_2_by_2():
    device = get_device()
    reorder = get_reordering_ring_to_xy(nside=2, device=device)
    assert reorder.device == device
    reorder = reorder.reshape(12, 2, 2).cpu()
    # taken from figure at https://nvlabs.github.io/earth2grid/tutorials/healpix.html  # noqa: E501
    assert torch.allclose(reorder[0], torch.tensor([[0, 5], [4, 13]]))
    assert torch.allclose(reorder[1], torch.tensor([[1, 7], [6, 15]]))
    assert torch.allclose(reorder[6], torch.tensor([[16, 24], [23, 32]]))
    assert torch.allclose(reorder[7], torch.tensor([[18, 26], [25, 34]]))
    assert torch.allclose(reorder[11], torch.tensor([[35, 43], [42, 47]]))


@pytest.mark.parametrize("nside", [1, 2, 4, 128])
def test_reorder_round_trip(nside: int):
    torch.manual_seed(0)
    device = get_device()
    xy_to_ring = get_reordering_xy_to_ring(nside=nside, device=device)
    ring_to_xy = get_reordering_ring_to_xy(nside=nside, device=device)
    assert torch.allclose(
        xy_to_ring[ring_to_xy], torch.arange(12 * nside * nside).to(device)
    )
    initial_xy = torch.randn(12 * nside * nside).to(device)
    ring = initial_xy[xy_to_ring]
    final_xy = ring[ring_to_xy]
    assert torch.allclose(initial_xy, final_xy)
