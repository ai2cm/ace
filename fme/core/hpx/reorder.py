from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).parent / "data"


def get_reordering_ring_to_xy(nside: int, device: torch.device) -> torch.Tensor:
    """
    Compute index mapping from Healpix ring order to flat-XY order.

    Args:
        nside: The nside of the Healpix grid.
        device: The device to run the computation on.
    """
    reorder = np.load(DATA_DIR / f"reorder_xyf2pix_{nside:05d}.npy")
    return torch.from_numpy(reorder).to(device)


def get_reordering_xy_to_ring(nside: int, device: torch.device) -> torch.Tensor:
    """
    Compute index mapping from flat-XY order (with given origin/clockwise input
    orientation) to Healpix ring order.

    Returns a LongTensor 'reorder' so that:
        x_ring = x_xy[reorder]

    Args:
        nside: The nside of the Healpix grid.
        device: The device to run the computation on.
    """
    ring_to_xy = get_reordering_ring_to_xy(nside, device)
    xy_to_ring = torch.argsort(ring_to_xy)
    return xy_to_ring
