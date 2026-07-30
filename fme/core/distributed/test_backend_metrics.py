"""
Backend-agnostic distributed tests for spatial metrics (zonal mean, weighted
mean, etc.).

These tests work with any backend (NonDistributed, TorchDistributed,
ModelTorchDistributed) and on CPU or GPU.  They can be run serially
(``pytest``) or in parallel (``torchrun --nproc-per-node N -m pytest``).
"""

import pytest
import torch

from fme.core import get_device, metrics
from fme.core.distributed import Distributed


@pytest.mark.parallel
def test_weighted_mean():
    """Distributed weighted mean matches non-distributed reference."""
    dist = Distributed.get_instance()
    device = get_device()
    global_shape = (2, 3, 8, 16)

    gen = torch.Generator(device="cpu").manual_seed(123)
    global_data = torch.randn(*global_shape, generator=gen).to(device)
    # Latitude-like weights: vary along h (dim=-2), broadcast over w.
    global_weights = torch.cos(
        torch.linspace(-1.5, 1.5, global_shape[-2], device=device)
    ).unsqueeze(-1)  # shape (8, 1)

    slices = dist.get_local_slices(global_shape)
    local_data = global_data[slices]
    local_weights = global_weights[slices[-2]]  # local h-slice, shape (local_h, 1)

    reduce_dims = (-2, -1)
    result = dist.weighted_mean(local_data, local_weights, dim=reduce_dims)

    # Reference: global weighted mean over spatial dims.
    w = global_weights.expand(global_shape)
    expected = (global_data * w).sum(dim=reduce_dims) / w.sum(dim=reduce_dims)

    torch.testing.assert_close(result, expected)


@pytest.mark.parallel
def test_zonal_mean():
    """Zonal mean of a known tensor matches non-distributed nanmean over lon."""
    dist = Distributed.get_instance()
    device = get_device()
    global_shape = (2, 3, 8, 16)

    # Use a fixed seed so every rank builds the same global tensor.
    gen = torch.Generator(device="cpu").manual_seed(42)
    global_data = torch.randn(*global_shape, generator=gen).to(device)

    slices = dist.get_local_slices(global_shape)
    local_data = global_data[slices]

    result = dist.zonal_mean(local_data)

    # Expected: full zonal mean restricted to local h-slice.
    expected_full = global_data.nanmean(dim=-1)
    expected = expected_full[slices[:-1]]  # drop w-slice (averaged away)

    torch.testing.assert_close(result, expected)


@pytest.mark.parallel
def test_zonal_mean_with_nans():
    """Zonal mean correctly ignores NaNs across distributed lon slices."""
    dist = Distributed.get_instance()
    device = get_device()
    global_shape = (2, 8, 16)

    gen = torch.Generator(device="cpu").manual_seed(99)
    global_data = torch.randn(*global_shape, generator=gen).to(device)
    # Sprinkle NaNs at fixed positions.
    global_data[0, 3, 5] = float("nan")
    global_data[1, 0, 14] = float("nan")
    global_data[0, 7, 0] = float("nan")

    slices = dist.get_local_slices(global_shape)
    local_data = global_data[slices]

    result = dist.zonal_mean(local_data)

    expected_full = global_data.nanmean(dim=-1)
    expected = expected_full[slices[:-1]]

    torch.testing.assert_close(result, expected)


@pytest.mark.parallel
def test_zonal_mean_all_nan_row_and_uneven_and_2d():
    """Cover three edge cases in one test:
    - an all-NaN longitude row produces NaN (not crash or 0)
    - uneven lon split (15 doesn't divide evenly by 2 or 4)
    - 2D (h, w) input works
    """
    dist = Distributed.get_instance()
    device = get_device()
    global_shape = (7, 15)  # 2D, uneven lon

    gen = torch.Generator(device="cpu").manual_seed(77)
    global_data = torch.randn(*global_shape, generator=gen).to(device)
    global_data[3, :] = float("nan")  # entire row is NaN

    slices = dist.get_local_slices(global_shape)
    local_data = global_data[slices]

    result = dist.zonal_mean(local_data)

    expected_full = global_data.nanmean(dim=-1)
    expected = expected_full[slices[:-1]]

    torch.testing.assert_close(result, expected, equal_nan=True)
    # The all-NaN row (global index 3) should be NaN if it falls in our h-slice.
    h_slice = slices[0]
    local_indices = range(*h_slice.indices(global_shape[0]))
    if 3 in local_indices:
        local_row = list(local_indices).index(3)
        assert result[local_row].isnan(), "all-NaN row must produce NaN"


@pytest.mark.parallel
def test_zonal_mean_with_data_parallel_dim():
    """Zonal mean with batch split across dp-ranks."""
    dist = Distributed.get_instance()
    device = get_device()
    n_dp = dist.total_data_parallel_ranks
    batch = 4 * n_dp
    global_shape = (batch, 8, 16)

    gen = torch.Generator(device="cpu").manual_seed(55)
    global_data = torch.randn(*global_shape, generator=gen).to(device)

    slices = dist.get_local_slices(global_shape, data_parallel_dim=0)
    local_data = global_data[slices]

    result = dist.zonal_mean(local_data)

    expected_full = global_data.nanmean(dim=-1)
    expected = expected_full[slices[:-1]]

    torch.testing.assert_close(result, expected, equal_nan=True)


@pytest.mark.parallel
def test_gradient_magnitude_percent_diff():
    """Distributed grad-mag percent diff matches non-distributed reference."""
    dist = Distributed.get_instance()
    device = get_device()
    global_shape = (2, 3, 8, 16)

    gen = torch.Generator(device="cpu").manual_seed(88)
    global_truth = torch.randn(*global_shape, generator=gen).to(device)
    global_pred = torch.randn(*global_shape, generator=gen).to(device)
    global_weights = torch.rand(global_shape[-2], global_shape[-1], generator=gen).to(
        device
    )

    slices = dist.get_local_slices(global_shape)
    local_truth = global_truth[slices]
    local_pred = global_pred[slices]
    local_weights = global_weights[slices[-2], slices[-1]]

    reduce_dims = (-2, -1)
    img_shape = (global_shape[-2], global_shape[-1])
    result = dist.gradient_magnitude_percent_diff(
        local_truth, local_pred, local_weights, dim=reduce_dims, img_shape=img_shape
    )

    expected = metrics.gradient_magnitude_percent_diff(
        global_truth, global_pred, weights=global_weights, dim=reduce_dims
    )

    torch.testing.assert_close(result, expected)
