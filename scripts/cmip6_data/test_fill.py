"""Parity tests for ``fill.py`` against the torch reference at
``fme.core.fill``. The ingest container intentionally doesn't pull in
torch, but the dev environment has both — this test asserts that
the numpy port produces values within tight numerical tolerance of
the torch implementation on identical inputs.

The tolerance accounts for float32 vs float64 accumulation order in
the scipy.ndimage.convolve vs torch.F.conv2d paths; values are
expected to agree to ~1e-5 relative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _make_fixture(B: int, T: int, H: int, W: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((B, T, H, W)).astype(np.float32) * 5 + 250.0
    # Carve out a few NaN patches of varying connectedness.
    arr[:, :, 0, 0:3] = np.nan  # corner strip
    arr[:, :, H // 2, W // 2 - 2 : W // 2 + 2] = np.nan  # interior patch
    arr[:, :, H - 1, :] = np.nan  # whole bottom row
    return arr


def _torch_reference(arr: np.ndarray, num_steps: int, k: int, sigma: float):
    """Run the torch reference and return numpy output."""
    torch = pytest.importorskip("torch")
    from fme.core.fill import _fast_flood_fill, _get_interior_mask

    t = torch.from_numpy(arr.copy())
    isnan = t.isnan().any(dim=tuple(range(t.ndim - 2)))
    interior = _get_interior_mask(isnan, num_steps=num_steps)
    interior = interior.view(1, 1, arr.shape[-2], arr.shape[-1])
    out = _fast_flood_fill(
        t,
        num_steps=num_steps,
        blur_kernel_size=k,
        blur_sigma=sigma,
        interior_mask=interior,
    )
    return out.numpy()


@pytest.mark.parametrize(
    "B,T,H,W,num_steps,k,sigma",
    [
        (1, 1, 12, 16, 4, 5, 1.0),
        (1, 3, 12, 16, 4, 5, 1.0),
        (2, 4, 10, 12, 6, 7, 1.5),
        (1, 2, 45, 90, 4, 5, 1.0),
    ],
)
def test_numpy_matches_torch(
    B: int, T: int, H: int, W: int, num_steps: int, k: int, sigma: float
):
    pytest.importorskip("torch")
    from fill import fast_flood_fill

    arr = _make_fixture(B, T, H, W)
    ref = _torch_reference(arr, num_steps, k, sigma)

    # Reproduce the same static interior_mask path the torch reference
    # uses (union of NaN across all (B, T)).
    union_nan = np.isnan(arr).any(axis=tuple(range(arr.ndim - 2)))
    from fill import get_interior_mask

    interior = get_interior_mask(union_nan, num_steps=num_steps).reshape(1, 1, H, W)
    out = fast_flood_fill(
        arr,
        num_steps=num_steps,
        blur_kernel_size=k,
        blur_sigma=sigma,
        interior_mask=interior,
    )
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_numpy_handles_all_nan_plane():
    """An entirely-NaN (lat, lon) timestep should fall back to a
    sensible default rather than producing NaN output, mirroring the
    torch path (which uses nanmean and substitutes 0 when no valid
    cells exist).
    """
    from fill import fast_flood_fill

    arr = np.full((1, 1, 8, 12), np.nan, dtype=np.float32)
    out = fast_flood_fill(arr, num_steps=4, blur_kernel_size=5, blur_sigma=1.0)
    assert not np.isnan(out).any()
    # No information present → output is identically zero (the fallback).
    assert (out == 0).all()
