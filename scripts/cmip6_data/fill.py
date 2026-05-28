"""Numpy/scipy port of ``fme.core.fill._fast_flood_fill``.

The CMIP6 ingest pipeline runs inside a stripped Docker image that
intentionally doesn't pull in torch (multi-GB layer, not worth it for
~150 lines of math). This module mirrors the three-phase smooth flood
fill algorithm from ``fme.core.fill`` so the on-disk values agree
exactly with what the training-time loader would produce.

Algorithm (same as the torch version):

1. **Interior pre-fill** — NaN pixels that won't be reached within
   ``num_steps`` of edge expansion are filled with the spatial mean of
   the originally-valid pixels per (batch, time).
2. **Edge-blend expansion** — a 3×3 average pool grows valid pixels
   one layer at a time, ``num_steps`` iterations. Longitude wraps
   circularly; latitude uses zero (NoOp) padding to mirror the torch
   implementation.
3. **Gaussian smoothing** — separable 1D Gaussian convolution
   (latitude with replicate pad, longitude with circular pad) followed
   by a boundary blend that uses the blurred valid mask to weight
   original-vs-blurred values.

The parity test in ``test_fill.py`` evaluates both implementations on
the same fixture and asserts byte-identical output.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve as _convolve

__all__ = [
    "fast_flood_fill",
    "get_interior_mask",
    "separable_gaussian_blur",
]


# 3x3 4-connected kernel matches torch's `torch.ones(1, 1, 3, 3)` sum-pool.
_3x3_KERNEL = np.ones((3, 3), dtype=np.float64)


def _pad_lat_lon(
    arr: np.ndarray, lat_pad: int, lon_pad: int, lat_mode: str
) -> np.ndarray:
    """Pad the last two axes of ``arr``: longitude with circular
    wraparound and latitude with the given mode (``zero`` or
    ``replicate``).
    """
    if lon_pad > 0:
        arr = np.concatenate([arr[..., -lon_pad:], arr, arr[..., :lon_pad]], axis=-1)
    if lat_pad > 0:
        if lat_mode == "zero":
            pad_shape = list(arr.shape)
            pad_shape[-2] = lat_pad
            zeros = np.zeros(pad_shape, dtype=arr.dtype)
            arr = np.concatenate([zeros, arr, zeros], axis=-2)
        elif lat_mode == "replicate":
            top = np.broadcast_to(
                arr[..., :1, :], (*arr.shape[:-2], lat_pad, arr.shape[-1])
            ).copy()
            bot = np.broadcast_to(
                arr[..., -1:, :], (*arr.shape[:-2], lat_pad, arr.shape[-1])
            ).copy()
            arr = np.concatenate([top, arr, bot], axis=-2)
        else:
            raise ValueError(f"unknown lat_mode {lat_mode!r}")
    return arr


def _conv3x3(plane: np.ndarray) -> np.ndarray:
    """3×3 sum convolution on a (..., H, W) array with circular X /
    zero-pad Y. Returns the unpadded shape.

    Matches the torch path: F.pad(circular X) then F.pad(constant 0 Y),
    then conv2d with the 3×3 ones kernel.
    """
    padded = _pad_lat_lon(plane, lat_pad=1, lon_pad=1, lat_mode="zero")
    # Broadcast the 3x3 kernel to the input ndim with leading singleton
    # axes so scipy treats only the last two dims as spatial.
    kernel_nd = _3x3_KERNEL.reshape((1,) * (padded.ndim - 2) + (3, 3))
    return _convolve(padded, kernel_nd, mode="constant", cval=0.0)[..., 1:-1, 1:-1]


def get_interior_mask(nan_mask: np.ndarray, num_steps: int) -> np.ndarray:
    """Return cells that remain NaN after ``num_steps`` of 3×3
    flood-fill expansion. Input/output share shape; the last two axes
    are treated as (lat, lon).
    """
    isnan = nan_mask.astype(bool).copy()
    valid = (~isnan).astype(np.float64)
    for _ in range(num_steps):
        neighbor_count = _conv3x3(valid)
        can_update = isnan & (neighbor_count > 0)
        valid = np.where(can_update, 1.0, valid)
        isnan = isnan & ~can_update
    return isnan


def _gauss_kernel_1d(size: int, sigma: float, dtype: np.dtype) -> np.ndarray:
    """1D Gaussian kernel matching the torch implementation. The
    reference uses ``coords = arange(-k // 2 + 1, k // 2 + 1)`` which
    in Python is ``((-k) // 2) + 1`` (floor of a negative), not
    ``-(k // 2) + 1`` — they differ by one for odd ``k``.
    """
    half_low = (-size) // 2 + 1
    half_high = size // 2 + 1
    coords = np.arange(half_low, half_high, dtype=dtype)
    g = np.exp(-(coords**2) / (2.0 * sigma * sigma))
    return (g / g.sum()).astype(dtype)


def _conv_separable_y(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply ``kernel`` along the Y (lat) axis with replicate padding."""
    k = kernel.size
    pad = k // 2
    padded = _pad_lat_lon(plane, lat_pad=pad, lon_pad=0, lat_mode="replicate")
    kernel_y = kernel.reshape((1,) * (padded.ndim - 2) + (k, 1))
    out = _convolve(padded, kernel_y, mode="constant", cval=0.0)
    return out[..., pad : padded.shape[-2] - pad, :]


def _conv_separable_x(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply ``kernel`` along the X (lon) axis with circular padding."""
    k = kernel.size
    pad = k // 2
    padded = _pad_lat_lon(plane, lat_pad=0, lon_pad=pad, lat_mode="zero")
    kernel_x = kernel.reshape((1,) * (padded.ndim - 2) + (1, k))
    out = _convolve(padded, kernel_x, mode="constant", cval=0.0)
    return out[..., :, pad : padded.shape[-1] - pad]


def separable_gaussian_blur(
    tensor: np.ndarray, blur_kernel_size: int, blur_sigma: float
) -> np.ndarray:
    """Separable Gaussian blur on a (..., H, W) array. Y is replicate-
    padded; X is circular-padded. Matches the torch implementation.
    """
    kernel = _gauss_kernel_1d(blur_kernel_size, blur_sigma, tensor.dtype)
    blurred_y = _conv_separable_y(tensor, kernel)
    return _conv_separable_x(blurred_y, kernel)


def fast_flood_fill(
    tensor: np.ndarray,
    num_steps: int = 4,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.0,
    interior_mask: np.ndarray | None = None,
    spatial_valid_mask: np.ndarray | None = None,
    blurred_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fill NaN regions in ``tensor`` of shape (B, T, H, W).

    Mirrors ``fme.core.fill._fast_flood_fill`` but in numpy. See the
    module docstring for the algorithm.

    ``interior_mask`` (optional) is the pre-computed mask of pixels to
    mean-fill before the expansion loop; pass shape (1, 1, H, W) or
    broadcastable to the input. If omitted it's computed per-(B, T)
    from the input's NaN pattern — note this is different from the
    torch implementation, which is buggy in that case; we just compute
    a per-(B, T) interior. Callers who want a stable static interior
    should pass it explicitly.
    """
    B, T, H, W = tensor.shape
    x = tensor.reshape(B * T, H, W).astype(np.float64, copy=True)

    if spatial_valid_mask is not None:
        valid = np.broadcast_to(spatial_valid_mask.astype(bool), (B * T, H, W)).copy()
    else:
        valid = ~np.isnan(x)

    isnan = ~valid
    x = np.where(isnan, 0.0, x)

    if blurred_valid_mask is None:
        original_valid = valid.astype(np.float64).reshape(B, T, H, W).copy()

    if interior_mask is None:
        interior = get_interior_mask(isnan, num_steps=num_steps)
    else:
        # Broadcast to (B*T, H, W).
        interior = (
            np.broadcast_to(
                interior_mask.reshape(-1, H, W)
                if interior_mask.ndim >= 2
                else interior_mask,
                (B * T, H, W),
            )
            .astype(bool)
            .copy()
        )
        # If the caller passes a static (1, 1, H, W) or (H, W) mask,
        # the broadcast above leaves it constant across (B, T) which
        # matches the torch path's `.expand(1, 1, H, W)` semantics.

    # Per-(B, T) spatial mean of originally-valid cells, used to pre-
    # fill the interior. nanmean of the original tensor handles the
    # case where the entire plane is NaN.
    orig_for_mean = tensor.reshape(B * T, H, W)
    with np.errstate(invalid="ignore"):
        mean_vals = np.nanmean(orig_for_mean, axis=(-2, -1), keepdims=True)
    mean_vals = np.where(np.isnan(mean_vals), 0.0, mean_vals)

    x = np.where(interior, mean_vals, x)
    valid = valid | interior
    isnan = isnan & ~interior

    for _ in range(num_steps):
        neighbor_count = _conv3x3(valid.astype(np.float64))
        neighbor_sum = _conv3x3(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            local_avg = np.where(neighbor_count > 0, neighbor_sum / neighbor_count, 0.0)
        can_update = isnan & (neighbor_count > 0)
        x = np.where(can_update, local_avg, x)
        valid = valid | can_update
        isnan = isnan & ~can_update

    out = x.reshape(B, T, H, W)
    blurred = separable_gaussian_blur(out, blur_kernel_size, blur_sigma)
    if blurred_valid_mask is None:
        blurred_valid = separable_gaussian_blur(
            original_valid, blur_kernel_size, blur_sigma
        )
    else:
        blurred_valid = np.broadcast_to(
            blurred_valid_mask.astype(np.float64), (B, T, H, W)
        ).copy()

    out = out * blurred_valid + blurred * (1.0 - blurred_valid)
    return out.astype(tensor.dtype)
