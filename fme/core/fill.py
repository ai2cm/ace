from typing import NamedTuple

import torch
import torch.nn.functional as F


class _Masks(NamedTuple):
    interior: torch.Tensor
    valid: torch.Tensor
    blurred_valid: torch.Tensor


class SmoothFloodFill:
    """Fill NaN regions using flood fill with smooth boundary transitions.

    Operates in three phases: (1) mean-fill deep interior NaN pixels that
    would not be reached within ``num_steps`` of edge expansion, (2) iterative
    neighbor-averaging that grows valid pixels inward from the boundary, and
    (3) Gaussian blur interpolation across the original valid/NaN boundary to
    avoid sharp seams.

    Interior masks are computed once per variable name and cached, so the
    NaN region for each variable must not change between calls.
    """

    def __init__(
        self,
        num_steps: int = 4,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
    ):
        """
        Args:
            num_steps: Number of iterative flood-fill expansion steps.
                Also determines which NaN pixels are classified as "interior"
                (unreachable within this many steps).
            blur_kernel_size: Size of the Gaussian blur kernel applied in the
                final smoothing pass.
            blur_sigma: Standard deviation of the Gaussian blur kernel.
        """
        self._num_steps = num_steps
        self._blur_kernel_size = blur_kernel_size
        self._blur_sigma = blur_sigma
        self._masks: dict[str, _Masks | None] = {}

    def _get_masks(self, tensor: torch.Tensor, name: str) -> _Masks | None:
        if name in self._masks:
            return self._masks[name]
        spatial_slice = tensor[(0,) * (tensor.ndim - 2)]
        # Mask construction is grad-free by virtue of going through .isnan()
        # (bool output cuts the autograd graph). The no_grad context makes
        # that invariant explicit and protects future refactors.
        with torch.no_grad():
            nan_mask = spatial_slice.isnan()
            if nan_mask.any():
                self._masks[name] = _create_masks(
                    nan_mask,
                    num_steps=self._num_steps,
                    blur_kernel_size=self._blur_kernel_size,
                    blur_sigma=self._blur_sigma,
                    dtype=tensor.dtype,
                )
            else:
                self._masks[name] = None
        return self._masks[name]

    def __call__(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Fill NaN regions in ``tensor`` and return the filled result.

        Args:
            tensor: Data tensor of shape (B, T, H, W) potentially containing
                NaN values to be filled.
            name: Variable name, used to look up (or compute and cache) the
                interior mask for this field.

        Returns:
            Tensor of the same shape with NaN regions filled.
        """
        masks = self._get_masks(tensor, name)
        if masks is None:
            return tensor

        # fast_flood_fill expects a channel dim at position -3. Wrap the
        # (B, T, H, W) input as (B, T, 1, H, W); the cached (H, W) masks
        # broadcast cleanly to (1, 1, H, W).
        wrapped = tensor.unsqueeze(-3)
        result = fast_flood_fill(
            wrapped,
            num_steps=self._num_steps,
            blur_kernel_size=self._blur_kernel_size,
            blur_sigma=self._blur_sigma,
            interior_mask=masks.interior,
            spatial_valid_mask=masks.valid,
            blurred_valid_mask=masks.blurred_valid,
        )
        return result.squeeze(-3)


class SmoothFloodFillPacked:
    """Smooth flood-fill for a packed multi-channel tensor.

    Like :class:`SmoothFloodFill`, but operates on a tensor whose trailing
    three dimensions are ``(C, H, W)`` and caches a single set of per-channel
    masks of shape ``(C, H, W)`` (rather than one set per variable name).

    On the first call, masks are computed from the ``(C, H, W)`` slice
    obtained by indexing all leading dims at zero, i.e. the NaN pattern of
    the very first leading-dim instance. The class assumes:

      * The NaN pattern of the trailing ``(C, H, W)`` is static across
        subsequent calls (e.g. the static ocean/land mask).
      * For paired calls (e.g. on prediction and target tensors), both
        share the same ``(C, H, W)`` NaN pattern.

    These assumptions are NOT validated at runtime, for efficiency. If they
    are violated (e.g. the network produces unexpected NaNs in unexpected
    places), the unexpected NaNs propagate through the loss as NaN, which
    is the desired diagnostic.
    """

    def __init__(
        self,
        num_steps: int = 4,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
    ):
        """
        Args:
            num_steps: Number of iterative flood-fill expansion steps.
            blur_kernel_size: Size of the Gaussian blur kernel applied in the
                final smoothing pass.
            blur_sigma: Standard deviation of the Gaussian blur kernel.
        """
        self._num_steps = num_steps
        self._blur_kernel_size = blur_kernel_size
        self._blur_sigma = blur_sigma
        self._masks: _Masks | None = None
        self._initialized: bool = False

    def _get_masks(self, tensor: torch.Tensor) -> _Masks | None:
        if self._initialized:
            return self._masks
        if tensor.ndim < 3:
            raise ValueError(
                "SmoothFloodFillPacked expects tensor with at least 3 "
                f"dimensions (..., C, H, W); got shape {tuple(tensor.shape)}"
            )
        # Mask construction is grad-free (see SmoothFloodFill._get_masks).
        with torch.no_grad():
            spatial_slice = tensor[(0,) * (tensor.ndim - 3)]  # (C, H, W)
            nan_mask = spatial_slice.isnan()
            if nan_mask.any():
                self._masks = _create_masks(
                    nan_mask,
                    num_steps=self._num_steps,
                    blur_kernel_size=self._blur_kernel_size,
                    blur_sigma=self._blur_sigma,
                    dtype=tensor.dtype,
                )
            else:
                self._masks = None
            self._initialized = True
        return self._masks

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fill NaN regions in ``tensor`` and return the filled result.

        Args:
            tensor: Data tensor of shape ``(*lead, C, H, W)`` potentially
                containing NaN values. Leading dims are batch-like and may
                be of arbitrary number.

        Returns:
            Tensor of the same shape with NaN regions filled.
        """
        masks = self._get_masks(tensor)
        if masks is None:
            return tensor
        return fast_flood_fill(
            tensor,
            num_steps=self._num_steps,
            blur_kernel_size=self._blur_kernel_size,
            blur_sigma=self._blur_sigma,
            interior_mask=masks.interior,
            spatial_valid_mask=masks.valid,
            blurred_valid_mask=masks.blurred_valid,
        )


def _create_masks(
    nan_mask: torch.Tensor,
    num_steps: int,
    blur_kernel_size: int,
    blur_sigma: float,
    dtype: torch.dtype,
) -> _Masks:
    """Compute static masks used by ``fast_flood_fill``.

    ``nan_mask`` may be 2-D (``(H, W)``) for single-channel fills or 3-D
    (``(C, H, W)``) for the packed multi-channel case. The returned
    ``blurred_valid`` carries an extra leading dim and is shaped
    ``(1, *nan_mask.shape)`` so it broadcasts directly against the working
    tensor in ``fast_flood_fill``.
    """
    interior_mask = _get_interior_mask(nan_mask, num_steps)
    valid_mask = ~nan_mask
    blurred_valid_mask = _separable_gaussian_blur(
        valid_mask.unsqueeze(0).to(dtype=dtype),
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
    )
    return _Masks(
        interior=interior_mask,
        valid=valid_mask,
        blurred_valid=blurred_valid_mask,
    )


def _get_interior_mask(nan_mask: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Finds NaN pixels that remain unfilled after ``num_steps`` of flood fill.

    Simulates the iterative edge-expansion process using a 3x3 convolution
    with circular padding on the longitude axis. Pixels still marked as NaN
    after all steps are considered "interior" and will be mean-filled rather
    than edge-blended.

    Args:
        nan_mask: Boolean tensor of shape (..., H, W) where True = NaN.
        num_steps: Number of flood fill expansion steps to simulate.

    Returns:
        Boolean tensor of the same shape as ``nan_mask``, where True marks
        interior NaN pixels unreachable within ``num_steps``.
    """
    isnan_mask = nan_mask.clone()
    valid_mask = (~isnan_mask).float()

    kernel = torch.ones(1, 1, 3, 3, device=nan_mask.device, dtype=torch.float32)

    orig_shape = isnan_mask.shape
    isnan_mask = isnan_mask.view(-1, 1, orig_shape[-2], orig_shape[-1])
    valid_mask = valid_mask.view(-1, 1, orig_shape[-2], orig_shape[-1])

    for _ in range(num_steps):
        pad_mask = F.pad(valid_mask, (1, 1, 0, 0), mode="circular")
        pad_mask = F.pad(pad_mask, (0, 0, 1, 1), mode="constant", value=0.0)

        neighbor_count = F.conv2d(pad_mask, kernel, padding=0)
        can_update = isnan_mask & (neighbor_count > 0)

        valid_mask = torch.where(can_update, 1.0, valid_mask)
        isnan_mask = isnan_mask & ~can_update

    return isnan_mask.view(orig_shape)


def _separable_gaussian_blur(
    tensor: torch.Tensor,
    blur_kernel_size: int,
    blur_sigma: float,
) -> torch.Tensor:
    """Applies a separable Gaussian blur with circular padding on the X-axis.

    The blur is applied as two 1-D convolutions (latitude then longitude).
    Longitude padding is circular to handle periodicity; latitude uses replicate
    padding to avoid polar artifacts.

    Args:
        tensor: Input tensor with at least two dimensions; the trailing two
            are treated as ``(H, W)`` and any leading dimensions are flattened
            into the conv batch axis.
        blur_kernel_size: Size of the 1-D Gaussian kernel.
        blur_sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred tensor of the same shape as the input.
    """
    if tensor.ndim < 2:
        raise ValueError(
            f"_separable_gaussian_blur expects at least 2 dims, got shape "
            f"{tuple(tensor.shape)}"
        )
    *lead, H, W = tensor.shape
    k = blur_kernel_size

    coords = torch.arange(
        -k // 2 + 1.0, k // 2 + 1.0, device=tensor.device, dtype=tensor.dtype
    )

    gauss_1d = torch.exp(-(coords**2) / (2 * blur_sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    kernel_y = gauss_1d.view(1, 1, k, 1)
    kernel_x = gauss_1d.view(1, 1, 1, k)

    x_blur = tensor.reshape(-1, 1, H, W)

    padded_x_blur = F.pad(x_blur, (0, 0, k // 2, k // 2), mode="replicate")
    blurred_y = F.conv2d(padded_x_blur, kernel_y, padding=0)
    padded_blur_y = F.pad(blurred_y, (k // 2, k // 2, 0, 0), mode="circular")
    blurred_x = F.conv2d(padded_blur_y, kernel_x, padding=0)

    return blurred_x.reshape(*lead, H, W)


def fast_flood_fill(
    tensor: torch.Tensor,
    num_steps: int = 4,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.0,
    interior_mask: torch.Tensor | None = None,
    spatial_valid_mask: torch.Tensor | None = None,
    blurred_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill NaN regions using iterative neighbor-averaging with smooth blending.

    The trailing three dimensions of ``tensor`` are treated as
    ``(C, H, W)``; leading dimensions are batch-like and may be of arbitrary
    number. Per-channel masks (when provided) should broadcast against
    ``(1, C, H, W)``; shapes ``(C, H, W)`` and ``(1, C, H, W)`` both work.

    Single-channel callers may wrap a ``(*lead, H, W)`` tensor as
    ``(*lead, 1, H, W)`` and pass masks of shape ``(H, W)``.

    The algorithm has three phases:

    1. **Interior mean-fill**: NaN pixels flagged by ``interior_mask`` are
       replaced with the per-(leading-instance, channel) spatial mean of
       valid pixels. The mean is computed under ``torch.no_grad`` so the
       interior fill does not route gradients globally through the channel
       mean back to all valid pixels.
    2. **Edge-blend expansion**: A 3x3 average-pooling loop grows valid
       pixels into the remaining NaN region one layer per step. Longitude
       padding is circular; latitude padding is zero. Stops after
       ``num_steps`` iterations.
    3. **Gaussian smoothing**: A separable Gaussian blur smoothly
       interpolates across the original valid/NaN boundary, preventing
       sharp seams between real and filled data.

    Args:
        tensor: Input tensor of shape ``(*lead, C, H, W)`` with NaN values
            to fill.
        num_steps: Maximum number of edge-expansion iterations.
        blur_kernel_size: Size of the Gaussian blur kernel for the final
            smoothing pass.
        blur_sigma: Standard deviation of the Gaussian blur kernel.
        interior_mask: Boolean tensor broadcastable to ``(1, C, H, W)``
            marking NaN pixels to mean-fill before the expansion loop.
            Typically produced by ``_get_interior_mask`` with the same value
            of ``num_steps``. NaNs may remain in the output if the interior
            mask was created using fewer than ``num_steps``.
        spatial_valid_mask: Pre-computed boolean mask broadcastable to
            ``(1, C, H, W)`` where True marks valid (non-NaN) pixels. When
            provided, avoids per-element NaN scanning of the input tensor.
        blurred_valid_mask: Pre-computed Gaussian-blurred valid mask,
            broadcastable to ``(1, C, H, W)``. When provided, skips
            computing it from scratch (the mask is constant for a given
            NaN pattern).

    Returns:
        Filled tensor of the same shape as the input with NaNs replaced.
    """
    if tensor.ndim < 3:
        raise ValueError(
            f"fast_flood_fill expects tensor with at least 3 dimensions "
            f"(*lead, C, H, W); got shape {tuple(tensor.shape)}"
        )
    *lead, C, H, W = tensor.shape
    N = 1
    for d in lead:
        N *= d

    x_orig = tensor.reshape(N, C, H, W)
    x = torch.nan_to_num(x_orig, nan=0.0)

    if spatial_valid_mask is not None:
        # Cached static mask -- broadcasts against the working tensor.
        # Reshape adds leading singleton dims for any (H, W) or (C, H, W)
        # input so the result is always rank 4.
        valid_mask = spatial_valid_mask.to(dtype=tensor.dtype).reshape(1, C, H, W)
        isnan_mask = valid_mask < 0.5
    else:
        isnan_mask = x_orig.isnan()
        valid_mask = (~isnan_mask).to(dtype=tensor.dtype)

    if blurred_valid_mask is None:
        original_valid_mask = valid_mask.clone()

    if interior_mask is None:
        interior_mask = _get_interior_mask(isnan_mask, num_steps=num_steps)

    while interior_mask.ndim < 4:
        interior_mask = interior_mask.unsqueeze(0)

    # Compute per-(instance, channel) spatial mean under no_grad so the
    # interior fill does not introduce a global-mean gradient pathway from
    # interior pixels back to all valid pixels of that channel.
    with torch.no_grad():
        mean_vals = x_orig.nanmean(dim=(-2, -1), keepdim=True)

    one = valid_mask.new_ones(())
    x = torch.where(interior_mask, mean_vals, x)
    valid_mask = torch.where(interior_mask, one, valid_mask)
    isnan_mask = isnan_mask & ~interior_mask

    # Iterative average pooling. Depthwise (groups=C) 3x3 conv lets each
    # channel use its own valid_mask without materializing per-channel
    # tensors as separate batches.
    kernel = torch.ones(C, 1, 3, 3, device=tensor.device, dtype=tensor.dtype)

    for _ in range(num_steps):
        pad_mask = F.pad(valid_mask, (1, 1, 0, 0), mode="circular")
        pad_mask = F.pad(pad_mask, (0, 0, 1, 1), mode="constant", value=0.0)

        pad_x = F.pad(x, (1, 1, 0, 0), mode="circular")
        pad_x = F.pad(pad_x, (0, 0, 1, 1), mode="constant", value=0.0)

        neighbor_count = F.conv2d(pad_mask, kernel, padding=0, groups=C)
        neighbor_sum = F.conv2d(pad_x, kernel, padding=0, groups=C)

        has_neighbors = neighbor_count > 0
        # Clamp the denominator before dividing so that pixels with no
        # valid neighbours don't produce 0/0 = NaN. Without this,
        # autograd propagates a NaN gradient through the division even
        # though ``torch.where`` masks the forward value below.
        safe_count = torch.where(
            has_neighbors, neighbor_count, neighbor_count.new_ones(())
        )
        can_update = isnan_mask & has_neighbors
        local_avg = torch.where(has_neighbors, neighbor_sum / safe_count, 0.0)

        x = torch.where(can_update, local_avg, x)
        valid_mask = torch.where(can_update, one, valid_mask)
        isnan_mask = isnan_mask & ~can_update

    blurred = _separable_gaussian_blur(x, blur_kernel_size, blur_sigma)
    if blurred_valid_mask is None:
        blurred_valid_mask = _separable_gaussian_blur(
            original_valid_mask, blur_kernel_size, blur_sigma
        )

    out = x * blurred_valid_mask + blurred * (1.0 - blurred_valid_mask)
    return out.reshape(*lead, C, H, W)
