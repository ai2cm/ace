import torch
import torch.nn.functional as F


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
        self._interior_masks: dict[str, torch.Tensor | None] = {}

    def _get_interior_mask(
        self, tensor: torch.Tensor, name: str
    ) -> torch.Tensor | None:
        if name in self._interior_masks:
            return self._interior_masks[name]
        spatial_slice = tensor[(0,) * (tensor.ndim - 2)]
        nan_mask = spatial_slice.isnan()
        if nan_mask.any():
            self._interior_masks[name] = _get_interior_mask(nan_mask, self._num_steps)
        else:
            self._interior_masks[name] = None
        return self._interior_masks[name]

    def __call__(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Fill NaN regions in ``tensor`` and return the filled result.

        Args:
            tensor: Data tensor of shape (B, C, H, W) potentially containing
                NaN values to be filled.
            name: Variable name, used to look up (or compute and cache) the
                interior mask for this field.

        Returns:
            Tensor of the same shape with NaN regions filled.
        """
        interior_mask = self._get_interior_mask(tensor, name)
        if interior_mask is None:
            return tensor
        filled = _fast_flood_fill(
            tensor,
            num_steps=self._num_steps,
            blur_kernel_size=self._blur_kernel_size,
            blur_sigma=self._blur_sigma,
            interior_mask=interior_mask,
        )
        return filled


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

    # 3x3 kernel
    kernel = torch.ones(1, 1, 3, 3, device=nan_mask.device, dtype=torch.float32)

    # Ensure 4D shape for conv2d
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
    tensor: torch.Tensor, blur_kernel_size: int = 5, blur_sigma: float = 1.0
) -> torch.Tensor:
    """Applies a separable Gaussian blur with circular padding on the X-axis.

    The blur is applied as two 1-D convolutions (latitude then longitude).
    Longitude padding is circular to handle periodicity; latitude uses replicate
    padding to avoid polar artifacts.

    Args:
        tensor: Input tensor of shape (B, C, H, W).
        blur_kernel_size: Size of the 1-D Gaussian kernel.
        blur_sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred tensor of the same shape as the input.
    """
    B, C, H, W = tensor.shape
    k = blur_kernel_size

    coords = torch.arange(
        -k // 2 + 1.0, k // 2 + 1.0, device=tensor.device, dtype=tensor.dtype
    )

    gauss_1d = torch.exp(-(coords**2) / (2 * blur_sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    kernel_y = gauss_1d.view(1, 1, k, 1)
    kernel_x = gauss_1d.view(1, 1, 1, k)

    x_blur = tensor.reshape(B * C, 1, H, W)

    padded_x_blur = F.pad(x_blur, (0, 0, k // 2, k // 2), mode="replicate")
    blurred_y = F.conv2d(padded_x_blur, kernel_y, padding=0)
    padded_blur_y = F.pad(blurred_y, (k // 2, k // 2, 0, 0), mode="circular")
    blurred_x = F.conv2d(padded_blur_y, kernel_x, padding=0)

    return blurred_x.view(B, C, H, W)


def _spatial_mean_fill(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Replace remaining NaN pixels with the per-(batch, channel) spatial mean.

    Used as a fallback after iterative flood fill to handle isolated NaN
    pixels that have no valid neighbors.

    Args:
        tensor: Tensor of shape (B, C, H, W) that may still contain NaNs.

    Returns:
        Tensor of the same shape with NaNs replaced by spatial means.
    """
    mean = tensor.nanmean(dim=(-2, -1), keepdim=True)
    tensor = torch.where(tensor.isnan(), mean, tensor)
    return tensor


def _fast_flood_fill(
    tensor: torch.Tensor,
    num_steps: int | None = None,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.0,
    interior_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill NaN regions using iterative neighbor-averaging with smooth blending.

    The algorithm has three phases:

    1. **Interior mean-fill**: NaN pixels flagged by ``interior_mask`` are
       replaced with the per-(batch, channel) spatial mean of valid pixels.
       This avoids wasting expansion steps on deep-interior pixels.
    2. **Edge-blend expansion**: A 3x3 average-pooling loop grows valid pixels
       into the remaining NaN region one layer per step. Longitude padding is
       circular; latitude padding is zero. Stops after ``num_steps`` iterations
       (or when no NaN pixels remain if ``num_steps`` is None).
    3. **Gaussian smoothing**: A separable Gaussian blur is used to smoothly
       interpolate across the original valid/NaN boundary, preventing sharp
       seams between real and filled data.

    Any NaN pixels still remaining after steps 1--2 (e.g. fully isolated
    pixels with no valid neighbors) are filled with the spatial mean as a
    fallback before the smoothing pass.

    Args:
        tensor: Input tensor of shape (B, C, H, W) with NaN values to fill.
        num_steps: Maximum number of edge-expansion iterations. If None, the
            loop runs until all NaN pixels are filled or no further progress
            can be made.
        blur_kernel_size: Size of the Gaussian blur kernel for the final
            smoothing pass.
        blur_sigma: Standard deviation of the Gaussian blur kernel.
        interior_mask: Boolean tensor broadcastable to (B, C, H, W) marking
            NaN pixels to mean-fill before the expansion loop. Typically
            produced by ``_get_interior_mask``.

    Returns:
        Filled tensor of the same shape as the input with all NaNs replaced.
    """
    B, C, H, W = tensor.shape
    original_valid_mask = (~tensor.isnan()).to(dtype=tensor.dtype)

    x = tensor.reshape(B * C, 1, H, W)
    isnan_mask = x.isnan()
    x = torch.nan_to_num(x, nan=0.0)
    valid_mask = (~isnan_mask).to(dtype=tensor.dtype)

    # Pre-fill interior with the nanmean
    if interior_mask is not None:
        # Compute spatial mean of valid ocean pixels per map
        mean_vals = tensor.nanmean(dim=(-2, -1), keepdim=True).view(B * C, 1, 1, 1)

        # Ensure mask shape matches x for broadcasting
        int_mask = interior_mask.expand(B, C, H, W).reshape(B * C, 1, H, W)

        # Inject mean into interior and mark as valid
        x = torch.where(int_mask, mean_vals, x)
        valid_mask = torch.where(int_mask, 1.0, valid_mask)
        isnan_mask = isnan_mask & ~int_mask

    # Iterative average pooling
    kernel = torch.ones(1, 1, 3, 3, device=tensor.device, dtype=tensor.dtype)
    step = 0

    while True:
        if num_steps is not None and step >= num_steps:
            break
        elif not isnan_mask.any():
            break

        pad_mask = F.pad(valid_mask, (1, 1, 0, 0), mode="circular")
        pad_mask = F.pad(pad_mask, (0, 0, 1, 1), mode="constant", value=0.0)

        pad_x = F.pad(x, (1, 1, 0, 0), mode="circular")
        pad_x = F.pad(pad_x, (0, 0, 1, 1), mode="constant", value=0.0)

        neighbor_count = F.conv2d(pad_mask, kernel, padding=0)
        neighbor_sum = F.conv2d(pad_x, kernel, padding=0)

        can_update = isnan_mask & (neighbor_count > 0)

        if num_steps is None and not can_update.any():
            break

        local_avg = torch.where(neighbor_count > 0, neighbor_sum / neighbor_count, 0.0)

        x = torch.where(can_update, local_avg, x)
        valid_mask = torch.where(can_update, 1.0, valid_mask)
        isnan_mask = isnan_mask & ~can_update

        step += 1

    tensor = x.view(B, C, H, W)

    # Fallback for remaining isolated NaNs
    tensor = _spatial_mean_fill(tensor)

    # Apply Gaussian blur
    blurred_tensor = _separable_gaussian_blur(tensor, blur_kernel_size, blur_sigma)
    blurred_valid_mask = _separable_gaussian_blur(
        original_valid_mask, blur_kernel_size, blur_sigma
    )

    # smoothly interpolate across the boundary
    tensor = (tensor * blurred_valid_mask) + (
        blurred_tensor * (1.0 - blurred_valid_mask)
    )

    return tensor
