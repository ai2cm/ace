import torch
import torch.nn.functional as F


class SmoothFloodFill:
    """Apply fast flood filling with smooth transitions for NaN regions."""

    def __init__(
        self,
        num_steps: int,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
    ):
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
        interior_mask = self._get_interior_mask(tensor, name)
        if interior_mask is None:
            return tensor
        filled, _ = _fast_flood_fill(
            tensor,
            num_steps=self._num_steps,
            blur_kernel_size=self._blur_kernel_size,
            blur_sigma=self._blur_sigma,
            interior_mask=interior_mask,
        )
        return filled


def _get_interior_mask(nan_mask: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Finds NaN pixels that remain unfilled after `num_steps` of flood fill.

    Args:
        nan_mask: Boolean tensor of shape (..., H, W) where True = NaN.
        num_steps: Number of flood fill steps to define interior mask.
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


def _spatial_mean_fill(
    tensor: torch.Tensor,
) -> torch.Tensor:
    mean = tensor.nanmean(dim=(-2, -1), keepdim=True)
    tensor = torch.where(tensor.isnan(), mean, tensor)
    return tensor


def _fast_flood_fill(
    tensor: torch.Tensor,
    num_steps: int | None = None,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.0,
    interior_mask: torch.Tensor = None,
) -> tuple[torch.Tensor, int]:
    # --- Setup ---
    B, C, H, W = tensor.shape
    original_isnan_mask = tensor.isnan()

    x = tensor.reshape(B * C, 1, H, W)
    isnan_mask = x.isnan()
    x = torch.nan_to_num(x, nan=0.0)
    valid_mask = (~isnan_mask).float()

    # --- Pre-fill Interior (The "Mean-Fill" Step) ---
    if interior_mask is not None:
        # Compute spatial mean of valid ocean pixels per map
        mean_vals = tensor.nanmean(dim=(-2, -1), keepdim=True).view(B * C, 1, 1, 1)

        # Ensure mask shape matches x for broadcasting
        int_mask = interior_mask.expand(B, C, H, W).reshape(B * C, 1, H, W)

        # Inject mean into interior and mark as valid
        x = torch.where(int_mask, mean_vals, x)
        valid_mask = torch.where(int_mask, 1.0, valid_mask)
        isnan_mask = isnan_mask & ~int_mask

    # --- Iterative Average Pooling (The "Edge-Blend" Step) ---
    kernel = torch.ones(1, 1, 3, 3, device=tensor.device, dtype=tensor.dtype)
    steps_taken = 0

    while True:
        if num_steps is not None and steps_taken >= num_steps:
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

        steps_taken += 1

    tensor = x.view(B, C, H, W)

    # Fallback for remaining isolated NaNs
    tensor = _spatial_mean_fill(tensor)

    # --- Separable Gaussian Blur ---
    k = blur_kernel_size
    coords = torch.arange(
        -k // 2 + 1.0, k // 2 + 1.0, device=tensor.device, dtype=tensor.dtype
    )

    gauss_1d = torch.exp(-(coords**2) / (2 * blur_sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    kernel_y = gauss_1d.view(1, 1, k, 1)
    kernel_x = gauss_1d.view(1, 1, 1, k)

    x_blur = tensor.view(B * C, 1, H, W)

    blurred_y = F.conv2d(x_blur, kernel_y, padding=(k // 2, 0))
    padded_blur_y = F.pad(blurred_y, (k // 2, k // 2, 0, 0), mode="circular")
    blurred_x = F.conv2d(padded_blur_y, kernel_x, padding=0)

    blurred_tensor = blurred_x.view(B, C, H, W)
    tensor = torch.where(original_isnan_mask, blurred_tensor, tensor)

    return tensor, steps_taken
