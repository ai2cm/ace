import torch


def block_mean(x: torch.Tensor, lat_weights: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Area-weighted mean of a fine-resolution field over coarse grid cells.

    Args:
        x: Fine-resolution tensor with trailing (lat, lon) dimensions, whose
            sizes are divisible by ``factor``.
        lat_weights: Area weights of the fine latitudes (e.g. cosine of
            latitude), of shape (lat,). Weights need not be normalized.
        factor: Number of fine cells along each dimension of a coarse cell.

    Returns:
        Tensor with trailing dimensions (lat // factor, lon // factor).
    """
    *leading, n_lat, n_lon = x.shape
    if n_lat % factor != 0 or n_lon % factor != 0:
        raise ValueError(
            f"Spatial shape {(n_lat, n_lon)} is not divisible by factor {factor}."
        )
    if lat_weights.shape != (n_lat,):
        raise ValueError(
            f"lat_weights must have shape ({n_lat},), got {tuple(lat_weights.shape)}."
        )
    weights = lat_weights.to(dtype=x.dtype, device=x.device)[:, None]
    blocks = (x * weights).reshape(
        *leading, n_lat // factor, factor, n_lon // factor, factor
    )
    weight_blocks = weights.expand(n_lat, factor).reshape(
        n_lat // factor, factor, 1, factor
    )
    return blocks.sum(dim=(-3, -1)) / weight_blocks.sum(dim=(-3, -1))


def project_to_block_mean(
    x: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    """
    Additively adjust a fine-resolution field so that its area-weighted block
    means equal ``target``, leaving the within-block anomalies unchanged.

    This is the orthogonal projection (with respect to the area-weighted
    inner product) of ``x`` onto the set of fine fields whose block means
    are ``target``.

    Args:
        x: Fine-resolution tensor with trailing (lat, lon) dimensions.
        target: Coarse-resolution tensor with trailing dimensions
            (lat // factor, lon // factor) and leading dimensions matching ``x``.
        lat_weights: Area weights of the fine latitudes, of shape (lat,).
        factor: Number of fine cells along each dimension of a coarse cell.

    Returns:
        Tensor with the same shape as ``x``.
    """
    correction = target.to(x.dtype) - block_mean(x, lat_weights, factor)
    correction = correction.repeat_interleave(factor, dim=-2).repeat_interleave(
        factor, dim=-1
    )
    return x + correction


def project_to_block_mean_nonnegative(
    x: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    """
    Adjust a fine-resolution field so that it is non-negative and its
    area-weighted block means equal ``target``.

    ``x`` is first clipped at zero and then rescaled within each block by
    ``target / block_mean(x)``, so the relative spatial structure of the
    clipped field is preserved. Blocks whose clipped field is identically zero
    are filled uniformly with the target value. Negative target values are
    treated as zero, since they cannot be the mean of a non-negative field.

    Args:
        x: Fine-resolution tensor with trailing (lat, lon) dimensions.
        target: Coarse-resolution tensor with trailing dimensions
            (lat // factor, lon // factor) and leading dimensions matching ``x``.
        lat_weights: Area weights of the fine latitudes, of shape (lat,).
        factor: Number of fine cells along each dimension of a coarse cell.

    Returns:
        Non-negative tensor with the same shape as ``x``.
    """
    x = x.clamp(min=0)
    target = target.to(x.dtype).clamp(min=0)
    current = block_mean(x, lat_weights, factor)
    has_mass = current > 0
    scale = torch.where(has_mass, target / torch.where(has_mass, current, 1), 0)
    fill = torch.where(has_mass, 0, target)

    def upsample(y: torch.Tensor) -> torch.Tensor:
        return y.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)

    return x * upsample(scale) + upsample(fill)
