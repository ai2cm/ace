import numpy as np


def _total_size_mb(shape: tuple[int, ...], bytes_per_element: int) -> int:
    """Calculate total size in MB for a given shape."""
    total_elements = np.prod(shape)
    return total_elements * bytes_per_element // (1024 * 1024)


class NotReducibleError(Exception):
    """Raised when shape cannot be reduced below target size."""

    pass


def _recursive_chunksize_search(
    shape: tuple[int, ...], bytes_per_element: int, reduce_dim: int, target_mb: int = 20
) -> tuple[int, ...]:
    """
    Recursively find optimal chunk shape by halving dimensions.

    Strategy:
    - Reduces dimensions in order: time, ensemble, lat/lon (alternating)
    - Each dimension is repeatedly halved until moving to the next
    - Spatial dimensions (lat/lon) alternate for balanced subdivisions
    - Raises error if target size cannot be achieved

    Args:
        shape: Current shape to evaluate
        bytes_per_element: Size of data type in bytes
        reduce_dim: Index of dimension to try reducing (0-3)
        target_mb: Target size in MB (default: 20)

    Returns:
        Optimized chunk shape meeting target size

    Raises:
        NotReducibleError: If all dimensions exhausted but still over target
    """
    if _total_size_mb(shape, bytes_per_element) <= target_mb:
        return shape
    elif bytes_per_element / 1024**2 > target_mb:
        raise NotReducibleError(
            f"Element size {bytes_per_element} bytes exceeds target chunk size "
            f"{target_mb}MB."
        )

    # Try to halve the current dimension
    reduce_dim_size = shape[reduce_dim] // 2

    if reduce_dim_size < 1:
        # Current dimension can't be reduced further, move to next
        reduce_dim += 1
        if reduce_dim >= len(shape):
            raise NotReducibleError(
                "Cannot reduce shape further to meet target chunk size."
            )
        return _recursive_chunksize_search(
            shape, bytes_per_element, reduce_dim, target_mb
        )

    # Successfully halved the dimension, update shape
    new_shape = list(shape)
    new_shape[reduce_dim] = reduce_dim_size

    # Determine next dimension to try
    next_reduce_dim = reduce_dim
    if reduce_dim == 2:
        # Alternate between lat and lon for balanced spatial chunks
        next_reduce_dim = 3
    elif reduce_dim == 3:
        next_reduce_dim = 2
    # For dimensions 0 (time) and 1 (ensemble), keep reducing the same dimension

    return _recursive_chunksize_search(
        tuple(new_shape), bytes_per_element, next_reduce_dim, target_mb
    )


def determine_zarr_chunks(
    dims: tuple[str, ...], data_shape: tuple[int, ...], bytes_per_element: int
) -> dict[str, int]:
    """
    Auto-generate zarr chunk sizes for the output data.

    Automatically determines chunk sizes targeting <=20MB per chunk by
    recursively halving dimensions until the target size is reached.

    Args:
        dims: Dimension names (time, ensemble, latitude, longitude)
        data_shape: Shape tuple matching dims
        bytes_per_element: Size of data type (e.g., 4 for float32)
    """
    if len(data_shape) != 4:
        raise ValueError(
            "Data shape must be of length 4 (time, ensemble, latitude, longitude)."
        )

    chunk_shape = _recursive_chunksize_search(
        data_shape, bytes_per_element, reduce_dim=0, target_mb=20
    )
    return dict(zip(dims, chunk_shape))
