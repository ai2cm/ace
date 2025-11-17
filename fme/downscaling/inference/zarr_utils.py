import numpy as np


def _total_size_mb(shape: tuple[int, ...], bytes_per_element: int) -> int:
    """Calculate total size in MB for a given shape."""
    total_elements = np.prod(shape)
    return total_elements * bytes_per_element / (1024 * 1024)


class NotReducibleError(Exception):
    """Raised when shape cannot be reduced below target size."""

    pass


def _recursive_latlon_chunksize_search(
    shape: tuple[int, ...],
    bytes_per_element: int,
    reduce_dim: int = 0,
    target_mb: int = 10,
) -> tuple[int, int]:
    """
    Recursively find optimal lat/lon chunk shape by alternating halving of dimensions.

    Args:
        shape: Current lat/lon shape to evaluate
        bytes_per_element: Size of data type in bytes
        reduce_dim: Index of dimension to start reducing (0 or 1)
        target_mb: Target size in MB (default: 10)

    Returns:
        Optimized chunk shape meeting target size

    Raises:
        NotReducibleError: If all dimensions exhausted but still over target
    """
    if len(shape) != 2:
        raise ValueError("Shape must be of length 2 (lat, lon).")

    if reduce_dim not in (0, 1):
        raise ValueError("reduce_dim must be 0 or 1.")

    if bytes_per_element < 1:
        raise ValueError("bytes_per_element must be a positive integer.")

    if target_mb < 1:
        raise ValueError("target_mb must be a positive integer.")

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
        if shape[not reduce_dim] > 1:
            # still have room to reduce the other dimension
            return _recursive_latlon_chunksize_search(
                shape,
                bytes_per_element,
                reduce_dim=int(not reduce_dim),
                target_mb=target_mb,
            )
        else:
            # Not sure if this is actually reachable given initial checks
            raise NotReducibleError(
                f"Cannot reduce dim {reduce_dim} further from size {shape[reduce_dim]}."
            )

    # Successfully halved the dimension, update shape
    new_shape = list(shape)
    new_shape[reduce_dim] = reduce_dim_size

    next_reduce_dim = int(not reduce_dim)
    if new_shape[next_reduce_dim] <= 1:
        next_reduce_dim = reduce_dim  # continue reducing the same dimension

    return _recursive_latlon_chunksize_search(
        tuple(new_shape), bytes_per_element, next_reduce_dim, target_mb
    )


def determine_zarr_chunks(
    dims: tuple[str, ...],
    data_shape: tuple[int, ...],
    bytes_per_element: int,
    target_mb: int = 10,
) -> dict[str, int]:
    """
    Determine zarr chunk sizes for a given data shape and dimension names.
    Currently, we are only optimizing lat/lon chunk sizes, reducing the
    time and ensemble dimensions to 1.  This will ensure they will fit into
    units of the "shard" which will be equivalent to a single unit
    of GPU work composed of time(s) and ensemble member(s).

    Args:
        dims: Dimension names (time, ensemble, latitude, longitude)
        data_shape: Shape of data to divide into chunks, if necessary.
        bytes_per_element: Size of data type (e.g., 4 for float32)
        target_mb: Target chunk size in MB (default: 10MB)
    """
    if len(data_shape) != 4 or len(dims) != 4:
        raise ValueError(
            "Data and dimension shape must be of length 4 "
            "(time, ensemble, latitude, longitude)."
        )

    lat_lon_chunk_shape = _recursive_latlon_chunksize_search(
        data_shape[-2:], bytes_per_element, target_mb=target_mb
    )
    return dict(zip(dims, [1, 1, *lat_lon_chunk_shape]))
