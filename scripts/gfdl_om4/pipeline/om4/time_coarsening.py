"""Time coarsening of a stream by block means (``time_block_mean_factor``).

A coarsened stream is selected at source time resolution in whole averaging
blocks (:func:`select_whole_blocks`), and each beam chunk is averaged to
output resolution (:func:`coarsen`). Kept free of beam so it can be tested
without the pipeline's runtime dependencies.
"""

import xarray as xr

from ..dims import TIME_DIM
from ..postprocess import append_derivation
from .config import StreamConfig


def time_block_mean_factor(stream: StreamConfig) -> int:
    """Source timesteps per output timestep of a stream (1 unless coarsened)."""
    return stream.time_block_mean_factor or 1


def select_whole_blocks(
    ds: xr.Dataset,
    factor: int,
    start_time: str | None,
    end_time: str | None,
    context: str,
) -> xr.Dataset:
    """Select the ``factor``-step blocks of ``ds``, anchored to its first
    timestep, whose final timestamps (their output labels) fall within the
    inclusive time range.

    Raises if ``factor`` does not evenly divide the full time axis, rather
    than silently dropping a trailing partial block.
    """
    size = ds.sizes[TIME_DIM]
    if size % factor:
        raise AssertionError(
            f"{context}: time_block_mean_factor {factor} does not evenly divide "
            f"the {size} timesteps of the source store"
        )
    n_blocks = size // factor
    if start_time is None and end_time is None:
        return ds
    labels = ds.indexes[TIME_DIM][factor - 1 :: factor]
    selected = labels.slice_indexer(start_time, end_time)
    first, stop = selected.start or 0, selected.stop
    stop = n_blocks if stop is None else stop
    return ds.isel({TIME_DIM: slice(first * factor, max(stop, first) * factor)})


def stream_output_time(stream: StreamConfig, ds: xr.Dataset) -> xr.DataArray:
    """The output time coordinate of a stream opened by
    :func:`pipeline.om4.run.open_stream`: its own time coordinate, or for a
    coarsened stream the final timestamp of each block."""
    factor = time_block_mean_factor(stream)
    return ds[TIME_DIM].isel({TIME_DIM: slice(factor - 1, None, factor)})


def coarsen(ds: xr.Dataset, time_offset: int, factor: int) -> tuple[int, xr.Dataset]:
    """Average a source-resolution chunk of a coarsened stream over its
    ``factor``-step blocks.

    The chunk must hold whole blocks (the read width is a multiple of
    ``factor``, and :func:`pipeline.om4.run.open_stream` selects the stream
    in whole blocks).
    Returns the chunk's time offset re-expressed in output timesteps, and the
    block means labeled with each block's final timestamp. NaN anywhere in a
    block propagates to its mean, so a time-varying footprint still fails the
    wetmask assertion downstream.
    """
    size = ds.sizes[TIME_DIM]
    if time_offset % factor or size % factor:
        raise AssertionError(
            f"chunk at time offset {time_offset} with {size} timesteps does not "
            f"hold whole {factor}-step blocks"
        )
    start = time_offset // factor
    labels = ds[TIME_DIM].values[factor - 1 :: factor]
    coarsened = (
        ds.drop_vars(TIME_DIM)
        .coarsen({TIME_DIM: factor})
        .mean(skipna=False, keep_attrs=True)
        .assign_coords({TIME_DIM: labels})
    )
    note = (
        f"mean over {factor}-step blocks of the source time axis, labeled "
        "with each block's final time"
    )
    for name in coarsened.data_vars:
        append_derivation(coarsened[name], note)
    return start, coarsened
