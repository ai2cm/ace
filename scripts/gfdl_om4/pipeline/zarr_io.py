"""Zarr store I/O and footprint checks shared by every pipeline: opening
stores by URL, refusing to overwrite an output store, asserting a variable's
footprint against the wetmask, and writing processed chunks into a sharded
zarr v3 store.
"""

import math

import apache_beam as beam
import fsspec
import numpy as np
import xarray as xr
import xarray_beam as xbeam
from obstore.store import from_url
from zarr.storage import ObjectStore

TIME_DIM = "time"
OUTPUT_DTYPE = np.float32


def make_zarr_store(url: str, read_only: bool = True):
    """Create a zarr store from a URL using obstore. If local, return the path."""
    if url.startswith("gs://"):
        return ObjectStore(from_url(url), read_only=read_only)
    else:
        return url


def assert_output_store_absent(path: str) -> None:
    """Refuse to initialize into a pre-existing output store.

    Output stores are written once and treated as immutable; initializing
    the template into an existing store would corrupt or silently overwrite
    it. Delete the store explicitly or pick a new output path.
    """
    fs, root = fsspec.url_to_fs(path)
    if fs.exists(root):
        raise FileExistsError(
            f"output store already exists at {path}; refusing to initialize "
            "into it. Delete it explicitly or choose a new output path."
        )


def assert_footprint(da: xr.DataArray, wetmask: xr.DataArray, context: str) -> None:
    """Assert a variable's valid-data footprint exactly equals the wetmask.

    Guards against a source variable whose land pattern disagrees with the
    wetmask's, which the normalized regrid would otherwise silently average
    as zeros — and guarantees the output NaN pattern equals the wetmask
    footprint at every timestep, which training assumes (a finite target at
    a masked cell NaNs the loss; a NaN target at an unmasked cell NaNs
    metrics).
    """
    valid, expected = xr.broadcast(da.notnull(), wetmask)
    mismatches = int((valid != expected).sum())
    if mismatches:
        raise AssertionError(
            f"{context}: valid-data footprint of {da.name!r} differs from the "
            f"wetmask at {mismatches} cells"
        )


def source_time_chunk_size(ds: xr.Dataset) -> int:
    """The dataset's own time chunk width, usable as the beam read width.

    Reading a narrower slice than the source is chunked re-fetches and
    re-decompresses the whole chunk once per slice, so a one-timestep read
    against a ten-timestep chunk costs ten times the bytes it uses. Matching
    the source width makes every chunk pay for itself once. Widening beyond
    it would buy nothing and cost worker memory.
    """
    widths = {
        int(da.encoding["preferred_chunks"][TIME_DIM])
        for da in ds.data_vars.values()
        if TIME_DIM in da.encoding.get("preferred_chunks", {})
    }
    if len(widths) != 1:
        raise AssertionError(
            f"expected one source time chunk width across the dataset's "
            f"variables; found {sorted(widths)}"
        )
    return widths.pop()


def shard_aligned_chunk_size(read_chunk_size: int, shard_size: int) -> int:
    """The width to split read chunks to before consolidating into shards.

    ``ConsolidateChunks`` groups a chunk by ``shard_size * (offset //
    shard_size)`` and asserts the group's first offset is the group key, so
    every chunk boundary must fall on a shard boundary. A read chunk wider
    than that alignment straddles one: a 10-timestep read against a
    365-timestep shard puts a chunk at offset 360 in the group for offset 0
    and the next, at 370, in a group keyed 365.

    The greatest common divisor is the widest split that lands every boundary
    on both a read and a shard boundary, and is the read width itself when the
    read width already divides the shard, leaving the split a no-op.
    """
    return math.gcd(read_chunk_size, shard_size)


class WriteShardedZarr(beam.PTransform):
    """Write processed (key, chunk) pairs into a templated zarr v3 store,
    chunked and sharded along time.

    Chunks are optionally split along time to ``split_chunks`` (see
    :func:`shard_aligned_chunk_size`), consolidated into shard-width chunks,
    and written with ``ChunksToZarr``.
    """

    def __init__(
        self,
        store,
        template: xr.Dataset,
        time_chunk_size: int,
        time_shard_size: int,
        split_chunks: dict[str, int] | None = None,
    ):
        super().__init__()
        self.store = store
        self.template = template
        self.zarr_chunks = {TIME_DIM: time_chunk_size}
        self.zarr_shards = {TIME_DIM: time_shard_size}
        self.split_chunks = split_chunks

    def expand(self, pcoll):
        if self.split_chunks is not None:
            pcoll = pcoll | "split" >> xbeam.SplitChunks(self.split_chunks)
        return (
            pcoll
            | "consolidate" >> xbeam.ConsolidateChunks(self.zarr_shards)
            | "to_zarr"
            >> xbeam.ChunksToZarr(
                self.store,
                self.template,
                zarr_chunks=self.zarr_chunks,
                zarr_shards=self.zarr_shards,
                zarr_format=3,
            )
        )
