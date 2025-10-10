import xarray as xr

OUTPUT_CHUNKING = {
    "time": 1,
    "latitude": -1,
    "longitude": -1,
}
OUTPUT_SHARDING = {
    "time": 20,
    "latitude": -1,
    "longitude": -1,
}


def set_shards_chunks(ds, shards=OUTPUT_SHARDING, chunks=OUTPUT_CHUNKING):
    """
    Set the chunking and sharding for the output dataset.

    Note that the outer "zarr shards" (set via encoding) need to be
    the same as "dask chunks" for xarray dask arrays to write to sharded
    and chunked zarr v3 stores. The inner "zarr chunks" are also set via
    encoding. This is easy to get wrong.
    See https://github.com/pydata/xarray/discussions/9938#discussioncomment-11821657
    """
    out_ds = xr.Dataset()
    for name, da in ds.data_vars.items():
        da_chunks = []
        da_shards = []
        chunking_dict = {}
        for dim in da.dims:
            if dim in shards:
                shard_size = shards[dim] if shards[dim] > 0 else da.sizes[dim]
            else:
                shard_size = da.sizes[dim]
            da_shards.append(shard_size)
            chunking_dict[dim] = shard_size
            if dim in chunks:
                chunk_size = chunks[dim] if chunks[dim] > 0 else da.sizes[dim]
            else:
                chunk_size = da.sizes[dim]
            da_chunks.append(chunk_size)
        da = da.chunk(chunking_dict)
        da.encoding["chunks"] = tuple(da_chunks)
        da.encoding["shards"] = tuple(da_shards)
        out_ds[name] = da
    return out_ds


def clear_encoding(ds):
    for variable in {**ds.coords, **ds.data_vars}.values():
        variable.encoding = {}
    return ds
