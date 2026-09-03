import numpy as np
import xarray as xr
from encoding import clear_encoding, set_shards_chunks


def _ds(n_time=4):
    return xr.Dataset(
        {
            "tvar": (("time", "latitude", "longitude"), np.random.rand(n_time, 2, 2)),
            "static2d": (("latitude", "longitude"), np.ones((2, 2))),
            "scalar": ((), np.float64(3.0)),
        },
        coords={
            "time": xr.date_range(
                "1978-10-01", periods=n_time, freq="6h", use_cftime=False
            ).values,
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
    )


def test_scalars_get_no_chunk_or_shard_encoding():
    """zarr rejects empty chunk/shard tuples, so 0-d variables must be left alone."""
    out = set_shards_chunks(clear_encoding(_ds()))
    assert out["scalar"].encoding.get("chunks") is None
    assert out["scalar"].encoding.get("shards") is None


def test_every_variable_is_writable(tmp_path):
    out = set_shards_chunks(clear_encoding(_ds()))
    for name in out.data_vars:
        out[[name]].to_zarr(tmp_path / f"{name}.zarr", mode="w")


def test_dask_chunks_match_shards():
    """xarray needs dask chunks equal to the zarr shards to write sharded stores."""
    out = set_shards_chunks(clear_encoding(_ds(n_time=40)))
    assert out["tvar"].encoding["shards"] == tuple(c[0] for c in out["tvar"].chunks)
    assert out["tvar"].encoding["chunks"] == (1, 2, 2)
