import apache_beam as beam
import numpy as np
import pytest
import xarray as xr
import xarray_beam as xbeam

from .zarr_io import TIME_DIM, WriteShardedZarr

N_TIME, TIME_CHUNK = 4, 2


def _dataset() -> xr.Dataset:
    lat = np.degrees(np.arcsin(np.linspace(-0.9, 0.9, 3)))
    lon = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    values = np.arange(N_TIME * lat.size * lon.size, dtype=np.float32)
    return xr.Dataset(
        {"sst": ((TIME_DIM, "lat", "lon"), values.reshape(N_TIME, lat.size, -1))},
        coords={TIME_DIM: np.arange(N_TIME), "lat": lat, "lon": lon},
    )


def _write_chunks(path: str, template: xr.Dataset, worker: xr.Dataset) -> None:
    """Write ``worker`` in time chunks against ``template``, as a pipeline
    whose workers recompute ``lat``/``lon`` would."""
    chunks = [
        (
            xbeam.Key({TIME_DIM: t, "lat": 0, "lon": 0}, vars=frozenset({"sst"})),
            worker.isel({TIME_DIM: slice(t, t + TIME_CHUNK)}),
        )
        for t in range(0, N_TIME, TIME_CHUNK)
    ]
    with beam.Pipeline(runner="DirectRunner") as p:
        _ = (
            p
            | beam.Create(chunks)
            | WriteShardedZarr(
                path,
                xbeam.make_template(template),
                time_chunk_size=TIME_CHUNK,
                time_shard_size=N_TIME,
            )
        )


def test_chunk_coords_one_ulp_off_template_written_with_template_coords(tmp_path):
    """Template built on one host, chunks on another whose libm or SIMD path
    rounds ``lat`` 1 ulp differently: the write takes the template's coords."""
    template = _dataset()
    worker = template.assign_coords(
        lat=np.nextafter(template.lat.values, np.inf),
        lon=np.nextafter(template.lon.values, -np.inf),
    )
    path = str(tmp_path / "out.zarr")
    _write_chunks(path, template, worker)
    written = xr.open_zarr(path)
    for dim in ("lat", "lon", TIME_DIM):
        np.testing.assert_array_equal(written[dim].values, template[dim].values)
    np.testing.assert_array_equal(written.sst.values, template.sst.values)


def test_chunk_coords_far_from_template_raise(tmp_path):
    template = _dataset()
    worker = template.assign_coords(lat=template.lat.values + 1.0)
    with pytest.raises(Exception, match="'lat'"):
        _write_chunks(str(tmp_path / "out.zarr"), template, worker)
