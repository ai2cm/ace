"""Tests of the time coarsening (``time_block_mean_factor``) in run.py."""

import numpy as np
import pytest
import xarray as xr

from ..config_io import OutputConfig, WetmaskConfig
from ..postprocess import DERIVATION_ATTR
from ..zarr_io import TIME_DIM
from .config import PipelineConfig, StaticsConfig, StreamConfig
from .run import _coarsen, open_stream, stream_output_time

VARIABLE = "tos"


def _daily_source(n_days: int) -> xr.Dataset:
    """A daily 2D source dataset on a noleap calendar starting 0151-01-01,
    whose values are distinct at every point and timestep."""
    times = xr.date_range(
        "0151-01-01", periods=n_days, calendar="noleap", use_cftime=True
    )
    data = np.arange(n_days * 2 * 3, dtype=np.float32).reshape(n_days, 2, 3)
    return xr.Dataset(
        {VARIABLE: ((TIME_DIM, "yh", "xh"), data)},
        coords={TIME_DIM: times},
    )


def _open_coarsened_stream(
    tmp_path,
    source: xr.Dataset,
    factor: int,
    start_time: str | None = None,
    end_time: str | None = None,
) -> tuple[StreamConfig, xr.Dataset]:
    """Write ``source`` to a local zarr store and open it as a stream
    coarsened by ``factor``."""
    store = str(tmp_path / "source.zarr")
    source.to_zarr(store, zarr_format=3)
    stream = StreamConfig(
        name="surface",
        store=store,
        variables=[VARIABLE],
        time_block_mean_factor=factor,
    )
    config = PipelineConfig(
        streams=[stream],
        statics=StaticsConfig(store=store, variables=[]),
        wetmask=WetmaskConfig(store=store, variable=VARIABLE),
        target_grid="F22.5",
        weights_url="unused",
        output=OutputConfig(path=str(tmp_path / "output.zarr")),
        start_time=start_time,
        end_time=end_time,
    )
    return stream, open_stream(stream, config)


def _expected_block_means(source: xr.Dataset, factor: int) -> xr.DataArray:
    """The source variable's ``factor``-step block means, computed directly
    with numpy (so NaN propagates), labeled with each block's final
    timestamp."""
    da = source[VARIABLE]
    data = da.values.reshape(-1, factor, *da.shape[1:]).mean(axis=1)
    return da.isel({TIME_DIM: slice(factor - 1, None, factor)}).copy(data=data)


def test_chunkwise_coarsening_matches_block_means(tmp_path):
    factor, read_width = 5, 10
    source = _daily_source(20)
    stream, ds = _open_coarsened_stream(tmp_path, source, factor)

    offsets, chunks = [], []
    for time_offset in range(0, ds.sizes[TIME_DIM], read_width):
        chunk = ds.isel({TIME_DIM: slice(time_offset, time_offset + read_width)})
        output_offset, coarsened = _coarsen(chunk, time_offset, factor)
        offsets.append(output_offset)
        chunks.append(coarsened)
    result = xr.concat(chunks, dim=TIME_DIM)

    assert offsets == [0, 2]
    expected = _expected_block_means(source, factor)
    xr.testing.assert_allclose(result[VARIABLE], expected)
    # Each block mean is labeled with its block's final timestamp, which is
    # the stream's output time coordinate used for cross-stream alignment.
    xr.testing.assert_equal(stream_output_time(stream, ds), expected[TIME_DIM])
    assert "mean over 5-step blocks" in result[VARIABLE].attrs[DERIVATION_ATTR]


def test_time_range_selects_whole_blocks_by_their_labels(tmp_path):
    factor = 5
    source = _daily_source(20)
    # Block labels are 01-05, 01-10, 01-15 and 01-20. The range selects the
    # blocks whose labels fall within it, including the source days of the
    # first block that precede start_time.
    stream, ds = _open_coarsened_stream(
        tmp_path, source, factor, start_time="0151-01-03", end_time="0151-01-17"
    )

    xr.testing.assert_identical(ds, source.isel({TIME_DIM: slice(0, 15)}))
    labels = xr.date_range(
        "0151-01-05", periods=3, freq="5D", calendar="noleap", use_cftime=True
    )
    expected_time = xr.DataArray(labels, dims=TIME_DIM, coords={TIME_DIM: labels})
    xr.testing.assert_equal(stream_output_time(stream, ds), expected_time)


def test_time_range_without_whole_block_raises(tmp_path):
    with pytest.raises(AssertionError, match="no timesteps"):
        _open_coarsened_stream(
            tmp_path,
            _daily_source(20),
            factor=5,
            start_time="0151-01-06",
            end_time="0151-01-09",
        )


def test_factor_not_dividing_source_time_axis_raises(tmp_path):
    with pytest.raises(AssertionError, match="does not evenly divide"):
        _open_coarsened_stream(tmp_path, _daily_source(21), factor=5)


def test_nan_in_block_propagates_to_its_mean_only():
    factor = 5
    source = _daily_source(10)
    source[VARIABLE][6, 1, 2] = np.nan

    _, coarsened = _coarsen(source, 0, factor)

    # The expected means are NaN only at that point of the second block, and
    # xr.testing.assert_allclose requires NaN in exactly the same places.
    xr.testing.assert_allclose(
        coarsened[VARIABLE], _expected_block_means(source, factor)
    )


@pytest.mark.parametrize(
    "time_offset, size",
    [pytest.param(3, 10, id="offset"), pytest.param(0, 7, id="size")],
)
def test_chunk_without_whole_blocks_raises(time_offset, size):
    chunk = _daily_source(size)
    with pytest.raises(AssertionError, match="whole 5-step blocks"):
        _coarsen(chunk, time_offset, 5)
