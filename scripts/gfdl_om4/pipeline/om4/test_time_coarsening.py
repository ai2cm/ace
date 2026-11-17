"""Tests of the time coarsening (``time_block_mean_factor``) of a stream."""

import numpy as np
import pytest
import xarray as xr
from packaging.version import Version

from ..dims import TIME_DIM
from ..postprocess import DERIVATION_ATTR
from .config import StreamConfig
from .time_coarsening import coarsen, select_whole_blocks, stream_output_time

VARIABLE = "tos"


# xarray 2026.9.0's Dataset coarsen mean rejects the skipna argument that
# coarsen passes. This is fixed upstream by
# https://github.com/pydata/xarray/pull/11686, which is not yet released.
xfail_with_xarray_2026_9_0 = pytest.mark.xfail(
    Version(xr.__version__) == Version("2026.9.0"),
    reason=(
        "xarray 2026.9.0 only: Dataset coarsen mean rejects skipna "
        "(pydata/xarray#11686)"
    ),
    raises=TypeError,
    strict=True,
)


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


def _coarsened_stream(factor: int) -> StreamConfig:
    return StreamConfig(
        name="surface",
        store="unused",
        variables=[VARIABLE],
        time_block_mean_factor=factor,
    )


def _expected_block_means(source: xr.Dataset, factor: int) -> xr.DataArray:
    """The source variable's ``factor``-step block means, computed directly
    with numpy (so NaN propagates), labeled with each block's final
    timestamp."""
    da = source[VARIABLE]
    data = da.values.reshape(-1, factor, *da.shape[1:]).mean(axis=1)
    return da.isel({TIME_DIM: slice(factor - 1, None, factor)}).copy(data=data)


@xfail_with_xarray_2026_9_0
def test_chunkwise_coarsening_matches_block_means():
    factor, read_width = 5, 10
    source = _daily_source(20)

    offsets, chunks = [], []
    for time_offset in range(0, source.sizes[TIME_DIM], read_width):
        chunk = source.isel({TIME_DIM: slice(time_offset, time_offset + read_width)})
        output_offset, coarsened = coarsen(chunk, time_offset, factor)
        offsets.append(output_offset)
        chunks.append(coarsened)
    result = xr.concat(chunks, dim=TIME_DIM)

    assert offsets == [0, 2]
    expected = _expected_block_means(source, factor)
    xr.testing.assert_allclose(result[VARIABLE], expected)
    # Each block mean is labeled with its block's final timestamp, which is
    # the stream's output time coordinate used for cross-stream alignment.
    stream = _coarsened_stream(factor)
    xr.testing.assert_equal(stream_output_time(stream, source), expected[TIME_DIM])
    assert "mean over 5-step blocks" in result[VARIABLE].attrs[DERIVATION_ATTR]


def test_time_range_selects_whole_blocks_by_their_labels():
    factor = 5
    source = _daily_source(20)
    # Block labels are 01-05, 01-10, 01-15 and 01-20. The range selects the
    # blocks whose labels fall within it, including the source days of the
    # first block that precede the start time.
    selected = select_whole_blocks(source, factor, "0151-01-03", "0151-01-17", "test")

    xr.testing.assert_identical(selected, source.isel({TIME_DIM: slice(0, 15)}))
    labels = xr.date_range(
        "0151-01-05", periods=3, freq="5D", calendar="noleap", use_cftime=True
    )
    expected_time = xr.DataArray(labels, dims=TIME_DIM, coords={TIME_DIM: labels})
    stream = _coarsened_stream(factor)
    xr.testing.assert_equal(stream_output_time(stream, selected), expected_time)


def test_time_range_without_whole_block_selects_nothing():
    selected = select_whole_blocks(
        _daily_source(20), 5, "0151-01-06", "0151-01-09", "test"
    )
    assert selected.sizes[TIME_DIM] == 0


def test_factor_not_dividing_source_time_axis_raises():
    with pytest.raises(AssertionError, match="does not evenly divide"):
        select_whole_blocks(_daily_source(21), 5, None, None, "test")


@xfail_with_xarray_2026_9_0
def test_nan_in_block_propagates_to_its_mean_only():
    factor = 5
    source = _daily_source(10)
    source[VARIABLE][6, 1, 2] = np.nan

    _, coarsened = coarsen(source, 0, factor)

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
        coarsen(chunk, time_offset, 5)
