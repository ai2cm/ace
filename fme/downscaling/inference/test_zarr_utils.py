import pytest

from fme.downscaling.inference.zarr_utils import (
    NotReducibleError,
    _recursive_latlon_chunksize_search,
    _total_size_mb,
    determine_zarr_chunks,
)

# Tests for _total_size_mb helper function


def test_total_size_mb_calculation():
    result = _total_size_mb((512, 512), 4)
    assert result == 1.0


# Tests for _recursive_chunksize_search core logic


@pytest.mark.parametrize(
    "shape,reduce_dim,expected",
    [
        # 512 x 512  @ float32 = 1 MB, equivalent to single model patch
        pytest.param((512, 512), 0, (512, 512), id="no_reduction_needed"),
        pytest.param(
            (512**2 * 4, 1), 0, (512**2, 1), id="reduces_only_first_dimension"
        ),
        pytest.param(
            (512**2 * 4, 1),
            1,
            (512**2, 1),
            id="reduces_only_first_dimension_starts_at_second",
        ),
        pytest.param(
            (1, 512**2 * 4), 1, (1, 512**2), id="reduces_only_second_dimension"
        ),
        pytest.param((1024, 1024), 0, (512, 512), id="reduces_both_dimensions"),
        pytest.param((2048, 1024), 0, (512, 512), id="reduces_multiple_times"),
        pytest.param((2048, 1024), 1, (1024, 256), id="reduces_multiple_times"),
        pytest.param((1001, 1001), 0, (500, 500), id="odd_dimensions"),
    ],
)
def test_recursive_chunksize_search(shape, reduce_dim, expected):
    """Test the recursive chunk size search logic."""
    target_mb = 1
    bytes_per_element = 4  # float32
    result = _recursive_latlon_chunksize_search(
        shape, bytes_per_element, reduce_dim, target_mb
    )
    assert result == expected


def test_recursive_chunksize_search_raises_when_element_exceeds_target():
    """Should raise NotReducibleError when single element > target."""
    shape = (1, 1)
    with pytest.raises(NotReducibleError):
        _recursive_latlon_chunksize_search(
            shape, bytes_per_element=1024**2 + 1, reduce_dim=0, target_mb=1
        )


# Tests for determine_zarr_chunks public API


def test_determine_zarr_chunks_returns_dict_with_correct_keys():
    """Should return dict with keys matching input dims."""
    dims = ("time", "ensemble", "latitude", "longitude")
    shape = (10, 5, 100, 100)
    result = determine_zarr_chunks(dims, shape, bytes_per_element=4)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(dims)
    assert len(result) == 4


def test_determine_zarr_chunks_preserves_dimension_order():
    """Output dict should preserve dimension order."""
    dims = ("time", "ensemble", "latitude", "longitude")
    shape = (10, 5, 100, 100)
    result = determine_zarr_chunks(dims, shape, bytes_per_element=4)

    assert tuple(result.keys()) == dims


@pytest.mark.parametrize(
    "dims, shape",
    [
        ([*"abcd"], (1, 1, 1)),
        ([*"abcd"], (1, 1, 1, 1, 1)),
        ([*"abc"], (1, 1, 1, 1)),
        ([*"abc"], (1, 1, 1, 1)),
        ([*"abc"], (1, 1, 1)),
    ],
)
def test_determine_zarr_chunks_raises_on_wrong_shape_length(dims, shape):
    """Should raise ValueError if shape is not length 4."""

    with pytest.raises(ValueError, match="must be of length 4"):
        determine_zarr_chunks(dims, shape, bytes_per_element=4)


@pytest.mark.parametrize(
    "max_time_ens_shape",
    [(5, 2), (1, 1)],
)
def test_determine_zarr_chunks(max_time_ens_shape):
    """
    Test full determine_zarr_chunks function end-to-end.
    Hard coded to always return 1,1 for time and ensemble dims.
    """
    shape = tuple(list(max_time_ens_shape) + [1024, 1024])
    dims = ("time", "ensemble", "latitude", "longitude")
    result = determine_zarr_chunks(dims, shape, bytes_per_element=4, target_mb=1)
    expected = dict(zip(dims, [1, 1, 512, 512]))
    assert result == expected
