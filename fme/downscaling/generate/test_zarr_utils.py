import pytest

from fme.downscaling.generate.zarr_utils import (
    NotReducibleError,
    _recursive_chunksize_search,
    _total_size_mb,
    determine_zarr_chunks,
)

# Tests for _total_size_mb helper function


@pytest.mark.parametrize(
    "shape,bytes_per_element,expected_mb",
    [
        pytest.param((10, 10, 10, 10), 4, 0, id="40kb_rounds_to_0mb"),
        pytest.param((100, 100, 100, 100), 4, 381, id="381mb"),
        pytest.param((256, 8, 64, 64), 4, 32, id="32mb_exactly"),
        pytest.param((1, 1, 1, 1), 8, 0, id="8_bytes"),
        pytest.param((512, 4, 128, 128), 2, 64, id="64mb"),
    ],
)
def test_total_size_mb_calculation(shape, bytes_per_element, expected_mb):
    """Test size calculations for various shapes and data types."""
    result = _total_size_mb(shape, bytes_per_element)
    assert result == expected_mb


# Tests for _recursive_chunksize_search core logic


@pytest.mark.parametrize(
    "shape,reduce_dim,expected",
    [
        # 2, 2, 256, 256  @ float32 = 1 MB
        #
        pytest.param((2, 2, 256, 256), 0, (2, 2, 256, 256), id="no_reduction_needed"),
        pytest.param((4, 2, 256, 256), 0, (2, 2, 256, 256), id="reduce_time_dimension"),
        pytest.param(
            (1, 8, 256, 256), 0, (1, 4, 256, 256), id="reduce_ensemble_dimension"
        ),
        pytest.param(
            (4, 2, 256, 256),
            1,
            (4, 1, 256, 256),
            id="reduce_ensemble_dimension_start_at_ens_dim",
        ),
        pytest.param(
            (2, 2, 512, 256), 2, (2, 2, 256, 256), id="reduce_latitude_dimension"
        ),
        pytest.param(
            (2, 2, 256, 512), 3, (2, 2, 256, 256), id="reduce_longitude_dimension"
        ),
        pytest.param(
            (2, 2, 1024, 512), 2, (2, 2, 256, 256), id="reduce_latlon_alternating"
        ),
        pytest.param(
            (4, 4, 2048, 2048), 0, (1, 1, 512, 512), id="multiple_reductions_needed"
        ),
    ],
)
def test_recursive_chunksize_search(shape, reduce_dim, expected):
    """Test the recursive chunk size search logic."""
    target_mb = 1
    bytes_per_element = 4  # float32
    result = _recursive_chunksize_search(
        shape, bytes_per_element, reduce_dim, target_mb
    )
    assert result == expected


def test_recursive_chunksize_search_raises_when_element_exceeds_target():
    """Should raise NotReducibleError when single element > target."""
    shape = (1, 1, 1, 1)
    # 1MB per element (1048576 bytes), but target is 0.5MB
    with pytest.raises(NotReducibleError):
        _recursive_chunksize_search(
            shape, bytes_per_element=int(5e6), reduce_dim=0, target_mb=1
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


def test_determine_zarr_chunks_raises_on_wrong_shape_length():
    """Should raise ValueError if shape is not length 4."""
    dims = ("time", "ensemble", "latitude", "longitude")

    with pytest.raises(ValueError, match="must be of length 4"):
        determine_zarr_chunks(dims, (10, 10, 10), bytes_per_element=4)

    with pytest.raises(ValueError, match="must be of length 4"):
        determine_zarr_chunks(dims, (10, 10, 10, 10, 10), bytes_per_element=4)
