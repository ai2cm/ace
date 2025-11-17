from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from fme.downscaling.data import BatchData
from fme.downscaling.inference.constants import ENSEMBLE_NAME, TIME_NAME
from fme.downscaling.inference.work_items import (
    LoadedSliceWorkItem,
    SliceItemDataset,
    SliceWorkItem,
    _generate_slices,
    _get_slice_step_size_for_distributed,
    _get_time_ens_slice_step_multi_gpu,
    get_work_items,
)


@pytest.mark.parametrize(
    "total,step,expected",
    [
        pytest.param(
            10,
            2,
            [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8), slice(8, 10)],
            id="even_division",
        ),
        pytest.param(10, 5, [slice(0, 5), slice(5, 10)], id="half"),
        pytest.param(10, 10, [slice(0, 10)], id="single_slice"),
        pytest.param(7, 3, [slice(0, 3), slice(3, 6), slice(6, 7)], id="remainder"),
        pytest.param(1, 1, [slice(0, 1)], id="single_element"),
        pytest.param(5, 100, [slice(0, 5)], id="step_larger_than_total"),
        pytest.param(0, 5, [], id="empty"),
    ],
)
def test_generate_slices(total, step, expected):
    """Test slice generation for various total and step combinations."""
    result = _generate_slices(total, step)
    assert result == expected


# Tests for SliceWorkItem dataclass


def test_slice_work_item_initialization():
    """Test basic creation and attribute access."""
    item = SliceWorkItem(
        time_slice=slice(0, 5), ens_slice=slice(0, 10), is_padding=False
    )
    assert item.time_slice == slice(0, 5)
    assert item.ens_slice == slice(0, 10)
    assert item.is_padding is False


@pytest.mark.parametrize(
    "bad_slice",
    [
        pytest.param(slice(1, None), id="stop_undefined"),
        pytest.param(slice(None, 5), id="start_undefined"),
        pytest.param(slice(-3, -1), id="negatives"),
        pytest.param(slice(-2, 10), id="negative_start"),
        pytest.param(slice(0, -5), id="negative_stop"),
        pytest.param(slice(3, 2), id="start_greater_than_stop"),
    ],
)
def test_slice_work_item_invalid_slices(bad_slice):
    """Test that invalid slices raise ValueError."""
    with pytest.raises(ValueError):
        SliceWorkItem(time_slice=bad_slice, ens_slice=slice(0, 5))

    with pytest.raises(ValueError):
        SliceWorkItem(time_slice=slice(0, 5), ens_slice=bad_slice)


def test_slice_work_item_n_ens_calculated():
    """Test that n_ens is calculated correctly in __post_init__."""
    item = SliceWorkItem(time_slice=slice(0, 5), ens_slice=slice(2, 10))
    assert item.n_ens == 8


@pytest.mark.parametrize(
    "time_slice,expected_indices",
    [
        pytest.param(slice(0, 3), [0, 1, 2], id="small_range"),
        pytest.param(slice(5, 8), [5, 6, 7], id="offset_range"),
        pytest.param(slice(0, 1), [0], id="single"),
    ],
)
def test_slice_work_item_time_indices(time_slice, expected_indices):
    """Test that time_indices returns correct list."""
    item = SliceWorkItem(time_slice=time_slice, ens_slice=slice(0, 5))
    assert item.time_indices == expected_indices


def test_slice_work_item_dim_insert_slices():
    """Test that dim_insert_slices returns correct dict."""
    item = SliceWorkItem(time_slice=slice(0, 5), ens_slice=slice(2, 10))
    slices = item.dim_insert_slices

    assert isinstance(slices, dict)
    assert TIME_NAME in slices
    assert ENSEMBLE_NAME in slices
    assert slices[TIME_NAME] == slice(0, 5)
    assert slices[ENSEMBLE_NAME] == slice(2, 10)


def test_slice_work_item_with_batch_creates_loaded_work_item():
    """Test with_batch classmethod creates LoadedWorkItem."""
    work_item = SliceWorkItem(time_slice=slice(0, 5), ens_slice=slice(0, 10))
    mock_batch = Mock(spec=BatchData)

    loaded_item = SliceWorkItem.with_batch(work_item, mock_batch)

    assert isinstance(loaded_item, LoadedSliceWorkItem)
    assert loaded_item.time_slice == work_item.time_slice
    assert loaded_item.ens_slice == work_item.ens_slice
    assert loaded_item.batch is mock_batch


# Tests for LoadedWorkItem dataclass


def test_loaded_work_item_inherits_from_slice_work_item():
    """LoadedWorkItem should inherit SliceWorkItem properties."""
    work_item = SliceWorkItem(time_slice=slice(0, 5), ens_slice=slice(3, 10))
    mock_batch = Mock(spec=BatchData)
    item = work_item.with_batch(work_item, mock_batch)

    # Should have all SliceWorkItem properties
    assert item.n_ens == 7
    assert item.time_indices == [0, 1, 2, 3, 4]
    assert item.dim_insert_slices == {"time": slice(0, 5), "ensemble": slice(3, 10)}


# Tests for get_work_items function


@pytest.mark.parametrize(
    "n_times,n_ens,max_item_per_gpu,expected_count",
    [
        pytest.param(1, 1, 4, 1, id="single_time_single_ensemble"),
        pytest.param(5, 1, 4, 2, id="multiple_times"),
        pytest.param(1, 8, 4, 2, id="multiple_ensembles"),
        pytest.param(10, 10, 4, 25, id="multiple_times_and_ensembles"),
        pytest.param(2, 2, 4, 1, id="fits_one_batch"),
    ],
)
def test_get_work_items_count(n_times, n_ens, max_item_per_gpu, expected_count):
    """Test that correct number of work items are generated."""
    work_items = get_work_items(n_times, n_ens, max_item_per_gpu, dist=None)
    # Account for potential padding
    assert len(work_items) >= expected_count


def test_get_work_items_slice_contents_single_gpu():
    """Test that generated work items have correct slices."""
    n_times = 4
    n_ens = 6
    max_item_per_gpu = 4

    work_items = get_work_items(n_times, n_ens, max_item_per_gpu, dist=None)
    assert len(work_items) == 8

    # Check first work item
    first_item = work_items[0]
    assert first_item.time_slice == slice(0, 1)
    assert first_item.ens_slice == slice(0, 4)

    # Check second work item
    second_item = work_items[1]
    assert second_item.time_slice == slice(0, 1)
    assert second_item.ens_slice == slice(4, 6)

    # Check second to last work item
    third_item = work_items[-2]
    assert third_item.time_slice == slice(3, 4)
    assert third_item.ens_slice == slice(0, 4)


def test__get_slice_step_size_for_distributed():
    """
    Test the slice step size adjustment for distributed training
    where it tries to find better divisors to prevent large imbalances
    in work item sizes across multiple GPUs.
    """
    # Case where total size divides evenly by requested step
    assert _get_slice_step_size_for_distributed(10, 2) == 2

    # Case where total size does not divide evenly
    # finds another divisor
    assert _get_slice_step_size_for_distributed(16, 5) == 4

    # Case where requested step is larger than total size
    assert _get_slice_step_size_for_distributed(5, 10) == 5

    # Case with no intermediate divisor, full reduction of step
    assert _get_slice_step_size_for_distributed(7, 3) == 2


def test__get_time_ens_slice_step_multi_gpu():
    """
    Test combined time and ensemble slice step size calculation
    for multi-GPU distributed generation.

    Checks that the function balances the step sizes to minimize
    the difference between any trailing work item sizes
    """

    # max smaller than n_ens, n_ens divisible
    t_step, e_step = _get_time_ens_slice_step_multi_gpu(10, 8, 4)
    assert e_step == 4
    assert t_step == 1

    # max smaller than n_ens, not evenly divisible
    # 12 would provide remainder of 1, for last ens slice
    # reduction to 7 gives to slices of len 7 and 6 (passes diff=1 test)
    # 12//7 = 1 time slice per work item,
    # slightly more than half the max requested itemsize
    t_step, e_step = _get_time_ens_slice_step_multi_gpu(10, 13, 12)
    assert e_step == 7
    assert t_step == 1

    # max larger than n_ens, n_ens smaller than time, time divisible
    # ens step = 3, (3 % 6)
    # time step = 2 (6 // 3), which divides 10 evenly
    t_step, e_step = _get_time_ens_slice_step_multi_gpu(10, 3, 6)
    assert e_step == 3
    assert t_step == 2

    # max larger than n_ens, n_ens smaller than time, time not divisible
    # ens step = 4 (4 % 16)
    # initial time step would be 12, but would have a large slice differential
    # of (0, 12) and (12, 13)
    # load balancing should find step of 2 in time, since (12, 6, 4, 3) all have
    # the length 1 slice problem, and all other values produce larger differentials
    t_step, e_step = _get_time_ens_slice_step_multi_gpu(13, 4, 16)
    assert e_step == 4
    assert t_step == 2


@pytest.mark.parametrize(
    "n_times,n_ens,max_item_per_gpu,expected_work_items",
    [
        # reproduce cases above to run an integration test
        # for load balancing
        (10, 8, 4, 20),
        (10, 13, 12, 20),
        (10, 3, 6, 5),
        (13, 4, 16, 7),
    ],
)
def test_get_work_items_multi_gpu_load_balancing(
    n_times, n_ens, max_item_per_gpu, expected_work_items
):
    """Test load balancing of work item sizes for multi-GPU distributed training.

    The load balancing algorithm reduces slice step sizes to minimize the difference
    between the largest and smallest work items.
    """
    mock_dist = MagicMock()
    mock_dist.is_distributed.return_value = True
    mock_dist.world_size = 2

    work_items = get_work_items(n_times, n_ens, max_item_per_gpu, dist=mock_dist)

    # Filter out padding items for load balance calculation
    non_padding_items = [item for item in work_items if not item.is_padding]

    assert len(non_padding_items) == expected_work_items

    # check slices cover the full range
    covered_times: set[int] = set()
    covered_ens: set[int] = set()
    for item in non_padding_items:
        covered_times.update(range(item.time_slice.start, item.time_slice.stop))
        covered_ens.update(range(item.ens_slice.start, item.ens_slice.stop))
    assert covered_times == set(range(n_times))
    assert covered_ens == set(range(n_ens))


def test_get_work_items_slice_contents_multi_gpu():
    """
    Test showcasing a specific example of distributed load balancing
    compared to the single GPU case.
    """
    n_times = 4
    n_ens = 6
    max_item_per_gpu = 4

    dist = MagicMock()
    dist.is_distributed.return_value = True
    dist.world_size = 2

    work_items = get_work_items(n_times, n_ens, max_item_per_gpu, dist=dist)
    assert len(work_items) == 8

    # Check first work item
    first_item = work_items[0]
    assert first_item.time_slice == slice(0, 1)
    assert first_item.ens_slice == slice(0, 3)

    # Check second work item
    second_item = work_items[1]
    assert second_item.time_slice == slice(0, 1)
    assert second_item.ens_slice == slice(3, 6)

    # Check second to last work item
    third_item = work_items[-2]
    assert third_item.time_slice == slice(3, 4)
    assert third_item.ens_slice == slice(0, 3)


def test_get_work_items_distributed_padding_added():
    """
    number of GPU > than number of work items produced from requested
    max_samples_per_gpu. requires padding to provide work to all
    GPUs for final distributed generation step.
    """

    # Mock distributed instance with world_size = 3
    mock_dist = MagicMock()
    mock_dist.is_distributed.return_value = True
    mock_dist.world_size = 3

    # Create work items that don't divide evenly by 3
    work_items = get_work_items(2, 2, 4, dist=mock_dist)

    # Should have padding to make total divisible by 3
    assert len(work_items) % 3 == 0

    # Count padding items
    padding_items = [item for item in work_items if item.is_padding]
    regular_items = [item for item in work_items if not item.is_padding]

    assert len(padding_items) == 2
    assert len(regular_items) == 1


def test_get_work_items_distributed_no_padding_when_evenly_divisible():
    """No padding needed when work items divide evenly by number of GPUs."""
    mock_dist = MagicMock()
    mock_dist.is_distributed.return_value = True
    mock_dist.world_size = 2

    work_items = get_work_items(8, 2, 4, dist=mock_dist)

    # Should be exactly 4 items, no padding needed
    padding_items = [item for item in work_items if item.is_padding]
    assert len(padding_items) == 0


# Fixtures for SliceItemDataset tests


@pytest.fixture
def mock_dataset():
    """Create a mock BatchItemDataset."""
    dataset = MagicMock()
    # Mock dataset items
    mock_item = MagicMock()
    mock_item.data = {"var1": torch.randn(10, 64, 64)}
    mock_item.horizontal_shape = (64, 64)
    dataset.__getitem__.return_value = mock_item
    return dataset


@pytest.fixture
def sample_work_items():
    """Create sample work items for testing."""
    return [
        SliceWorkItem(time_slice=slice(0, 2), ens_slice=slice(0, 4)),
        SliceWorkItem(time_slice=slice(2, 4), ens_slice=slice(0, 4)),
        SliceWorkItem(time_slice=slice(0, 2), ens_slice=slice(4, 8)),
    ]


# Tests for SliceItemDataset class


def test_slice_item_dataset_length(sample_work_items, mock_dataset):
    """Test __len__ returns correct value."""
    dataset = SliceItemDataset(sample_work_items, mock_dataset)
    assert len(dataset) == 3


def test_slice_item_dataset_getitem_returns_loaded_work_item(
    sample_work_items, mock_dataset
):
    """Test __getitem__ returns LoadedWorkItem."""
    dataset = SliceItemDataset(sample_work_items, mock_dataset)

    # Mock BatchData.from_sequence
    with patch(
        "fme.downscaling.inference.work_items.BatchData.from_sequence"
    ) as mock_from_sequence:
        mock_batch = Mock(spec=BatchData)
        mock_from_sequence.return_value = mock_batch

        result = dataset[0]
        assert isinstance(result, LoadedSliceWorkItem)


def test_slice_item_dataset_max_output_shape(
    sample_work_items,
    mock_dataset,
):
    """Test max_output_shape property."""
    dataset = SliceItemDataset(sample_work_items, mock_dataset)
    shape = dataset.max_output_shape

    # First item: time_slice=slice(0,2), ens_slice=slice(0,4)
    # n_times = 2, n_ens = 4, spatial = (64, 64)
    assert shape == (2, 4, 64, 64)


def test_slice_item_dataset_dtype_property(
    sample_work_items,
    mock_dataset,
):
    """Test dtype property infers from dataset."""
    # Set up mock to return specific dtype
    mock_item = MagicMock()
    mock_tensor = torch.randn(10, 10, dtype=torch.float32)
    mock_item.data = {"var1": mock_tensor}
    mock_dataset.__getitem__.return_value = mock_item

    dataset = SliceItemDataset(sample_work_items, mock_dataset)
    assert dataset.dtype == torch.float32
