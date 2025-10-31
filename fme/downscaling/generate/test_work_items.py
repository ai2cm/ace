from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from fme.downscaling.data import BatchData
from fme.downscaling.generate.work_items import (
    LoadedWorkItem,
    SliceItemDataset,
    SliceWorkItem,
    _generate_slices,
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


def test_slice_work_item_insert_slices():
    """Test that insert_slices returns correct dict."""
    item = SliceWorkItem(time_slice=slice(0, 5), ens_slice=slice(2, 10))
    slices = item.insert_slices

    assert isinstance(slices, dict)
    assert "time" in slices
    assert "ensemble" in slices
    assert slices["time"] == slice(0, 5)
    assert slices["ensemble"] == slice(2, 10)


def test_slice_work_item_with_batch_creates_loaded_work_item():
    """Test with_batch classmethod creates LoadedWorkItem."""
    work_item = SliceWorkItem(time_slice=slice(0, 5), ens_slice=slice(0, 10))
    mock_batch = Mock(spec=BatchData)

    loaded_item = SliceWorkItem.with_batch(work_item, mock_batch)

    assert isinstance(loaded_item, LoadedWorkItem)
    assert loaded_item.time_slice == work_item.time_slice
    assert loaded_item.ens_slice == work_item.ens_slice
    assert loaded_item.batch is mock_batch


# Tests for LoadedWorkItem dataclass


def test_loaded_work_item_requires_batch_data():
    """LoadedWorkItem should raise error if batch is None."""
    with pytest.raises(ValueError, match="must be created with batch data"):
        LoadedWorkItem(time_slice=slice(0, 5), ens_slice=slice(0, 10), batch=None)


def test_loaded_work_item_inherits_from_slice_work_item():
    """LoadedWorkItem should inherit SliceWorkItem properties."""
    work_item = SliceWorkItem(time_slice=slice(0, 5), ens_slice=slice(3, 10))
    mock_batch = Mock(spec=BatchData)
    item = work_item.with_batch(work_item, mock_batch)

    # Should have all SliceWorkItem properties
    assert item.n_ens == 7
    assert item.time_indices == [0, 1, 2, 3, 4]
    assert item.insert_slices == {"time": slice(0, 5), "ensemble": slice(3, 10)}


# Tests for get_work_items function


@pytest.mark.parametrize(
    "n_times,n_ens,n_items_per_gpu,expected_count",
    [
        pytest.param(1, 1, 4, 1, id="single_time_single_ensemble"),
        pytest.param(5, 1, 4, 2, id="multiple_times"),
        pytest.param(1, 8, 4, 2, id="multiple_ensembles"),
        pytest.param(10, 10, 4, 25, id="multiple_times_and_ensembles"),
        pytest.param(2, 2, 4, 1, id="fits_one_batch"),
    ],
)
def test_get_work_items_count(n_times, n_ens, n_items_per_gpu, expected_count):
    """Test that correct number of work items are generated."""
    work_items = get_work_items(n_times, n_ens, n_items_per_gpu, dist=None)
    # Account for potential padding
    assert len(work_items) >= expected_count


def test_get_work_items_distributed_padding_added():
    """Test that padding is added for distributed training."""
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


def test_get_work_items_no_padding_when_evenly_divisible():
    """No padding needed when work items divide evenly by world_size."""
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
    mock_item.data = {"var1": torch.randn(10, 10)}
    dataset.__getitem__.return_value = mock_item
    return dataset


@pytest.fixture
def mock_topography():
    """Create a mock Topography object."""
    topo = MagicMock()
    topo.data = MagicMock()
    topo.data.shape = (64, 64)
    return topo


@pytest.fixture
def sample_work_items():
    """Create sample work items for testing."""
    return [
        SliceWorkItem(time_slice=slice(0, 2), ens_slice=slice(0, 4)),
        SliceWorkItem(time_slice=slice(2, 4), ens_slice=slice(0, 4)),
        SliceWorkItem(time_slice=slice(0, 2), ens_slice=slice(4, 8)),
    ]


# Tests for SliceItemDataset class


def test_slice_item_dataset_length(sample_work_items, mock_dataset, mock_topography):
    """Test __len__ returns correct value."""
    dataset = SliceItemDataset(sample_work_items, mock_dataset, mock_topography)
    assert len(dataset) == 3


def test_slice_item_dataset_getitem_returns_loaded_work_item(
    sample_work_items, mock_dataset, mock_topography
):
    """Test __getitem__ returns LoadedWorkItem with topography."""
    dataset = SliceItemDataset(sample_work_items, mock_dataset, mock_topography)

    # Mock BatchData.from_sequence
    with patch(
        "fme.downscaling.generate.work_items.BatchData.from_sequence"
    ) as mock_from_sequence:
        mock_batch = Mock(spec=BatchData)
        mock_from_sequence.return_value = mock_batch

        result = dataset[0]

        # Should return tuple of (LoadedWorkItem, Topography)
        loaded_item, topo = result
        assert isinstance(loaded_item, LoadedWorkItem)
        assert topo is mock_topography


def test_slice_item_dataset_max_output_shape(
    sample_work_items, mock_dataset, mock_topography
):
    """Test max_output_shape property."""
    dataset = SliceItemDataset(sample_work_items, mock_dataset, mock_topography)
    shape = dataset.max_output_shape

    # First item: time_slice=slice(0,2), ens_slice=slice(0,4)
    # n_times = 2, n_ens = 4, spatial = (64, 64)
    assert shape == (2, 4, 64, 64)


def test_slice_item_dataset_dtype_property(
    sample_work_items, mock_dataset, mock_topography
):
    """Test dtype property infers from dataset."""
    # Set up mock to return specific dtype
    mock_item = MagicMock()
    mock_tensor = torch.randn(10, 10, dtype=torch.float32)
    mock_item.data = {"var1": mock_tensor}
    mock_dataset.__getitem__.return_value = mock_item

    dataset = SliceItemDataset(sample_work_items, mock_dataset, mock_topography)
    assert dataset.dtype == torch.float32
