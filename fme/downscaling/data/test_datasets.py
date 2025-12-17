from unittest.mock import MagicMock

import pytest
import torch
import xarray as xr

from fme.core.dataset.properties import DatasetProperties
from fme.downscaling.data.datasets import (
    BatchData,
    BatchItem,
    BatchItemDatasetAdapter,
    ContiguousDistributedSampler,
    FineCoarsePairedDataset,
    HorizontalSubsetDataset,
    LatLonCoordinates,
    PairedBatchItem,
)
from fme.downscaling.data.utils import BatchedLatLonCoordinates, ClosedInterval


@pytest.mark.parametrize("drop_last", [True, False])
def test_ContiguousDistributedSampler(drop_last):
    dataset = list(range(22))
    world_size = 4
    samplers = [
        ContiguousDistributedSampler(
            dataset, num_replicas=world_size, rank=i, drop_last=drop_last
        )
        for i in range(world_size)
    ]
    sampled = []
    batches = []
    for i, sampler in enumerate(samplers):
        rank_batch = list(iter(sampler))
        batches.append(rank_batch)
    # assert sample elements are consecutive integers

    for i, rank_batch in enumerate(batches):
        if drop_last:
            # drop last two, divide first 20 elements into 4 ranks of size 5
            assert len(rank_batch) == 5
            assert all([b - a == 1 for a, b in zip(rank_batch[:-1], rank_batch[1:])])
        else:
            # each batch padded to cover full dataset of length 22 -> 4 ranks of size 6
            if i == 3:
                # last batch is padded with the starting elements
                assert rank_batch == [15, 16, 17, 18, 19, 20, 21]
            else:
                assert len(rank_batch) == 5
                assert all(
                    [b - a == 1 for a, b in zip(rank_batch[:-1], rank_batch[1:])]
                )

    sampled = sum(batches, [])
    if drop_last:
        assert sampled == dataset[:20]


def random_named_tensor(var_names, shape):
    return {var_name: torch.rand(shape) for var_name in var_names}


def data_array(time, dims=["time"]):
    return xr.DataArray((time), dims=dims).squeeze()


def get_example_latlon_coordinates(lat_dim=4, lon_dim=5):
    return LatLonCoordinates(
        lat=torch.linspace(0.0, 90.0, lat_dim),
        lon=torch.linspace(0.0, 360.0, lon_dim),
    )


def get_example_data_tuples(num_items=3, lat_dim=8, lon_dim=16):
    data = random_named_tensor(["x"], (lat_dim, lon_dim))
    time = data_array([0])
    return [
        (
            data,
            time,
            get_example_latlon_coordinates(lat_dim, lon_dim),
        )
        for _ in range(num_items)
    ]


def get_batch_items(num_items=3, lat_dim=8, lon_dim=16):
    return [
        BatchItem(*item)
        for item in get_example_data_tuples(num_items, lat_dim, lon_dim)
    ]


def test_batched_latlon_coordinates_from_sequence():
    coords = get_example_latlon_coordinates()
    items = [coords] * 3

    # Test from_sequence method
    batched = BatchedLatLonCoordinates.from_sequence(items)

    # Verify shape matches shapes
    assert batched.lat.shape == (3, 4)  # batch_size=3, n_lat=4
    assert batched.lon.shape == (3, 5)  # batch_size=3, n_lon=5
    assert torch.allclose(batched.lat[0], coords.lat)
    assert torch.allclose(batched.lon[1], coords.lon)

    # Test dunder methods
    assert len(batched) == 3
    batched2 = BatchedLatLonCoordinates.from_sequence(items)
    assert batched == batched2

    item = batched[0]
    assert isinstance(item, LatLonCoordinates)
    assert torch.allclose(item.lat, coords.lat)
    assert torch.allclose(item.lon, coords.lon)


def test_batched_latlon_shape_inconsistent():
    coords = get_example_latlon_coordinates()
    # inconsistent shapes in join
    mishaped_coord = LatLonCoordinates(
        lat=torch.tensor(
            [
                0.0,
                30.0,
            ]
        ),
        lon=torch.tensor([0.0, 90.0, 180.0, 270.0, 360.0]),
    )
    with pytest.raises(RuntimeError):
        BatchedLatLonCoordinates.from_sequence([coords, mishaped_coord])

    # inconsistent leading dimensions
    lat = torch.randn(3, 5, 6)
    lon = torch.randn(5, 3, 12)
    with pytest.raises(ValueError):
        BatchedLatLonCoordinates(lat=lat, lon=lon)

    # inconsistent leading dimensions
    lat = torch.randn(5, 6)
    lon = torch.randn(3, 5, 12)
    with pytest.raises(ValueError):
        BatchedLatLonCoordinates(lat=lat, lon=lon)

    # fail on single dimensional input
    lat = torch.randn(5)
    lon = torch.randn(3, 12)
    with pytest.raises(ValueError):
        BatchedLatLonCoordinates(lat=lat, lon=lon)


def test_batched_latlon_coordinates_area_weights():
    # Create example coordinates
    coords = get_example_latlon_coordinates()
    items = [coords] * 3

    # Test from_sequence method
    batched = BatchedLatLonCoordinates.from_sequence(items)
    area_weights = batched.area_weights
    assert area_weights.shape == (3, 4, 5)


def test_batch_item():
    item = get_example_data_tuples(num_items=1)[0]
    batch_item = BatchItem(*item)
    batch_item.to_device()
    # test unpack
    data, time, latlon = batch_item
    assert isinstance(data, dict)
    assert isinstance(time, xr.DataArray)
    assert isinstance(latlon, LatLonCoordinates)


@pytest.mark.parametrize(
    "key, failing_value",
    [
        pytest.param("data", {"x": torch.randn(1, 8, 16)}, id="data_3D"),
        pytest.param("time", data_array([0, 1]), id="time_2D"),
        pytest.param(
            "latlon_coordinates",
            LatLonCoordinates(lat=torch.rand(2, 5), lon=torch.rand(10)),
            id="lat_2D",
        ),
        pytest.param(
            "latlon_coordinates",
            LatLonCoordinates(lat=torch.rand(10), lon=torch.rand(2, 5)),
            id="lon_2D",
        ),
    ],
)
def test_batch_item_validation(key, failing_value):
    item = get_example_data_tuples(num_items=1)[0]
    kwargs = dict(zip(["data", "time", "latlon_coordinates"], item))
    kwargs[key] = failing_value
    with pytest.raises(ValueError):
        BatchItem(**kwargs)


@pytest.mark.parametrize(
    "lat_interval,lon_interval,n_lat,n_lon,expected_n_lat,expected_n_lon",
    [
        pytest.param(("-inf", "inf"), (0, "inf"), 4, 8, 4, 8, id="no-bounds"),
        pytest.param((0.0, 0.5), (0.0, "inf"), 4, 8, 2, 8, id="lat-bounds"),
        pytest.param(("-inf", "inf"), (0.0, 0.5), 4, 8, 4, 4, id="lon-bounds"),
        pytest.param((0.0, 0.5), (0.0, 0.5), 4, 8, 2, 4, id="lat-lon-bounds"),
    ],
)
def test_horizontal_subset(
    lat_interval, lon_interval, n_lat, n_lon, expected_n_lat, expected_n_lon
):
    batch_size, n_timesteps = 2, 1
    coords = LatLonCoordinates(
        lat=torch.linspace(0.0, 1.0, n_lat), lon=torch.linspace(0.0, 1.0, n_lon)
    )

    datum: tuple[dict[str, torch.Tensor], xr.DataArray, set[str], int] = (
        {"x": torch.zeros(batch_size, n_timesteps, n_lat, n_lon)},
        xr.DataArray([0.0]),
        set(),
        0,
    )
    base_dataset = MagicMock(spec=torch.utils.data.Dataset)
    properties = MagicMock(spec=DatasetProperties)
    properties.horizontal_coordinates = coords
    properties.all_labels = MagicMock(spec=set)
    base_dataset.__getitem__.return_value = datum
    dataset = HorizontalSubsetDataset(
        dataset=base_dataset,
        properties=properties,
        lat_interval=ClosedInterval(float(lat_interval[0]), float(lat_interval[1])),
        lon_interval=ClosedInterval(float(lon_interval[0]), float(lon_interval[1])),
    )

    subset, _, labels, _ = dataset[0]
    assert labels is properties.all_labels
    assert subset["x"].shape == (
        batch_size,
        n_timesteps,
        expected_n_lat,
        expected_n_lon,
    )
    assert dataset.subset_latlon_coordinates.lat.shape == (expected_n_lat,)
    assert dataset.subset_latlon_coordinates.lon.shape == (expected_n_lon,)


def test_batch_data_from_sequence():
    num_items = 3
    items = get_batch_items(num_items=num_items)
    batched = BatchData.from_sequence(items)

    assert batched.data["x"].shape == (num_items, 8, 16)
    assert torch.equal(batched.data["x"][0], items[0].data["x"])

    assert len(batched.time) == num_items
    xr.testing.assert_equal(batched.time[0], items[0].time)

    assert batched.latlon_coordinates.lat.shape == (num_items, 8)
    assert batched.latlon_coordinates.lon.shape == (num_items, 16)


def test_batch_data_get_item():
    batch_items = get_batch_items()
    batched = BatchData.from_sequence(batch_items)
    expected = BatchItem(*batch_items[0])
    item = batched[0]
    assert item == expected


def test_batch_data_expand_and_fold():
    batch_items = get_batch_items()
    batched = BatchData.from_sequence(batch_items)
    expanded = batched.expand_and_fold(10, 1, "generated_samples")

    assert expanded.data["x"].shape == (30, 8, 16)
    assert torch.equal(expanded.data["x"][9], batched.data["x"][0])
    assert expanded.time.shape == (30,)
    assert expanded.time[9].values == batched.time[0].values
    assert expanded.latlon_coordinates.lat.shape == (30, 8)
    assert expanded.latlon_coordinates.lon.shape == (30, 16)


def get_mock_dataset(field_leading_dim=1):
    # Mock dataset that returns (data, time) tuples
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=2)
    dataset.__getitem__ = MagicMock(
        return_value=(
            {"x": torch.rand(field_leading_dim, 8, 16)},
            data_array([0]),
            set(),
            0,
        )
    )
    return dataset


def test_batch_item_dataset_adapter():
    dataset = get_mock_dataset()

    # Create adapter with example coordinates
    coords = get_example_latlon_coordinates()
    adapter = BatchItemDatasetAdapter(dataset, coords)

    # Test length matches underlying dataset
    assert len(adapter) == 2

    # Test returns BatchItem with correct attributes
    item = adapter[0]
    assert isinstance(item, BatchItem)
    assert "x" in item.data
    assert item.data["x"].shape == (8, 16)
    assert item.time == 0
    assert torch.equal(item.latlon_coordinates.lat, coords.lat)
    assert torch.equal(item.latlon_coordinates.lon, coords.lon)


def test_batch_item_dataset_adapter_validation():
    # Test validation of input data dimensions
    dataset = get_mock_dataset(field_leading_dim=2)

    coords = get_example_latlon_coordinates()
    adapter = BatchItemDatasetAdapter(dataset, coords)

    with pytest.raises(ValueError, match="Expected 2D spatial data"):
        _ = adapter[0]


def test_paired_batch_item_validation():
    fine_item = get_batch_items(num_items=1, lat_dim=8, lon_dim=16)[0]
    coarse_item = get_batch_items(num_items=1, lat_dim=4, lon_dim=8)[0]
    mismatched_dim_scales = get_batch_items(num_items=1, lat_dim=2, lon_dim=8)[0]
    non_divisible_dim = get_batch_items(num_items=1, lat_dim=3, lon_dim=8)[0]

    # This should work - times match
    paired = PairedBatchItem(fine_item, coarse_item)

    # Create coarse item with different time
    coarse_time_mismatch = data_array([1])  # Different time
    coarse_item_mismatch = BatchItem(
        coarse_item.data, coarse_time_mismatch, coarse_item.latlon_coordinates
    )

    # This should raise ValueError due to time mismatch
    with pytest.raises(ValueError):
        PairedBatchItem(fine_item, coarse_item_mismatch)

    with pytest.raises(ValueError):
        PairedBatchItem(fine_item, mismatched_dim_scales)

    with pytest.raises(ValueError):
        PairedBatchItem(fine_item, non_divisible_dim)

    # Test device movement
    paired.to_device()


def test_fine_coarse_paired_dataset():
    # Create two identical mock datasets
    dataset1 = get_mock_dataset()
    dataset2 = get_mock_dataset()

    coords = get_example_latlon_coordinates()
    adapter1 = BatchItemDatasetAdapter(dataset1, coords)
    adapter2 = BatchItemDatasetAdapter(dataset2, coords)

    # Test with identical datasets - should pass
    paired = FineCoarsePairedDataset(adapter1, adapter2)

    # Test length matches underlying datasets
    assert len(paired) == 2
    paired[0]

    # Create dataset with different length
    dataset3 = get_mock_dataset()
    dataset3.__len__.return_value = 3
    adapter3 = BatchItemDatasetAdapter(dataset3, coords)
    with pytest.raises(ValueError):
        FineCoarsePairedDataset(adapter1, adapter3)


def test_BatchData_slice_latlon():
    sample_data = get_batch_items()
    batch = BatchData.from_sequence(sample_data)

    lat_slice = slice(1, 3)
    lon_slice = slice(2, 5)
    batch_slice = batch.latlon_slice(
        lat_slice=lat_slice,
        lon_slice=lon_slice,
    )

    assert torch.equal(
        batch_slice.latlon_coordinates.lat,
        batch.latlon_coordinates.lat[:, lat_slice],
    )
    assert torch.equal(
        batch_slice.latlon_coordinates.lon,
        batch.latlon_coordinates.lon[:, lon_slice],
    )
    assert torch.equal(
        batch_slice.data["x"],
        batch.data["x"][:, lat_slice, lon_slice],
    )
