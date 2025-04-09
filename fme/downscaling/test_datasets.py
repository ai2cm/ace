from unittest.mock import MagicMock

import pytest
import torch
import xarray as xr

from fme.core.dataset.data_typing import Dataset
from fme.core.dataset.xarray import DatasetProperties
from fme.downscaling.datasets import (
    BatchData,
    BatchedLatLonCoordinates,
    BatchItem,
    BatchItemDatasetAdapter,
    ClosedInterval,
    FineCoarsePairedDataset,
    HorizontalSubsetDataset,
    LatLonCoordinates,
    PairedBatchItem,
    RandomSpatialSubsetPairedDataset,
    _generate_random_extent_slice,
    _scale_slice,
    _subset_fine_coarse,
    _subset_horizontal,
)


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
            torch.randn(lat_dim, lon_dim),
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
    data, time, latlon, topo = batch_item
    assert isinstance(data, dict)
    assert isinstance(time, xr.DataArray)
    assert isinstance(latlon, LatLonCoordinates)
    assert isinstance(topo, torch.Tensor)


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
        pytest.param("topography", torch.randn(1, 8, 16), id="topo_3D"),
    ],
)
def test_batch_item_validation(key, failing_value):
    item = get_example_data_tuples(num_items=1)[0]
    kwargs = dict(zip(["data", "time", "latlon_coordinates", "topography"], item))
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

    datum = (
        {"x": torch.zeros(batch_size, n_timesteps, n_lat, n_lon)},
        xr.DataArray([0.0]),
    )
    base_dataset = MagicMock(spec=Dataset)
    properties = MagicMock(spec=DatasetProperties)
    properties.horizontal_coordinates = coords
    topography = torch.randn(n_lat, n_lon)
    base_dataset.__getitem__.return_value = datum
    dataset = HorizontalSubsetDataset(
        dataset=base_dataset,
        properties=properties,
        lat_interval=ClosedInterval(float(lat_interval[0]), float(lat_interval[1])),
        lon_interval=ClosedInterval(float(lon_interval[0]), float(lon_interval[1])),
        topography=topography,
    )

    subset, _ = dataset[0]
    assert subset["x"].shape == (
        batch_size,
        n_timesteps,
        expected_n_lat,
        expected_n_lon,
    )
    assert dataset.subset_latlon_coordinates.lat.shape == (expected_n_lat,)
    assert dataset.subset_latlon_coordinates.lon.shape == (expected_n_lon,)
    assert dataset.subset_topography is not None
    assert dataset.subset_topography.shape == (expected_n_lat, expected_n_lon)
    assert torch.equal(
        dataset.subset_topography, topography[:expected_n_lat, :expected_n_lon]
    )


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
    assert batched.topography is not None
    assert batched.topography.shape == (num_items, 8, 16)
    assert torch.equal(batched.topography[0], items[0].topography)

    # Test no topography
    data, times, lalo, _ = items[0]
    no_topo = BatchItem(data, times, lalo, None)
    items = [items[0], no_topo]
    batched = BatchData.from_sequence(items)
    assert batched.topography is None


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
    assert expanded.topography.shape == (30, 8, 16)


def get_mock_dataset(field_leading_dim=1):
    # Mock dataset that returns (data, time) tuples
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=2)
    dataset.__getitem__ = MagicMock(
        return_value=(
            {"x": torch.rand(field_leading_dim, 8, 16)},
            data_array([0]),
        )
    )
    return dataset


def test_batch_item_dataset_adapter():
    dataset = get_mock_dataset()

    # Create adapter with example coordinates
    coords = get_example_latlon_coordinates()
    topography = torch.rand(8, 16)
    adapter = BatchItemDatasetAdapter(dataset, coords, topography)

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
    assert torch.equal(item.topography, topography)


def test_batch_item_dataset_adapter_no_topography():
    dataset = get_mock_dataset()
    coords = get_example_latlon_coordinates()
    adapter = BatchItemDatasetAdapter(dataset, coords, topography=None)

    item = adapter[0]
    assert item.topography is None


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


def test_generate_random_extent_slice():
    # Test when extent is None
    slice_ = _generate_random_extent_slice(10, None)
    assert slice_ == slice(None)

    with pytest.raises(ValueError):
        _generate_random_extent_slice(10, 10)

    # Test when extent > dim_len raises ValueError
    with pytest.raises(ValueError):
        _generate_random_extent_slice(5, 10)

    # Test multiple calls give different results
    slices = [_generate_random_extent_slice(10, 5) for _ in range(5)]
    starts = [s.start for s in slices]
    # Check that at least two starts are different (random)
    assert len(set(starts)) > 1


@pytest.mark.parametrize(
    "input_slice,expected",
    [
        pytest.param(slice(None), slice(None), id="none"),
        pytest.param(slice(None, 5), slice(None, 10), id="start_none"),
        pytest.param(slice(3, None), slice(6, None), id="stop_none"),
        pytest.param(slice(2, 4), slice(4, 8), id="both"),
    ],
)
def test_scale_slice(input_slice, expected):
    scaled = _scale_slice(input_slice, scale=2)
    assert scaled.start == expected.start
    assert scaled.stop == expected.stop


def test__subset_horizontal_component_consistency():
    # Create example batch item with known dimensions and easy to check data
    n_lat, n_lon = 5, 10
    spatial = torch.arange(n_lon).repeat(n_lat, 1)
    lat = torch.arange(n_lat)
    lon = torch.arange(n_lon)
    data = {"x": spatial}
    topography = spatial

    lat_ext = (1, 3)
    lon_ext = (2, 5)

    item = BatchItem(data, xr.DataArray(0), LatLonCoordinates(lat, lon), topography)
    lat_slice = slice(*lat_ext)
    lon_slice = slice(*lon_ext)
    result = _subset_horizontal(item, lat_slice, lon_slice)

    # check that subsetting applied equivalently to each BatchItem component
    expected_lat = torch.arange(*lat_ext)
    assert torch.equal(result.latlon_coordinates.lat, expected_lat)

    expected_lon = torch.arange(*lon_ext)
    assert torch.equal(result.latlon_coordinates.lon, expected_lon)

    expected_data = torch.arange(*lon_ext).repeat(2, 1)
    assert torch.equal(result.data["x"], expected_data)
    assert torch.equal(result.topography, expected_data)


@pytest.mark.parametrize(
    "coarse_lat_extent,coarse_lon_extent,expected_coarse_shape,expected_fine_shape",
    [
        pytest.param(None, None, (4, 8), (8, 16), id="no-extents"),
        pytest.param(2, 4, (2, 4), (4, 8), id="both-extents"),
        pytest.param(2, None, (2, 8), (4, 16), id="lat-extent-only"),
        pytest.param(None, 4, (4, 4), (8, 8), id="lon-extent-only"),
    ],
)
def test_subset_fine_coarse(
    coarse_lat_extent, coarse_lon_extent, expected_coarse_shape, expected_fine_shape
):
    # Create example fine and coarse batch items with known dimensions
    fine_item = get_batch_items(num_items=1, lat_dim=8, lon_dim=16)[0]
    coarse_item = get_batch_items(num_items=1, lat_dim=4, lon_dim=8)[0]
    coarse_item.topography = None

    paired = PairedBatchItem(fine_item, coarse_item)
    scale = 2

    # Test subsetting
    subset = _subset_fine_coarse(paired, scale, coarse_lat_extent, coarse_lon_extent)

    # Check dimensions of subset data
    assert subset.coarse.data["x"].shape == expected_coarse_shape
    assert subset.fine.data["x"].shape == expected_fine_shape

    # Check lat/lon coordinates were properly subset
    assert len(subset.coarse.latlon_coordinates.lat) == expected_coarse_shape[0]
    assert len(subset.coarse.latlon_coordinates.lon) == expected_coarse_shape[1]
    assert len(subset.fine.latlon_coordinates.lat) == expected_fine_shape[0]
    assert len(subset.fine.latlon_coordinates.lon) == expected_fine_shape[1]

    # Check topography was properly subset
    assert subset.coarse.topography is None
    assert subset.fine.topography is not None
    assert subset.fine.topography.shape == expected_fine_shape

    # Check time is preserved
    assert subset.coarse.time == paired.coarse.time
    assert subset.fine.time == paired.fine.time


def test_random_spatial_subset_paired_dataset():
    # Create example fine and coarse batch items with known dimensions
    # Fine is 2x scale
    fine_item = get_batch_items(num_items=1, lat_dim=8, lon_dim=16)[0]
    coarse_item = get_batch_items(num_items=1, lat_dim=4, lon_dim=8)[0]

    # Mock dataset that returns paired items
    dataset = MagicMock(spec=FineCoarsePairedDataset)
    dataset.__len__ = MagicMock(return_value=5)
    dataset.__getitem__ = MagicMock(
        return_value=PairedBatchItem(fine_item, coarse_item)
    )

    # Test successful case with valid scale factor
    subset_dataset = RandomSpatialSubsetPairedDataset(
        dataset, coarse_lat_extent=2, coarse_lon_extent=4
    )
    assert len(subset_dataset) == 5
    subset_dataset[0]

    # Test with mismatched aspect ratio
    fine_item_bad = BatchItem(
        random_named_tensor(["x"], (2, 8)),  # Square shape
        fine_item.time,
        get_example_latlon_coordinates(2, 8),
        torch.rand(2, 8),
    )

    dataset_bad = MagicMock(spec=FineCoarsePairedDataset)
    dataset_bad.__getitem__ = MagicMock(
        return_value=PairedBatchItem(fine_item_bad, coarse_item)
    )

    with pytest.raises(ValueError, match="Aspect ratio must match between lat and lon"):
        RandomSpatialSubsetPairedDataset(dataset_bad)


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
    assert torch.equal(
        batch_slice.topography,
        batch.topography[:, lat_slice, lon_slice],  # type: ignore
    )
