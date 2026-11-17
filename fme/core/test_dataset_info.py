import datetime
import logging
from typing import Any

import pytest
import torch

from fme.core.area_weights_provider import (
    AreaWeightsProvider,
    AreaWeightsProviderABC,
    NullAreaWeightsProvider,
)
from fme.core.coordinates import (
    DepthCoordinate,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
    NullVerticalCoordinate,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import (
    DatasetInfo,
    IncompatibleDatasetInfo,
    MissingDatasetInfo,
    get_keys_with_conflicts,
)
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.metrics import spherical_area_weights
from fme.core.spatial_mask_provider import SpatialMaskProvider


def _make_depth_coordinate_with_ocean_volume_per_area() -> DepthCoordinate:
    mask = torch.ones(2, 3, 2, device=get_device())
    mask[0, 0, 1] = 0.0
    return DepthCoordinate(
        idepth=torch.tensor([0.0, 10.0, 50.0], device=get_device()),
        mask=mask,
        deptho=torch.full((2, 3), 50.0, device=get_device()),
        ocean_volume_per_area=torch.full((2, 3, 2), 20.0, device=get_device()),
    )


@pytest.mark.parametrize(
    "dataset_info",
    [
        pytest.param(
            DatasetInfo(),
            id="empty",
        ),
        pytest.param(
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-4, 4, device=get_device()),
                    lon=torch.arange(16, device=get_device()),
                ),
                vertical_coordinate=HybridSigmaPressureCoordinate(
                    ak=torch.arange(10, device=get_device()),
                    bk=torch.arange(10, device=get_device()),
                ),
                timestep=datetime.timedelta(hours=1),
            ),
            id="vertical_coordinate",
        ),
        pytest.param(
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-4, 4, device=get_device()),
                    lon=torch.arange(16, device=get_device()),
                ),
                spatial_mask_provider=SpatialMaskProvider(
                    masks={"mask_0": torch.ones(10, 10, device=get_device())}
                ),
                timestep=datetime.timedelta(hours=1),
            ),
            id="spatial_mask_provider",
        ),
        pytest.param(
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(2, device=get_device()),
                    lon=torch.arange(3, device=get_device()),
                ),
                vertical_coordinate=_make_depth_coordinate_with_ocean_volume_per_area(),
                timestep=datetime.timedelta(days=5),
            ),
            id="depth_coordinate_with_ocean_volume_per_area",
        ),
        pytest.param(
            DatasetInfo(
                variable_metadata={
                    "var_0": VariableMetadata(
                        "m",
                        "Variable 0",
                    ),
                },
            ),
            id="variable_metadata",
        ),
    ],
)
def test_dataset_info_round_trip(dataset_info: DatasetInfo):
    state = dataset_info.get_state()
    dataset_info_reloaded = DatasetInfo.from_state(state)
    assert dataset_info == dataset_info_reloaded


def test_dataset_info_round_trip_keeps_ocean_volume_per_area_dz():
    dataset_info = DatasetInfo(
        vertical_coordinate=_make_depth_coordinate_with_ocean_volume_per_area()
    )
    reloaded = DatasetInfo.from_state(dataset_info.get_state())
    coordinate = reloaded.vertical_coordinate
    assert isinstance(coordinate, DepthCoordinate)
    assert coordinate.ocean_volume_per_area is not None
    torch.testing.assert_close(
        coordinate.dz,
        _make_depth_coordinate_with_ocean_volume_per_area().dz,
    )


def test_dataset_info_from_state_without_ocean_volume_per_area():
    old_coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, 10.0, 50.0], device=get_device()),
        mask=torch.ones(2, 3, 2, device=get_device()),
        deptho=torch.full((2, 3), 30.0, device=get_device()),
    )
    state = DatasetInfo(vertical_coordinate=old_coordinate).get_state()
    assert "ocean_volume_per_area" not in state["vertical_coordinate"]
    reloaded = DatasetInfo.from_state(state)
    coordinate = reloaded.vertical_coordinate
    assert isinstance(coordinate, DepthCoordinate)
    assert coordinate.ocean_volume_per_area is None
    assert coordinate == old_coordinate
    torch.testing.assert_close(coordinate.dz, old_coordinate.dz)


@pytest.mark.parametrize(
    "a, b",
    [
        pytest.param(
            DatasetInfo(),
            DatasetInfo(),
            id="empty",
        ),
        pytest.param(
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-4, 4),
                    lon=torch.arange(16),
                )
            ),
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-4, 4),
                    lon=torch.arange(16),
                )
            ),
            id="horizontal_coordinates_equal",
        ),
        pytest.param(
            DatasetInfo(),
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-4, 4),
                    lon=torch.arange(16),
                )
            ),
            id="horizontal_coordinates_missing_from_first",
        ),  # model without HorizontalCoordinates doesn't care what inference uses
        pytest.param(
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-4, 4),
                    lon=torch.arange(16),
                )
            ),
            DatasetInfo(),
            id="horizontal_coordinates_missing_from_second",
        ),
        pytest.param(
            DatasetInfo(
                vertical_coordinate=HybridSigmaPressureCoordinate(
                    ak=torch.arange(7), bk=torch.arange(7)
                )
            ),
            DatasetInfo(
                vertical_coordinate=HybridSigmaPressureCoordinate(
                    ak=torch.arange(7), bk=torch.arange(7)
                )
            ),
            id="vertical_coordinate_equal",
        ),
        pytest.param(
            DatasetInfo(
                vertical_coordinate=HybridSigmaPressureCoordinate(
                    ak=torch.arange(7), bk=torch.arange(7)
                )
            ),
            DatasetInfo(),
            id="vertical_coordinate_missing_from_second",
        ),
        pytest.param(
            DatasetInfo(),
            DatasetInfo(
                vertical_coordinate=HybridSigmaPressureCoordinate(
                    ak=torch.arange(7), bk=torch.arange(7)
                )
            ),
            id="vertical_coordinate_missing_from_first",
        ),
        pytest.param(
            DatasetInfo(
                vertical_coordinate=NullVerticalCoordinate(),
            ),
            DatasetInfo(
                vertical_coordinate=HybridSigmaPressureCoordinate(
                    ak=torch.arange(7), bk=torch.arange(7)
                )
            ),
            id="null_vertical_coordinate_first",
        ),
        pytest.param(
            DatasetInfo(
                spatial_mask_provider=SpatialMaskProvider(
                    masks={"mask_0": torch.ones(10, 10)}
                )
            ),
            DatasetInfo(
                spatial_mask_provider=SpatialMaskProvider(
                    masks={"mask_0": torch.ones(10, 10)}
                )
            ),
            id="spatial_mask_provider_equal",
        ),
        pytest.param(
            DatasetInfo(
                spatial_mask_provider=SpatialMaskProvider(
                    masks={"mask_0": torch.ones(10, 10)}
                )
            ),
            DatasetInfo(
                spatial_mask_provider=SpatialMaskProvider(
                    masks={
                        "mask_0": torch.ones(10, 10),
                        "mask_1": torch.ones(10, 10),
                    }
                )
            ),
            id="spatial_mask_provider_subset",
        ),
        pytest.param(
            DatasetInfo(spatial_mask_provider=SpatialMaskProvider()),
            DatasetInfo(),
            id="spatial_mask_provider_missing_from_second",
        ),
        pytest.param(
            DatasetInfo(),
            DatasetInfo(spatial_mask_provider=SpatialMaskProvider()),
            id="spatial_mask_provider_missing_from_first",
        ),
        pytest.param(
            DatasetInfo(),
            DatasetInfo(timestep=datetime.timedelta(hours=1)),
            id="timestep_missing_from_first",
        ),  # a model trained with no timestep can work with any timestep
        pytest.param(
            DatasetInfo(),
            DatasetInfo(
                variable_metadata={
                    "var_0": VariableMetadata(
                        "m",
                        "Variable 0",
                    )
                }
            ),
            id="variable_metadata_missing_from_first",
        ),  # different variable metadata is allowed
    ],
)
def test_assert_compatible_with_compatible_dataset_info(a: DatasetInfo, b: DatasetInfo):
    a.assert_compatible_with(b)


@pytest.mark.parametrize(
    "a, b, msgs",
    [
        pytest.param(
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-4, 4),
                    lon=torch.arange(16),
                )
            ),
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-5, 3),
                    lon=torch.arange(16),
                )
            ),
            ["horizontal_coordinates"],
            id="horizontal_coordinates_values_differ",
        ),
        pytest.param(
            DatasetInfo(
                vertical_coordinate=HybridSigmaPressureCoordinate(
                    ak=torch.arange(7), bk=torch.arange(7)
                )
            ),
            DatasetInfo(
                vertical_coordinate=HybridSigmaPressureCoordinate(
                    ak=torch.arange(10), bk=torch.arange(10)
                )
            ),
            ["vertical_coordinate"],
            id="vertical_coordinate_values_differ",
        ),
        pytest.param(
            DatasetInfo(
                spatial_mask_provider=SpatialMaskProvider(
                    masks={"mask_0": torch.ones(10, 10)}
                )
            ),
            DatasetInfo(
                spatial_mask_provider=SpatialMaskProvider(
                    masks={"mask_0": torch.zeros(10, 10)}
                )
            ),
            ["spatial_mask_provider"],
            id="spatial_mask_provider_masks_differ",
        ),
        pytest.param(
            DatasetInfo(
                spatial_mask_provider=SpatialMaskProvider(
                    masks={
                        "mask_0": torch.ones(10, 10),
                        "mask_1": torch.ones(10, 10),
                    }
                )
            ),
            DatasetInfo(
                spatial_mask_provider=SpatialMaskProvider(
                    masks={"mask_0": torch.zeros(10, 10)}
                )
            ),
            ["spatial_mask_provider"],
            id="spatial_mask_provider_not_subset",
        ),
        pytest.param(
            DatasetInfo(timestep=datetime.timedelta(hours=1)),
            DatasetInfo(timestep=datetime.timedelta(hours=2)),
            ["timestep"],
            id="timestep_values_differ",
        ),
        pytest.param(
            DatasetInfo(timestep=datetime.timedelta(hours=1)),
            DatasetInfo(),
            ["timestep"],
            id="timestep_missing_from_second",
        ),  # a model trained with a timestep cannot work with arbitrary timesteps
        pytest.param(
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-4, 4),
                    lon=torch.arange(16),
                ),
                timestep=datetime.timedelta(hours=1),
            ),
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-5, 3),
                    lon=torch.arange(16),
                ),
                timestep=datetime.timedelta(hours=2),
            ),
            ["horizontal_coordinates", "timestep"],
            id="multiple_values_differ",
        ),
    ],
)
def test_assert_compatible_with_incompatible_dataset_info(
    a: DatasetInfo, b: DatasetInfo, msgs: list[str]
):
    with pytest.raises(IncompatibleDatasetInfo) as exc_info:
        a.assert_compatible_with(b)
    for msg in msgs:
        assert msg in str(exc_info.value)
    non_messages = set(
        ["horizontal_coordinates", "vertical_coordinate", "timestep", "img_shape"]
    ).difference(msgs)
    for msg in non_messages:
        assert msg not in str(exc_info.value)


def test_get_keys_with_conflicts():
    dict_a = {
        "var_1": VariableMetadata(
            "m",
            "Variable 1",
        ),
    }
    dict_b = {
        "var_0": VariableMetadata(
            "m",
            "Variable 0",
        ),
        "var_1": VariableMetadata(
            "cm",
            "Variable 1",
        ),
    }
    keys_with_conflicts = get_keys_with_conflicts(dict_a, dict_b)
    assert len(keys_with_conflicts) == 1
    assert next(iter(keys_with_conflicts.keys())) == "var_1"
    assert keys_with_conflicts["var_1"] == (
        VariableMetadata("m", "Variable 1"),
        VariableMetadata("cm", "Variable 1"),
    )


def test_compatibility_logs_metadata_keys(caplog):
    dataset_info_a = DatasetInfo(
        variable_metadata={
            "var_1": VariableMetadata(
                "m",
                "Variable 1",
            ),
        }
    )
    dataset_info_b = DatasetInfo(
        variable_metadata={
            "var_1": VariableMetadata(
                "cm",
                "Variable 1's other name",
            ),
        }
    )
    with caplog.at_level(logging.WARNING):
        dataset_info_a.assert_compatible_with(dataset_info_b)
    assert (
        "DatasetInfo has different metadata from other DatasetInfo for key var_1"
        in caplog.text
    )


def test_backwards_compatibility_with_gridded_ops():
    """Previously DatasetInfo was initialized with img_shape and
    gridded_operations. Now we use horizontal_coordinates. This test ensures
    that old-style DatasetInfo can still be deserialized."""
    legacy_state: dict[str, Any] = {
        "img_shape": (4, 4),
        "gridded_operations": {
            "type": "LatLonOperations",
            "state": {"area_weights": torch.ones(4, 4)},
        },
    }
    dataset_info = DatasetInfo.from_state(legacy_state)
    with pytest.raises(MissingDatasetInfo, match="horizontal_coordinates"):
        dataset_info.horizontal_coordinates
    assert isinstance(dataset_info.gridded_operations, LatLonOperations)
    assert dataset_info.img_shape == (4, 4)


def test_dataset_info_raises_error_with_conflicting_inputs():
    coords = LatLonCoordinates(lat=torch.arange(-4, 4), lon=torch.arange(16))
    with pytest.raises(ValueError, match="provide both img_shape"):
        DatasetInfo(horizontal_coordinates=coords, img_shape=(10, 10))
    with pytest.raises(ValueError, match="provide both gridded_operations"):
        ops = LatLonOperations(area_weights=torch.ones(10, 10))
        DatasetInfo(horizontal_coordinates=coords, gridded_operations=ops)


def test_masked_gridded_ops():
    lon = torch.arange(4)
    lat = torch.arange(2)
    coords = LatLonCoordinates(lat=lat, lon=lon)
    spatial_mask_provider = SpatialMaskProvider(masks={"mask_0": torch.ones(10, 10)})
    dataset_info = DatasetInfo(
        horizontal_coordinates=coords,
        spatial_mask_provider=spatial_mask_provider,
    )
    assert dataset_info.horizontal_coordinates == coords
    assert dataset_info.spatial_mask_provider == spatial_mask_provider
    expected_gridded_ops = LatLonOperations(
        area_weights=spherical_area_weights(lat, len(lon)),
        spatial_mask_provider=spatial_mask_provider,
    )
    assert dataset_info.gridded_operations == expected_gridded_ops


def _make_hybrid_sigma_pressure_coordinate():
    return HybridSigmaPressureCoordinate(ak=torch.arange(10), bk=torch.arange(10))


def _make_depth_coordinate():
    return DepthCoordinate(
        idepth=torch.arange(4, dtype=torch.float),
        mask=torch.ones(3),
    )


class TestAtmosphereVerticalCoordinate:
    def test_returns_coordinate_for_atmosphere(self):
        vc = _make_hybrid_sigma_pressure_coordinate()
        info = DatasetInfo(vertical_coordinate=vc)
        assert info.atmosphere_vertical_coordinate is vc

    def test_returns_none_for_null(self):
        info = DatasetInfo(vertical_coordinate=NullVerticalCoordinate())
        assert info.atmosphere_vertical_coordinate is None

    def test_raises_for_ocean_coordinate(self):
        info = DatasetInfo(vertical_coordinate=_make_depth_coordinate())
        with pytest.raises(RuntimeError, match="atmosphere vertical coordinate"):
            info.atmosphere_vertical_coordinate


class TestOceanVerticalCoordinate:
    def test_returns_coordinate_for_ocean(self):
        vc = _make_depth_coordinate()
        info = DatasetInfo(vertical_coordinate=vc)
        assert info.ocean_vertical_coordinate is vc

    def test_returns_none_for_null(self):
        info = DatasetInfo(vertical_coordinate=NullVerticalCoordinate())
        assert info.ocean_vertical_coordinate is None

    def test_raises_for_atmosphere_coordinate(self):
        info = DatasetInfo(vertical_coordinate=_make_hybrid_sigma_pressure_coordinate())
        with pytest.raises(RuntimeError, match="ocean vertical coordinate"):
            info.ocean_vertical_coordinate


def _area_weights_dataset_info(
    area_weights_provider: AreaWeightsProviderABC | None,
) -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.linspace(-60.0, 60.0, 4, device=get_device()),
            lon=torch.arange(8, device=get_device()) * 45.0,
        ),
        timestep=datetime.timedelta(hours=1),
        area_weights_provider=area_weights_provider,
    )


def _area_weights_provider(seed: int = 0) -> AreaWeightsProvider:
    torch.manual_seed(seed)
    return AreaWeightsProvider(torch.rand(4, 8, device=get_device()))


def test_dataset_info_area_weights_provider_round_trip():
    dataset_info = _area_weights_dataset_info(_area_weights_provider())
    reloaded = DatasetInfo.from_state(dataset_info.get_state())
    assert reloaded == dataset_info
    assert reloaded != _area_weights_dataset_info(None)
    x = torch.randn(2, 4, 8, device=get_device())
    torch.testing.assert_close(
        reloaded.gridded_operations.area_weighted_mean(x),
        dataset_info.gridded_operations.area_weighted_mean(x),
    )


def test_dataset_info_state_without_area_weights_provider_loads_unchanged():
    dataset_info = _area_weights_dataset_info(None)
    state = dataset_info.get_state()
    assert state["area_weights_provider"] is None
    del state["area_weights_provider"]  # as in checkpoints from before the key
    reloaded = DatasetInfo.from_state(state)
    assert reloaded == dataset_info
    assert reloaded.area_weights_provider is NullAreaWeightsProvider
    x = torch.randn(2, 4, 8, device=get_device())
    latitude_ops = reloaded.horizontal_coordinates.get_gridded_operations()
    torch.testing.assert_close(
        reloaded.gridded_operations.area_weighted_mean(x),
        latitude_ops.area_weighted_mean(x),
    )


def test_dataset_info_null_area_weights_provider_stored_as_none():
    assert _area_weights_dataset_info(
        NullAreaWeightsProvider
    ) == _area_weights_dataset_info(None)


def test_get_gridded_operations_without_area_weights_provider():
    dataset_info = _area_weights_dataset_info(_area_weights_provider())
    x = torch.randn(2, 4, 8, device=get_device())
    latitude_ops = dataset_info.horizontal_coordinates.get_gridded_operations()
    latitude_mean = latitude_ops.area_weighted_mean(x)
    torch.testing.assert_close(
        dataset_info.get_gridded_operations(
            use_area_weights_provider=False
        ).area_weighted_mean(x),
        latitude_mean,
    )
    assert not torch.allclose(
        dataset_info.gridded_operations.area_weighted_mean(x), latitude_mean
    )


@pytest.mark.parametrize(
    "a, b, compatible",
    [
        pytest.param(_area_weights_provider(0), _area_weights_provider(0), True),
        pytest.param(_area_weights_provider(0), None, True),
        pytest.param(None, _area_weights_provider(0), True),
        pytest.param(_area_weights_provider(0), _area_weights_provider(1), False),
    ],
)
def test_assert_compatible_with_area_weights_provider(a, b, compatible):
    info_a, info_b = _area_weights_dataset_info(a), _area_weights_dataset_info(b)
    if compatible:
        info_a.assert_compatible_with(info_b)
    else:
        with pytest.raises(IncompatibleDatasetInfo, match="area_weights_provider"):
            info_a.assert_compatible_with(info_b)
