import datetime
import logging
from typing import Any

import pytest
import torch

from fme.core.coordinates import (
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
from fme.core.gridded_ops import LatLonOperations
from fme.core.mask_provider import MaskProvider
from fme.core.metrics import spherical_area_weights


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
                    lat=torch.arange(-4, 4), lon=torch.arange(16)
                ),
                vertical_coordinate=HybridSigmaPressureCoordinate(
                    ak=torch.arange(10),
                    bk=torch.arange(10),
                ),
                timestep=datetime.timedelta(hours=1),
            ),
            id="vertical_coordinate",
        ),
        pytest.param(
            DatasetInfo(
                horizontal_coordinates=LatLonCoordinates(
                    lat=torch.arange(-4, 4), lon=torch.arange(16)
                ),
                mask_provider=MaskProvider(masks={"mask_0": torch.ones(10, 10)}),
                timestep=datetime.timedelta(hours=1),
            ),
            id="mask_provider",
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
    state = dataset_info.to_state()
    dataset_info_reloaded = DatasetInfo.from_state(state)
    assert dataset_info == dataset_info_reloaded


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
                mask_provider=MaskProvider(masks={"mask_0": torch.ones(10, 10)})
            ),
            DatasetInfo(
                mask_provider=MaskProvider(masks={"mask_0": torch.ones(10, 10)})
            ),
            id="mask_provider_equal",
        ),
        pytest.param(
            DatasetInfo(mask_provider=MaskProvider()),
            DatasetInfo(),
            id="mask_provider_missing_from_second",
        ),
        pytest.param(
            DatasetInfo(),
            DatasetInfo(mask_provider=MaskProvider()),
            id="mask_provider_missing_from_first",
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
                mask_provider=MaskProvider(masks={"mask_0": torch.ones(10, 10)})
            ),
            DatasetInfo(
                mask_provider=MaskProvider(masks={"mask_0": torch.zeros(10, 10)})
            ),
            ["mask_provider"],
            id="mask_provider_masks_differ",
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
    mask_provider = MaskProvider(masks={"mask_0": torch.ones(10, 10)})
    dataset_info = DatasetInfo(
        horizontal_coordinates=coords,
        mask_provider=mask_provider,
    )
    assert dataset_info.horizontal_coordinates == coords
    assert dataset_info.mask_provider == mask_provider
    expected_gridded_ops = LatLonOperations(
        area_weights=spherical_area_weights(lat, len(lon)),
        mask_provider=mask_provider,
    )
    assert dataset_info.gridded_operations == expected_gridded_ops
