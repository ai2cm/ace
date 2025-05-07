import datetime
import logging

import pytest
import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import (
    DatasetInfo,
    IncompatibleDatasetInfo,
    get_keys_with_conflicts,
)
from fme.core.gridded_ops import LatLonOperations
from fme.core.mask_provider import MaskProvider


@pytest.mark.parametrize(
    "dataset_info",
    [
        pytest.param(
            DatasetInfo(),
            id="empty",
        ),
        pytest.param(
            DatasetInfo(
                img_shape=(10, 10),
                gridded_operations=LatLonOperations(area_weights=torch.ones(10, 10)),
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
                img_shape=(10, 10),
                gridded_operations=LatLonOperations(area_weights=torch.ones(10, 10)),
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
            DatasetInfo(img_shape=(10, 10)),
            DatasetInfo(img_shape=(10, 10)),
            id="img_shape_equal",
        ),
        pytest.param(
            DatasetInfo(
                gridded_operations=LatLonOperations(area_weights=torch.ones(10, 10))
            ),
            DatasetInfo(
                gridded_operations=LatLonOperations(area_weights=torch.ones(10, 10))
            ),
            id="gridded_operations_equal",
        ),
        pytest.param(
            DatasetInfo(),
            DatasetInfo(
                gridded_operations=LatLonOperations(area_weights=torch.ones(10, 10))
            ),
            id="gridded_operations_missing_from_first",
        ),  # model trained without GriddedOperations doesn't care what inference uses
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
            DatasetInfo(img_shape=(10, 10)),
            DatasetInfo(img_shape=(10, 11)),
            ["img_shape"],
            id="img_shape",
        ),
        pytest.param(
            DatasetInfo(),
            DatasetInfo(img_shape=(10, 10)),
            ["img_shape"],
            id="img_shape_missing_from_first",
        ),
        pytest.param(
            DatasetInfo(img_shape=(10, 10)),
            DatasetInfo(),
            ["img_shape"],
            id="img_shape_missing_from_second",
        ),
        pytest.param(
            DatasetInfo(
                gridded_operations=LatLonOperations(area_weights=torch.ones(10, 10))
            ),
            DatasetInfo(
                gridded_operations=LatLonOperations(area_weights=torch.zeros(10, 10))
            ),
            ["gridded_operations"],
            id="gridded_operations_values_differ",
        ),
        pytest.param(
            DatasetInfo(
                gridded_operations=LatLonOperations(area_weights=torch.ones(10, 10))
            ),
            DatasetInfo(),
            ["gridded_operations"],
            id="gridded_operations_missing_from_second",
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
                gridded_operations=LatLonOperations(area_weights=torch.ones(10, 10)),
                timestep=datetime.timedelta(hours=1),
            ),
            DatasetInfo(
                gridded_operations=LatLonOperations(area_weights=torch.zeros(10, 10)),
                timestep=datetime.timedelta(hours=2),
            ),
            ["gridded_operations", "timestep"],
            id="multiple_values_differ",
        ),
        pytest.param(
            DatasetInfo(
                gridded_operations=LatLonOperations(area_weights=torch.ones(10, 10)),
                timestep=datetime.timedelta(hours=1),
            ),
            DatasetInfo(),
            ["gridded_operations", "timestep"],
            id="multiple_values_missing_from_second",
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
        ["gridded_operations", "vertical_coordinate", "timestep", "img_shape"]
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
