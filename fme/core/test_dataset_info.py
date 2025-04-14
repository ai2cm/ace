import datetime
from typing import List

import pytest
import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset_info import DatasetInfo, IncompatibleDatasetInfo
from fme.core.gridded_ops import LatLonOperations


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
            DatasetInfo(),
            DatasetInfo(timestep=datetime.timedelta(hours=1)),
            id="timestep_missing_from_first",
        ),  # a model trained with no timestep can work with any timestep
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
    a: DatasetInfo, b: DatasetInfo, msgs: List[str]
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
