import datetime

import pytest
import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset_info import DatasetInfo
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
