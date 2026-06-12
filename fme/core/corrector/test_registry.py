import datetime

import pytest
import torch

from fme.core.coordinates import NullVerticalCoordinate
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.ice import IceCorrectorConfig
from fme.core.corrector.ocean import OceanCorrectorConfig
from fme.core.corrector.registry import EpochScheduledCorrector
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import LatLonOperations


def _get_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        vertical_coordinate=NullVerticalCoordinate(),
        gridded_operations=LatLonOperations(area_weights=torch.ones(2, 2)),
        timestep=datetime.timedelta(hours=6),
    )


def test_corrector_disabled_epochs_must_be_non_negative():
    with pytest.raises(ValueError, match="corrector_disabled_epochs"):
        AtmosphereCorrectorConfig(corrector_disabled_epochs=-1)


@pytest.mark.parametrize(
    "config",
    [
        AtmosphereCorrectorConfig(corrector_disabled_epochs=1),
        OceanCorrectorConfig(corrector_disabled_epochs=1),
        IceCorrectorConfig(corrector_disabled_epochs=1),
    ],
)
def test_corrector_configs_support_disabled_epochs(config):
    corrector = config.get_corrector(_get_dataset_info())

    assert isinstance(corrector, EpochScheduledCorrector)


def test_scheduled_corrector_requires_state_when_disabled_epochs_configured():
    corrector = AtmosphereCorrectorConfig(corrector_disabled_epochs=1).get_corrector(
        _get_dataset_info()
    )

    with pytest.raises(ValueError, match="corrector_disabled"):
        corrector.load_state({})
