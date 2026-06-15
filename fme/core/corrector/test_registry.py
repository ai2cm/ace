import datetime

import pytest
import torch

from fme.core.coordinates import NullVerticalCoordinate
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.ice import IceCorrectorConfig
from fme.core.corrector.ocean import OceanCorrectorConfig
from fme.core.corrector.registry import CorrectorABC, EpochScheduledCorrector
from fme.core.corrector.state import CorrectorState
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import LatLonOperations
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorDict, TensorMapping


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
def test_corrector_configs_wrap_when_disabled_epochs_set(config):
    corrector = config.get_corrector(_get_dataset_info())
    assert isinstance(corrector, EpochScheduledCorrector)


def test_corrector_not_wrapped_when_disabled_epochs_zero():
    corrector = AtmosphereCorrectorConfig().get_corrector(_get_dataset_info())
    assert not isinstance(corrector, EpochScheduledCorrector)
    # the bare corrector inherits the base no-op lifecycle methods
    assert corrector.train(False) is corrector
    corrector.set_epoch(5)  # no-op, must not raise
    assert corrector.get_state() == {}
    corrector.load_state({"ignored": 1})  # no-op, must not raise


def test_corrector_selector_rejects_disabled_epochs():
    # corrector_disabled_epochs must be set on the wrapped corrector config,
    # not on the selector, to avoid two places that could schedule.
    with pytest.raises(ValueError, match="not on the CorrectorSelector"):
        CorrectorSelector(
            type="atmosphere_corrector",
            config={},
            corrector_disabled_epochs=1,
        )


def test_corrector_selector_disabled_epochs_set_on_wrapped_config():
    selector = CorrectorSelector(
        type="atmosphere_corrector",
        config={"corrector_disabled_epochs": 1},
    )
    corrector = selector.get_corrector(_get_dataset_info())
    assert isinstance(corrector, EpochScheduledCorrector)


def test_scheduled_corrector_requires_state_when_disabled_epochs_configured():
    corrector = AtmosphereCorrectorConfig(corrector_disabled_epochs=1).get_corrector(
        _get_dataset_info()
    )
    with pytest.raises(ValueError, match="corrector_disabled"):
        corrector.load_state({})


class _LifecycleRecordingCorrector(CorrectorABC):
    def __init__(self):
        self.train_modes: list[bool] = []
        self.epochs: list[int] = []
        self.loaded_state: dict[str, object] | None = None

    def train(self, mode: bool = True) -> "_LifecycleRecordingCorrector":
        self.train_modes.append(mode)
        return self

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)

    def get_state(self) -> dict[str, object]:
        return {"wrapped_value": 3}

    def load_state(self, state: dict[str, object]) -> None:
        self.loaded_state = state

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        return dict(gen_data), corrector_state


def test_scheduled_corrector_forwards_lifecycle_and_state():
    wrapped = _LifecycleRecordingCorrector()
    corrector = EpochScheduledCorrector(wrapped=wrapped, disabled_epochs=2)

    assert corrector.train(False) is corrector
    corrector.set_epoch(3)
    state = corrector.get_state()
    corrector.load_state(state)

    assert wrapped.train_modes == [False]
    assert wrapped.epochs == [3]
    assert state == {
        "corrector_disabled": False,
        "wrapped": {"wrapped_value": 3},
    }
    assert wrapped.loaded_state == {"wrapped_value": 3}
