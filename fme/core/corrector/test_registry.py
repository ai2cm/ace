import datetime
from collections.abc import Collection

import pytest
import torch

from fme.core.coordinates import NullVerticalCoordinate
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.ice import IceCorrectorConfig
from fme.core.corrector.ocean import OceanCorrectorConfig
from fme.core.corrector.output import CorrectorOutput
from fme.core.corrector.registry import (
    CorrectionSequence,
    CorrectorABC,
    CorrectorConfigABC,
    EpochScheduledCorrector,
)
from fme.core.corrector.state import CorrectorState
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import LatLonOperations
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorMapping


def _get_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        vertical_coordinate=NullVerticalCoordinate(),
        gridded_operations=LatLonOperations(area_weights=torch.ones(2, 2)),
        img_shape=(2, 2),
        timestep=datetime.timedelta(hours=6),
    )


def _get_corrector(
    config: CorrectorConfigABC,
    input_names: Collection[str] = (),
    gen_names: Collection[str] = (),
    forcing_names: Collection[str] = (),
) -> CorrectorABC:
    return config.get_corrector(
        _get_dataset_info(),
        input_names=input_names,
        gen_names=gen_names,
        forcing_names=forcing_names,
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
    corrector = _get_corrector(config)
    assert isinstance(corrector, EpochScheduledCorrector)


def test_corrector_not_wrapped_when_disabled_epochs_zero():
    corrector = _get_corrector(AtmosphereCorrectorConfig())
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
    corrector = _get_corrector(selector)
    assert isinstance(corrector, EpochScheduledCorrector)


def test_scheduled_corrector_requires_state_when_disabled_epochs_configured():
    corrector = _get_corrector(AtmosphereCorrectorConfig(corrector_disabled_epochs=1))
    with pytest.raises(ValueError, match="corrector_disabled"):
        corrector.load_state({})


class _LifecycleRecordingCorrector(CorrectorABC):
    def __init__(self):
        self.train_modes: list[bool] = []
        self.epochs: list[int] = []
        self.loaded_state: dict[str, object] | None = None
        self.discovery_calls = 0

    @property
    def modified_names(self) -> frozenset[str]:
        return frozenset({"wrapped_delta"})

    def discover_modified_names(
        self,
        input_names: Collection[str],
        gen_names: Collection[str],
        forcing_names: Collection[str],
        img_shape: tuple[int, int],
    ) -> None:
        self.discovery_calls += 1

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
    ) -> CorrectorOutput:
        return CorrectorOutput(
            corrected=dict(gen_data), corrector_state=corrector_state
        )


def test_scheduled_corrector_forwards_lifecycle_and_state():
    wrapped = _LifecycleRecordingCorrector()
    corrector = EpochScheduledCorrector(wrapped=wrapped, disabled_epochs=2)

    assert corrector.train(False) is corrector
    corrector.set_epoch(3)
    state = corrector.get_state()
    corrector.load_state(state)
    corrector.discover_modified_names([], [], [], (2, 2))

    assert wrapped.train_modes == [False]
    assert wrapped.epochs == [3]
    assert wrapped.discovery_calls == 1
    # forwarded independent of the epoch-disabled state
    assert corrector.modified_names == frozenset({"wrapped_delta"})
    assert state == {
        "corrector_disabled": False,
        "wrapped": {"wrapped_value": 3},
    }
    assert wrapped.loaded_state == {"wrapped_value": 3}


def test_modified_names_raises_before_discovery():
    # a CorrectionSequence built without going through get_corrector has never
    # run discovery, and says so rather than reporting "modifies nothing".
    corrector = CorrectionSequence([ConstantOffsetCorrection("a", 1.0)])
    with pytest.raises(RuntimeError, match="discovery has not run"):
        corrector.modified_names
    corrector.discover_modified_names(["a"], ["a"], [], (2, 2))
    assert corrector.modified_names == frozenset({"a"})


class ConstantOffsetCorrection:
    """Adds a constant offset to one named field, returning only that field."""

    def __init__(self, name: str, offset: float):
        self._name = name
        self._offset = offset

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[dict, CorrectorState | None]:
        return {self._name: gen_data[self._name] + self._offset}, corrector_state


def test_correction_sequence_accumulates_modified_keys():
    gen_data = {
        "a": torch.zeros(2, 2),
        "b": torch.zeros(2, 2),
        "c": torch.zeros(2, 2),
    }
    sequence = CorrectionSequence(
        [
            ConstantOffsetCorrection("a", 1.0),
            ConstantOffsetCorrection("b", 2.0),
        ]
    )
    result = sequence({}, gen_data, {}, None)
    # corrected carries every field; delta only the modified ones
    assert set(result.corrected) == {"a", "b", "c"}
    assert set(result.diagnostics.delta) == {"a", "b"}
    assert set(result.modified_names) == {"a", "b"}
    torch.testing.assert_close(result.diagnostics.delta["a"], torch.full((2, 2), 1.0))
    torch.testing.assert_close(result.diagnostics.delta["b"], torch.full((2, 2), 2.0))
    assert "c" not in result.diagnostics.delta


def test_correction_sequence_unions_repeated_modified_key():
    # Two corrections write the same field: the delta is the net change and the
    # key appears once.
    gen_data = {"a": torch.zeros(2, 2)}
    sequence = CorrectionSequence(
        [
            ConstantOffsetCorrection("a", 1.0),
            ConstantOffsetCorrection("a", 3.0),
        ]
    )
    result = sequence({}, gen_data, {}, None)
    assert set(result.diagnostics.delta) == {"a"}
    torch.testing.assert_close(result.diagnostics.delta["a"], torch.full((2, 2), 4.0))


def test_epoch_scheduled_corrector_disabled_returns_empty_diagnostics():
    wrapped = CorrectionSequence([ConstantOffsetCorrection("a", 1.0)])
    corrector = EpochScheduledCorrector(wrapped=wrapped, disabled_epochs=1)
    corrector.train(True)  # disabled for train-mode steps in the first epoch
    gen_data = {"a": torch.zeros(2, 2)}

    disabled = corrector({}, gen_data, {}, None)
    # nothing applied: passthrough corrected, empty delta
    torch.testing.assert_close(disabled.corrected["a"], gen_data["a"])
    assert dict(disabled.diagnostics.delta) == {}
    assert set(disabled.modified_names) == set()

    corrector.eval()  # always applied in eval mode -> delegates to wrapped
    enabled = corrector({}, gen_data, {}, None)
    assert set(enabled.modified_names) == {"a"}
    torch.testing.assert_close(enabled.diagnostics.delta["a"], torch.full((2, 2), 1.0))


def test_construction_records_modified_names():
    config = OceanCorrectorConfig(force_positive_names=["a", "b"])
    corrector = _get_corrector(config, gen_names=["a", "b", "c"])
    gen_data = {
        "a": torch.zeros(2, 2),
        "b": torch.zeros(2, 2),
        "c": torch.zeros(2, 2),
    }
    result = corrector({}, gen_data, {}, None)
    # construction records exactly the delta keys a real call produces
    assert corrector.modified_names == frozenset(result.diagnostics.delta)
    assert corrector.modified_names == frozenset({"a", "b"})


def test_discovery_through_epoch_schedule():
    config = OceanCorrectorConfig(
        force_positive_names=["a"], corrector_disabled_epochs=1
    )
    corrector = _get_corrector(config, gen_names=["a", "b"])
    assert isinstance(corrector, EpochScheduledCorrector)
    corrector.train(True)  # disabled for train-mode steps in the first epoch
    # modified_names describes what the corrector produces when active, so it is
    # independent of the epoch-disabled state
    assert corrector.modified_names == frozenset({"a"})
    corrector.set_epoch(2)  # enabled
    assert corrector.modified_names == frozenset({"a"})


def test_modified_names_empty_without_corrections():
    corrector = _get_corrector(AtmosphereCorrectorConfig(), gen_names=["a"])
    assert corrector.modified_names == frozenset()
