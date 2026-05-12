from unittest.mock import MagicMock, Mock

import pytest

from fme.ace.data_loading.inference import InferenceInitialConditionIndices
from fme.ace.stepper.time_length_probabilities import (
    TimeLengthProbabilities,
    TimeLengthProbability,
)
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.loss import StepLossConfig
from fme.core.typing_ import Slice
from fme.coupled.aggregator import InferenceEvaluatorAggregatorConfig
from fme.coupled.data_loading.config import CoupledDatasetWithOptionalOceanConfig
from fme.coupled.data_loading.inference import InferenceDataLoaderConfig
from fme.coupled.stepper import ComponentTrainingConfig
from fme.coupled.train.train_config import InlineInferenceConfig, TrainConfig
from fme.coupled.typing_ import CoupledOptionalInt

from .train_config import _validate_n_steps

_MOCK_LOSS = Mock(spec=StepLossConfig)


def _make_stepper_mock():
    stepper = MagicMock()
    stepper.n_inner_steps = 1
    return stepper


def _make_stepper_training_mock():
    stepper_training = MagicMock()
    stepper_training.n_coupled_steps = 1
    stepper_training.component_n_steps_max = CoupledOptionalInt(
        ocean=None, atmosphere=None
    )
    return stepper_training


def _make_inference_config(
    name: str | None = None, weight: float = 1.0, epochs: Slice | None = None
) -> InlineInferenceConfig:
    dataset = CoupledDatasetWithOptionalOceanConfig(
        atmosphere=XarrayDataConfig(data_path=""),
        ocean=XarrayDataConfig(data_path=""),
    )
    return InlineInferenceConfig(
        loader=InferenceDataLoaderConfig(
            dataset=dataset,
            start_indices=InferenceInitialConditionIndices(
                first=0, n_initial_conditions=1, interval=1
            ),
        ),
        n_coupled_steps=1,
        coupled_steps_in_memory=1,
        aggregator=InferenceEvaluatorAggregatorConfig(
            log_global_mean_time_series=False,
            log_global_mean_norm_time_series=False,
        ),
        epochs=epochs if epochs is not None else Slice(),
        name=name,
        weight=weight,
    )


def _make_train_config(
    tmp_path,
    inference: InlineInferenceConfig | list[InlineInferenceConfig],
    max_epochs: int = 5,
) -> TrainConfig:
    return TrainConfig(
        train_loader=MagicMock(),
        validation_loader=MagicMock(),
        stepper=_make_stepper_mock(),
        stepper_training=_make_stepper_training_mock(),
        optimization=MagicMock(has_lr_schedule=False),
        logging=MagicMock(),
        max_epochs=max_epochs,
        save_checkpoint=False,
        experiment_dir=str(tmp_path),
        inference=inference,
    )


def test_negative_weight_raises():
    with pytest.raises(ValueError, match="non-negative"):
        _make_inference_config(weight=-1.0)


def test_zero_weight_accepted():
    config = _make_inference_config(weight=0.0)
    assert config.weight == 0.0


def test_default_weight_is_one():
    config = _make_inference_config()
    assert config.weight == 1.0


def test_inference_single_config_gives_list(tmp_path):
    config = _make_train_config(tmp_path, _make_inference_config())
    assert isinstance(config.inference, InlineInferenceConfig)
    assert isinstance(config.inference_list, list)
    assert len(config.inference_list) == 1
    assert config.inference_names == ["inference"]


def test_inference_names_single_unnamed(tmp_path):
    config = _make_train_config(tmp_path, [_make_inference_config()])
    assert config.inference_names == ["inference"]


def test_inference_names_multiple_unnamed(tmp_path):
    config = _make_train_config(
        tmp_path, [_make_inference_config(), _make_inference_config()]
    )
    assert config.inference_names == ["inference_0", "inference_1"]


def test_inference_names_explicit(tmp_path):
    config = _make_train_config(
        tmp_path,
        [_make_inference_config(name="weather"), _make_inference_config(name="clim")],
    )
    assert config.inference_names == ["weather", "clim"]


def test_inference_names_mixed(tmp_path):
    config = _make_train_config(
        tmp_path,
        [_make_inference_config(name="weather"), _make_inference_config()],
    )
    assert config.inference_names == ["weather", "inference_1"]


def test_inference_names_empty(tmp_path):
    config = _make_train_config(tmp_path, [])
    assert config.inference_names == []


def test_duplicate_inference_names_raises(tmp_path):
    with pytest.raises(ValueError, match="Duplicate inference names"):
        _make_train_config(
            tmp_path,
            [_make_inference_config(name="same"), _make_inference_config(name="same")],
        )


@pytest.mark.parametrize("reserved_name", ["train", "val"])
def test_reserved_inference_name_raises(tmp_path, reserved_name):
    with pytest.raises(ValueError, match="collide with reserved names"):
        _make_train_config(tmp_path, [_make_inference_config(name=reserved_name)])


def test_get_inference_epoch_sets_empty(tmp_path):
    config = _make_train_config(tmp_path, [], max_epochs=5)
    assert config.get_inference_epoch_sets() == []
    assert config.get_inference_epochs() == []


def test_get_inference_epoch_sets_single_default(tmp_path):
    config = _make_train_config(tmp_path, [_make_inference_config()], max_epochs=3)
    assert config.get_inference_epoch_sets() == [{0, 1, 2, 3}]
    assert config.get_inference_epochs() == [0, 1, 2, 3]


def test_get_inference_epoch_sets_per_config(tmp_path):
    config = _make_train_config(
        tmp_path,
        [
            _make_inference_config(epochs=Slice(step=2)),
            _make_inference_config(epochs=Slice(step=3)),
        ],
        max_epochs=6,
    )
    epoch_sets = config.get_inference_epoch_sets()
    assert epoch_sets[0] == {0, 2, 4, 6}
    assert epoch_sets[1] == {0, 3, 6}
    assert config.get_inference_epochs() == [0, 2, 3, 4, 6]


def test_validate_n_steps_passes_when_unbounded():
    _validate_n_steps(
        n_coupled_steps=1,
        n_inner_steps=2,
        component_n_steps_max=CoupledOptionalInt(ocean=None, atmosphere=None),
    )


def test_validate_n_steps_passes_at_limit():
    # Equality is allowed: n_steps==n_coupled_steps means losses for steps
    # 0..n_steps-1, all within range.
    _validate_n_steps(
        n_coupled_steps=4,
        n_inner_steps=2,
        component_n_steps_max=CoupledOptionalInt(ocean=4, atmosphere=8),
    )


def test_validate_n_steps_rejects_ocean_overshoot():
    with pytest.raises(ValueError, match=r"ocean.*exceeds n_coupled_steps"):
        _validate_n_steps(
            n_coupled_steps=2,
            n_inner_steps=3,
            component_n_steps_max=CoupledOptionalInt(ocean=3, atmosphere=None),
        )


def test_validate_n_steps_rejects_atmosphere_overshoot():
    with pytest.raises(
        ValueError,
        match=r"atmosphere.*exceeds n_coupled_steps \* n_inner_steps",
    ):
        _validate_n_steps(
            n_coupled_steps=2,
            n_inner_steps=3,
            component_n_steps_max=CoupledOptionalInt(ocean=None, atmosphere=7),
        )


def test_validate_n_steps_lists_both_components_when_both_misconfigured():
    with pytest.raises(ValueError) as exc_info:
        _validate_n_steps(
            n_coupled_steps=2,
            n_inner_steps=3,
            component_n_steps_max=CoupledOptionalInt(ocean=5, atmosphere=10),
        )
    msg = str(exc_info.value)
    assert "ocean" in msg
    assert "atmosphere" in msg


def test_validate_n_steps_uses_sampler_max_via_config():
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=1, probability=0.5),
            TimeLengthProbability(steps=5, probability=0.5),
        ]
    )
    config = ComponentTrainingConfig(loss=_MOCK_LOSS, n_steps=sampler)
    bounds = CoupledOptionalInt(ocean=config.n_steps_max, atmosphere=None)
    with pytest.raises(ValueError, match=r"ocean"):
        _validate_n_steps(
            n_coupled_steps=2, n_inner_steps=2, component_n_steps_max=bounds
        )


def test_validate_n_steps_does_not_short_circuit_on_null_weight():
    # A loss_weight=0 component still has a non-None n_steps_max if a value was
    # explicitly set, and the validator surfaces the misconfiguration even
    # though the loss is null. This avoids silently accepting confused configs.
    null_config = ComponentTrainingConfig(
        loss=_MOCK_LOSS,
        loss_weight=0.0,
        n_steps=999,
    )
    bounds = CoupledOptionalInt(ocean=null_config.n_steps_max, atmosphere=None)
    with pytest.raises(ValueError, match=r"ocean"):
        _validate_n_steps(
            n_coupled_steps=1, n_inner_steps=1, component_n_steps_max=bounds
        )
