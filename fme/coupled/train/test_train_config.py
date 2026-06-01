from unittest.mock import MagicMock, patch

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
from fme.coupled.data_loading.config import (
    CoupledDataLoaderConfig,
    CoupledDatasetWithOptionalOceanConfig,
)
from fme.coupled.data_loading.inference import InferenceDataLoaderConfig
from fme.coupled.stepper import ComponentTrainingConfig
from fme.coupled.train.train_config import (
    InlineInferenceConfig,
    InlineValidationConfig,
    TrainConfig,
)
from fme.coupled.typing_ import CoupledOptionalInt

from .train_config import _validate_n_steps

_MOCK_LOSS = MagicMock(spec=StepLossConfig)


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


def _make_validation_config(
    name: str | None = None, weight: float = 1.0
) -> InlineValidationConfig:
    return InlineValidationConfig(
        loader=MagicMock(spec=CoupledDataLoaderConfig),
        name=name,
        weight=weight,
    )


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


def _make_train_config_for_validation(
    tmp_path,
    validation: InlineValidationConfig | list[InlineValidationConfig],
    max_epochs: int = 5,
) -> TrainConfig:
    return TrainConfig(
        train_loader=MagicMock(),
        validation=validation,
        stepper=_make_stepper_mock(),
        stepper_training=_make_stepper_training_mock(),
        optimization=MagicMock(has_lr_schedule=False),
        logging=MagicMock(),
        max_epochs=max_epochs,
        save_checkpoint=False,
        experiment_dir=str(tmp_path),
    )


def _make_train_config_for_inference(
    tmp_path,
    inference: InlineInferenceConfig | list[InlineInferenceConfig],
    max_epochs: int = 5,
) -> TrainConfig:
    return TrainConfig(
        train_loader=MagicMock(),
        validation=_make_validation_config(),
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
    config = _make_train_config_for_inference(tmp_path, _make_inference_config())
    assert isinstance(config.inference, InlineInferenceConfig)
    assert isinstance(config.inference_list, list)
    assert len(config.inference_list) == 1
    assert config.inference_names == ["inference"]


def test_inference_names_single_unnamed(tmp_path):
    config = _make_train_config_for_inference(tmp_path, [_make_inference_config()])
    assert config.inference_names == ["inference"]


def test_inference_names_multiple_unnamed(tmp_path):
    config = _make_train_config_for_inference(
        tmp_path, [_make_inference_config(), _make_inference_config()]
    )
    assert config.inference_names == ["inference_0", "inference_1"]


def test_inference_names_explicit(tmp_path):
    config = _make_train_config_for_inference(
        tmp_path,
        [_make_inference_config(name="weather"), _make_inference_config(name="clim")],
    )
    assert config.inference_names == ["weather", "clim"]


def test_inference_names_mixed(tmp_path):
    config = _make_train_config_for_inference(
        tmp_path,
        [_make_inference_config(name="weather"), _make_inference_config()],
    )
    assert config.inference_names == ["weather", "inference_1"]


def test_inference_names_empty(tmp_path):
    config = _make_train_config_for_inference(tmp_path, [])
    assert config.inference_names == []


def test_duplicate_inference_names_raises(tmp_path):
    with pytest.raises(ValueError, match="Duplicate inference names"):
        _make_train_config_for_inference(
            tmp_path,
            [_make_inference_config(name="same"), _make_inference_config(name="same")],
        )


@pytest.mark.parametrize("reserved_name", ["train", "val"])
def test_reserved_inference_name_raises(tmp_path, reserved_name):
    with pytest.raises(ValueError, match="collide with reserved names"):
        _make_train_config_for_inference(
            tmp_path, [_make_inference_config(name=reserved_name)]
        )


def test_get_inference_epoch_sets_empty(tmp_path):
    config = _make_train_config_for_inference(tmp_path, [], max_epochs=5)
    assert config.get_inference_epoch_sets() == []
    assert config.get_inference_epochs() == []


def test_get_inference_epoch_sets_single_default(tmp_path):
    config = _make_train_config_for_inference(
        tmp_path, [_make_inference_config()], max_epochs=3
    )
    assert config.get_inference_epoch_sets() == [{0, 1, 2, 3}]
    assert config.get_inference_epochs() == [0, 1, 2, 3]


def test_get_inference_epoch_sets_per_config(tmp_path):
    config = _make_train_config_for_inference(
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


def test_validation_negative_weight_raises():
    with pytest.raises(ValueError, match="non-negative"):
        _make_validation_config(weight=-1.0)


def test_validation_zero_weight_accepted():
    config = _make_validation_config(weight=0.0)
    assert config.weight == 0.0


def test_validation_default_weight_is_one():
    config = _make_validation_config()
    assert config.weight == 1.0


def test_validation_single_config_gives_list(tmp_path):
    config = _make_train_config_for_validation(tmp_path, _make_validation_config())
    assert isinstance(config.validation, InlineValidationConfig)
    assert isinstance(config.validation_list, list)
    assert len(config.validation_list) == 1
    assert config.validation_names == ["val"]


def test_validation_names_single_unnamed(tmp_path):
    config = _make_train_config_for_validation(tmp_path, [_make_validation_config()])
    assert config.validation_names == ["val"]


def test_validation_names_multiple_unnamed(tmp_path):
    config = _make_train_config_for_validation(
        tmp_path, [_make_validation_config(), _make_validation_config()]
    )
    assert config.validation_names == ["val_0", "val_1"]


def test_validation_names_explicit(tmp_path):
    config = _make_train_config_for_validation(
        tmp_path,
        [
            _make_validation_config(name="era5"),
            _make_validation_config(name="obs"),
        ],
    )
    assert config.validation_names == ["era5", "obs"]


def test_validation_names_mixed(tmp_path):
    config = _make_train_config_for_validation(
        tmp_path,
        [_make_validation_config(name="era5"), _make_validation_config()],
    )
    assert config.validation_names == ["era5", "val_1"]


def test_empty_validation_raises(tmp_path):
    with pytest.raises(ValueError, match="At least one validation entry"):
        _make_train_config_for_validation(tmp_path, [])


def test_duplicate_validation_names_raises(tmp_path):
    with pytest.raises(ValueError, match="Duplicate validation names"):
        _make_train_config_for_validation(
            tmp_path,
            [
                _make_validation_config(name="same"),
                _make_validation_config(name="same"),
            ],
        )


class TestGetValidationCallback:
    """Smoke test for `get_validation_callback` wiring.

    Helper behavior (weighted loss, missing-metric raise, overlap raise, etc.)
    is covered by `TestBuildValidationCallback` in
    `fme.core.generics.test_trainer`. This test only verifies that entry name
    and weight flow correctly from config through to the shared helper.
    """

    def test_entries_wired_to_tasks(self):
        from fme.coupled.train.train import get_validation_callback

        entries = [
            (_make_validation_config(name="a", weight=2.0), MagicMock(), "a"),
            (_make_validation_config(name="b", weight=3.0), MagicMock(), "b"),
        ]
        stepper = MagicMock()
        with patch(
            "fme.core.generics.trainer.run_validation",
            side_effect=[{"a/mean/loss": 0.1}, {"b/mean/loss": 0.2}],
        ):
            callback = get_validation_callback(
                validation_entries=entries,
                stepper=stepper,
                dataset_info=MagicMock(),
                loss_scaling=MagicMock(),
                save_per_epoch_diagnostics=False,
                output_dir="/tmp/out",
            )
            _, loss = callback(epoch=1)
        assert loss == pytest.approx(2.0 * 0.1 + 3.0 * 0.2)


class TestGetInferenceCallback:
    @staticmethod
    def _make_entry(name, weight=1.0, n_coupled_steps=1):
        config = MagicMock()
        config.weight = weight
        config.n_coupled_steps = n_coupled_steps
        data = MagicMock()
        # data.loader is iterated once to get a batch for initial_times
        data.loader = iter([MagicMock()])
        return (config, data, name)

    @staticmethod
    def _call(
        entries,
        inference_one_epoch_side_effect,
        epoch=1,
        inference_epochs=(1,),
        inference_epoch_sets=None,
    ):
        from fme.coupled.train.train import get_inference_callback

        if inference_epoch_sets is None:
            inference_epoch_sets = [{1} for _ in entries]
        stepper = MagicMock()
        with patch(
            "fme.coupled.train.train.inference_one_epoch",
            side_effect=inference_one_epoch_side_effect,
        ):
            callback = get_inference_callback(
                inference_entries=entries,
                inference_epochs=list(inference_epochs),
                inference_epoch_sets=list(inference_epoch_sets),
                stepper=stepper,
                dataset_info=MagicMock(),
                output_dir="/tmp/out",
                save_per_epoch_diagnostics=False,
            )
            return callback(epoch=epoch)

    def test_epoch_not_in_inference_epochs_returns_empty(self):
        entries = [self._make_entry("inference")]
        logs, error = self._call(
            entries,
            inference_one_epoch_side_effect=[],
            epoch=2,
            inference_epochs=(1,),
        )
        assert logs == {}
        assert error is None

    def test_single_entry_weighted_error(self):
        entries = [self._make_entry("inference", weight=2.0)]
        logs, error = self._call(
            entries,
            [{"inference/time_mean_norm/rmse/channel_mean": 0.4}],
        )
        assert error == pytest.approx(2.0 * 0.4)
        assert "inference/time_mean_norm/rmse/channel_mean" in logs

    def test_missing_metric_returns_none_error(self):
        entries = [self._make_entry("a", weight=1.0)]
        logs, error = self._call(
            entries,
            [{"a/other_metric": 1.0}],
        )
        assert error is None
        assert "a/other_metric" in logs

    def test_multiple_weighted_entries(self):
        entries = [
            self._make_entry("a", weight=2.0),
            self._make_entry("b", weight=3.0),
        ]
        logs, error = self._call(
            entries,
            [
                {"a/time_mean_norm/rmse/channel_mean": 0.1},
                {"b/time_mean_norm/rmse/channel_mean": 0.2},
            ],
        )
        assert error == pytest.approx(2.0 * 0.1 + 3.0 * 0.2)

    def test_entry_skipped_when_not_in_epoch_set(self):
        entries = [
            self._make_entry("a", weight=1.0),
            self._make_entry("b", weight=1.0),
        ]
        logs, error = self._call(
            entries,
            [{"a/time_mean_norm/rmse/channel_mean": 0.5}],
            epoch=1,
            inference_epochs=(1,),
            inference_epoch_sets=[{1}, {2}],
        )
        assert error == pytest.approx(0.5)
        assert "a/time_mean_norm/rmse/channel_mean" in logs
        assert "b/time_mean_norm/rmse/channel_mean" not in logs
