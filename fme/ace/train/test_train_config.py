import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from fme.ace.aggregator.inference.main import InferenceEvaluatorAggregatorConfig
from fme.ace.aggregator.inference.time_mean import TimeMeanMetricConfig
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.inference import (
    InferenceDataLoaderConfig,
    InferenceInitialConditionIndices,
)
from fme.ace.stepper.single_module import (
    ModuleSelector,
    NetworkAndLossNormalizationConfig,
    NormalizationConfig,
    StepperConfig,
    TrainStepperConfig,
)
from fme.ace.train.train_config import (
    InlineInferenceConfig,
    InlineValidationConfig,
    TrainConfig,
)
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import OptimizationConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.typing_ import Slice


def _make_validation_config(
    name: str | None = None, weight: float = 1.0
) -> InlineValidationConfig:
    return InlineValidationConfig(
        loader=DataLoaderConfig(dataset=XarrayDataConfig(data_path=""), batch_size=1),
        name=name,
        weight=weight,
    )


def _make_inference_config(
    name: str | None = None,
    weight: float = 1.0,
    epochs: Slice | None = None,
    aggregator: InferenceEvaluatorAggregatorConfig | None = None,
) -> InlineInferenceConfig:
    return InlineInferenceConfig(
        loader=InferenceDataLoaderConfig(
            dataset=XarrayDataConfig(data_path=""),
            start_indices=InferenceInitialConditionIndices(
                first=0, n_initial_conditions=1, interval=1
            ),
        ),
        n_forward_steps=1,
        forward_steps_in_memory=1,
        aggregator=aggregator or InferenceEvaluatorAggregatorConfig(),
        epochs=epochs if epochs is not None else Slice(),
        name=name,
        weight=weight,
    )


def _make_stepper_config() -> StepperConfig:
    step = StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                in_names=[],
                out_names=[],
                normalization=NetworkAndLossNormalizationConfig(
                    network=NormalizationConfig(
                        global_means_path="", global_stds_path=""
                    ),
                ),
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet", config={}
                ),
            ),
        ),
    )
    return StepperConfig(step=step)


def _make_train_config(
    tmp_path,
    inference: InlineInferenceConfig | list[InlineInferenceConfig],
    max_epochs: int = 5,
    validation: InlineValidationConfig | list[InlineValidationConfig] | None = None,
) -> TrainConfig:
    if validation is None:
        validation = _make_validation_config()
    return TrainConfig(
        experiment_dir=str(tmp_path),
        stepper=_make_stepper_config(),
        stepper_training=TrainStepperConfig(n_forward_steps=1),
        train_loader=DataLoaderConfig(
            dataset=XarrayDataConfig(data_path=""), batch_size=1
        ),
        validation=validation,
        optimization=OptimizationConfig(),
        logging=LoggingConfig(),
        max_epochs=max_epochs,
        save_checkpoint=False,
        inference=inference,
    )


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


def test_negative_weight_raises():
    with pytest.raises(ValueError, match="non-negative"):
        _make_inference_config(weight=-1.0)


def test_zero_weight_accepted():
    config = _make_inference_config(weight=0.0)
    assert config.weight == 0.0


def test_default_weight_is_one():
    config = _make_inference_config()
    assert config.weight == 1.0


def test_disabled_time_mean_norm_with_positive_weight_raises():
    agg = InferenceEvaluatorAggregatorConfig(
        time_mean_norm=TimeMeanMetricConfig(target="norm", enabled=False),
    )
    with pytest.raises(ValueError, match="time_mean_norm must be enabled"):
        _make_inference_config(weight=1.0, aggregator=agg)


def test_disabled_time_mean_norm_with_zero_weight_accepted():
    agg = InferenceEvaluatorAggregatorConfig(
        time_mean_norm=TimeMeanMetricConfig(target="norm", enabled=False),
    )
    config = _make_inference_config(weight=0.0, aggregator=agg)
    assert config.weight == 0.0


def test_get_inference_epoch_sets_empty(tmp_path):
    config = _make_train_config(tmp_path, [], max_epochs=5)
    assert config.get_inference_epoch_sets() == []
    assert config.get_inference_epochs() == []


def test_get_inference_epoch_sets_single_default(tmp_path):
    config = _make_train_config(tmp_path, [_make_inference_config()], max_epochs=3)
    assert config.get_inference_epoch_sets() == [{1, 2, 3}]
    assert config.get_inference_epochs() == [1, 2, 3]


def test_get_inference_epoch_sets_per_config_zero_weight(tmp_path):
    config = _make_train_config(
        tmp_path,
        [
            _make_inference_config(epochs=Slice(step=2), weight=1.0),
            _make_inference_config(epochs=Slice(step=3), weight=0.0),
        ],
        max_epochs=6,
    )
    epoch_sets = config.get_inference_epoch_sets()
    assert epoch_sets[0] == {1, 3, 5}
    assert epoch_sets[1] == {1, 4}
    assert config.get_inference_epochs() == [1, 3, 4, 5]


def test_get_inference_epoch_sets_same_weighted_epochs(tmp_path):
    config = _make_train_config(
        tmp_path,
        [
            _make_inference_config(epochs=Slice(step=2), weight=1.0),
            _make_inference_config(epochs=Slice(step=2), weight=2.0),
        ],
        max_epochs=6,
    )
    epoch_sets = config.get_inference_epoch_sets()
    assert epoch_sets[0] == {1, 3, 5}
    assert epoch_sets[1] == {1, 3, 5}


def test_get_inference_epoch_sets_different_weighted_epochs_raises(tmp_path):
    with pytest.raises(ValueError, match="weight > 0 must share the same epoch"):
        _make_train_config(
            tmp_path,
            [
                _make_inference_config(epochs=Slice(step=2), weight=1.0),
                _make_inference_config(epochs=Slice(step=3), weight=1.0),
            ],
            max_epochs=6,
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
    config = _make_train_config(tmp_path, [], validation=_make_validation_config())
    assert isinstance(config.validation, InlineValidationConfig)
    assert isinstance(config.validation_list, list)
    assert len(config.validation_list) == 1
    assert config.validation_names == ["val"]


def test_validation_names_single_unnamed(tmp_path):
    config = _make_train_config(tmp_path, [], validation=[_make_validation_config()])
    assert config.validation_names == ["val"]


def test_validation_names_multiple_unnamed(tmp_path):
    config = _make_train_config(
        tmp_path,
        [],
        validation=[_make_validation_config(), _make_validation_config()],
    )
    assert config.validation_names == ["val_0", "val_1"]


def test_validation_names_explicit(tmp_path):
    config = _make_train_config(
        tmp_path,
        [],
        validation=[
            _make_validation_config(name="era5"),
            _make_validation_config(name="obs"),
        ],
    )
    assert config.validation_names == ["era5", "obs"]


def test_validation_names_mixed(tmp_path):
    config = _make_train_config(
        tmp_path,
        [],
        validation=[_make_validation_config(name="era5"), _make_validation_config()],
    )
    assert config.validation_names == ["era5", "val_1"]


def test_duplicate_validation_names_raises(tmp_path):
    with pytest.raises(ValueError, match="Duplicate validation names"):
        _make_train_config(
            tmp_path,
            [],
            validation=[
                _make_validation_config(name="same"),
                _make_validation_config(name="same"),
            ],
        )


def test_empty_validation_raises(tmp_path):
    with pytest.raises(ValueError, match="At least one validation entry"):
        _make_train_config(tmp_path, [], validation=[])


class TestGetValidationCallback:
    """Smoke test for `get_validation_callback` wiring.

    Helper behavior (weighted loss, missing-metric raise, overlap raise, etc.)
    is covered by `TestBuildValidationCallback` in
    `fme.core.generics.test_trainer`. This test only verifies that entry name
    and weight flow correctly from config through to the shared helper.
    """

    def test_entries_wired_to_tasks(self):
        from fme.ace.train.train import get_validation_callback

        # Use MagicMock for entry configs so we can control the aggregator
        # returned by entry_config.aggregator.build(...).
        def make_entry(name, weight, loss):
            config = MagicMock()
            config.weight = weight
            config.aggregator.build.return_value.get_loss.return_value = loss
            return (config, MagicMock(), name)

        entries = [
            make_entry("a", weight=2.0, loss=0.1),
            make_entry("b", weight=3.0, loss=0.2),
        ]
        stepper = MagicMock()
        with patch(
            "fme.core.generics.trainer.run_validation",
            side_effect=[{}, {}],
        ):
            callback = get_validation_callback(
                validation_entries=entries,
                stepper=stepper,
                dataset_info=MagicMock(),
                loss_scaling=None,
                loss_names=None,
                save_per_epoch_diagnostics=False,
                output_dir="/tmp/out",
            )
            _, loss = callback(epoch=1)
        assert loss == pytest.approx(2.0 * 0.1 + 3.0 * 0.2)


class TestGetInferenceCallback:
    @staticmethod
    def _make_entry(name, weight=1.0, aggregator_loss=None):
        config = MagicMock()
        config.weight = weight
        config.n_forward_steps = 1
        config.n_ensemble_per_ic = 1
        config.aggregator.build.return_value.get_loss.return_value = aggregator_loss
        data = MagicMock()
        dataset_info = MagicMock()
        return (config, data, dataset_info, name)

    @staticmethod
    def _call(
        entries,
        inference_one_epoch_side_effect,
        epoch=1,
        inference_epochs=(1,),
        inference_epoch_sets=None,
    ):
        from fme.ace.train.train import get_inference_callback

        if inference_epoch_sets is None:
            inference_epoch_sets = [{1} for _ in entries]
        stepper = MagicMock()
        with patch(
            "fme.core.generics.trainer.inference_one_epoch",
            side_effect=inference_one_epoch_side_effect,
        ):
            callback = get_inference_callback(
                inference_entries=entries,
                inference_epochs=list(inference_epochs),
                inference_epoch_sets=list(inference_epoch_sets),
                stepper=stepper,
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
        entries = [self._make_entry("inference", weight=2.0, aggregator_loss=0.4)]
        logs, error = self._call(
            entries,
            [{"inference/some_metric": 0.4}],
        )
        assert error == pytest.approx(2.0 * 0.4)
        assert "inference/some_metric" in logs

    def test_zero_weight_excluded_from_error(self):
        entries = [
            self._make_entry("a", weight=1.0, aggregator_loss=0.3),
            self._make_entry("b", weight=0.0, aggregator_loss=999.0),
        ]
        logs, error = self._call(
            entries,
            [
                {"a/some_metric": 0.3},
                {"b/some_metric": 999.0},
            ],
        )
        assert error == pytest.approx(0.3)
        assert "a/some_metric" in logs
        assert "b/some_metric" in logs

    def test_multiple_weighted_entries(self):
        entries = [
            self._make_entry("a", weight=2.0, aggregator_loss=0.1),
            self._make_entry("b", weight=3.0, aggregator_loss=0.2),
        ]
        logs, error = self._call(
            entries,
            [
                {"a/some_metric": 0.1},
                {"b/some_metric": 0.2},
            ],
        )
        assert error == pytest.approx(2.0 * 0.1 + 3.0 * 0.2)

    def test_entry_skipped_when_not_in_epoch_set(self):
        entries = [
            self._make_entry("a", weight=1.0, aggregator_loss=0.5),
            self._make_entry("b", weight=1.0, aggregator_loss=0.7),
        ]
        logs, error = self._call(
            entries,
            [{"a/some_metric": 0.5}],
            epoch=1,
            inference_epochs=(1,),
            inference_epoch_sets=[{1}, {2}],
        )
        assert error == pytest.approx(0.5)
        assert "a/some_metric" in logs
        assert "b/some_metric" not in logs

    def test_weighted_entry_missing_metric_raises(self):
        entries = [self._make_entry("a", weight=1.0, aggregator_loss=None)]
        with pytest.raises(RuntimeError, match="did not produce a loss"):
            self._call(
                entries,
                [{"a/other_metric": 1.0}],
            )
