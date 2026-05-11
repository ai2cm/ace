import dataclasses

import pytest

from fme.ace.aggregator.inference.main import InferenceEvaluatorAggregatorConfig
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
from fme.ace.train.train_config import InlineInferenceConfig, TrainConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import OptimizationConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.typing_ import Slice


def _make_inference_config(
    name: str | None = None, weight: float = 1.0, epochs: Slice | None = None
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
        aggregator=InferenceEvaluatorAggregatorConfig(
            log_global_mean_time_series=False,
            log_global_mean_norm_time_series=False,
            log_step_means=[],
        ),
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
) -> TrainConfig:
    dummy_loader = DataLoaderConfig(
        dataset=XarrayDataConfig(data_path=""), batch_size=1
    )
    return TrainConfig(
        experiment_dir=str(tmp_path),
        stepper=_make_stepper_config(),
        stepper_training=TrainStepperConfig(n_forward_steps=1),
        train_loader=dummy_loader,
        validation_loader=dummy_loader,
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
