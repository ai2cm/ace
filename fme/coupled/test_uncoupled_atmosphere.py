"""Tests for uncoupled atmosphere batches in coupled training
(``objective_per_batch.uncoupled_atmosphere`` and
``CoupledTrainStepperConfig.uncoupled_atmosphere``)."""

import dataclasses

import pytest
import torch

from fme.ace.stepper.parameter_init import ParameterInitializationConfig
from fme.ace.stepper.time_length_probabilities import (
    TimeLengthProbabilities,
    TimeLengthProbability,
)
from fme.core.loss import StepLossConfig
from fme.core.optimization import (
    NullOptimization,
    OptimizationConfig,
    build_optimization_dict,
)
from fme.core.registry.module import ModuleSelector
from fme.coupled.data_loading.mixed import MixedGriddedData
from fme.coupled.stepper import (
    ComponentTrainingConfig,
    CoupledTrainStepperConfig,
    ObjectivePerBatchConfig,
)
from fme.coupled.test_stepper import get_data, get_train_stepper_and_batch

_ATMOSPHERE_IN_NAMES = ["surface_temperature", "ocean_fraction"]


def _mse(**kwargs) -> ComponentTrainingConfig:
    return ComponentTrainingConfig(loss=StepLossConfig(type="MSE"), **kwargs)


def _uncoupled(n_steps=2, **kwargs) -> ComponentTrainingConfig:
    return _mse(
        n_steps=TimeLengthProbabilities(outcomes=[TimeLengthProbability(n_steps, 1.0)]),
        **kwargs,
    )


def _config(objective: ObjectivePerBatchConfig, **kwargs) -> CoupledTrainStepperConfig:
    return CoupledTrainStepperConfig(
        n_coupled_steps=1,
        ocean=_mse(n_steps=1),
        atmosphere=_mse(n_steps=2),
        objective_per_batch=objective,
        uncoupled_atmosphere=_uncoupled(),
        **kwargs,
    )


def _train_stepper(objective: ObjectivePerBatchConfig, **kwargs):
    torch.manual_seed(0)
    return get_train_stepper_and_batch(
        train_stepper_config=_config(objective, **kwargs),
        ocean_in_names=["sst"],
        ocean_out_names=["sst"],
        atmosphere_in_names=_ATMOSPHERE_IN_NAMES,
        atmosphere_out_names=["surface_temperature"],
        n_forward_times_ocean=1,
        n_forward_times_atmosphere=2,
        n_samples=2,
        ocean_builder=ModuleSelector(
            type="prebuilt", config={"module": torch.nn.Conv2d(1, 1, 1)}
        ),
        atmosphere_builder=ModuleSelector(
            type="prebuilt", config={"module": torch.nn.Conv2d(2, 1, 1)}
        ),
    )


def _uncoupled_batch():
    return get_data(
        _ATMOSPHERE_IN_NAMES, n_samples=2, n_time=3, realm="atmosphere"
    ).data


_D2D = ObjectivePerBatchConfig(ocean=1.0, uncoupled_atmosphere=1.0)


@pytest.mark.parametrize(
    "objective, component, match",
    [
        pytest.param(
            ObjectivePerBatchConfig(ocean=1.0, uncoupled_atmosphere=1.0),
            None,
            "set exactly when",
            id="weight_without_config",
        ),
        pytest.param(
            ObjectivePerBatchConfig(ocean=1.0),
            _uncoupled(),
            "set exactly when",
            id="config_without_weight",
        ),
        pytest.param(None, _uncoupled(), "set exactly when", id="no_objective"),
        pytest.param(_D2D, _mse(), "n_steps must be set", id="unbounded_n_steps"),
        pytest.param(_D2D, _mse(n_steps=0), "must be > 0", id="zero_n_steps"),
        pytest.param(
            _D2D,
            _uncoupled(
                parameter_init=ParameterInitializationConfig(weights_path="x.tar")
            ),
            "parameter_init",
            id="parameter_init",
        ),
        pytest.param(_D2D, _uncoupled(loss_weight=2.0), "loss_weight", id="weight"),
    ],
)
def test_uncoupled_atmosphere_config_validation(objective, component, match):
    with pytest.raises(ValueError, match=match):
        CoupledTrainStepperConfig(
            n_coupled_steps=1,
            ocean=_mse(),
            atmosphere=_mse(),
            objective_per_batch=objective,
            uncoupled_atmosphere=component,
        )


@pytest.mark.parametrize(
    "weights, match",
    [
        ({"uncoupled_atmosphere": 1.0}, "at least one weight"),
        ({"ocean": 1.0, "uncoupled_atmosphere": -1.0}, "must be >= 0"),
    ],
)
def test_objective_per_batch_uncoupled_atmosphere_weight_validation(weights, match):
    with pytest.raises(ValueError, match=match):
        ObjectivePerBatchConfig(**weights)


def test_uncoupled_atmosphere_share_and_coupled_draw_weights():
    objective = ObjectivePerBatchConfig(
        atmosphere=1.0, ocean=1.0, joint=2.0, uncoupled_atmosphere=4.0
    )
    assert objective.uncoupled_atmosphere_share() == 0.5
    assert set(objective.as_dict()) == {"atmosphere", "ocean", "joint"}


def test_eval_objective_draw_excludes_uncoupled_atmosphere():
    train_stepper, coupled_data, _, _ = _train_stepper(_D2D)
    train_stepper.set_eval()
    train_stepper.seed_eval(0)
    for _ in range(4):
        train_stepper.train_on_batch(coupled_data.data, NullOptimization())
        assert not train_stepper._loss.step_is_optimized("atmosphere", 0)
        assert train_stepper._loss.step_is_optimized("ocean", 0)


def test_uncoupled_atmosphere_data_requirements():
    train_stepper, _, stepper_config, _ = _train_stepper(_D2D)
    config = _config(_D2D)
    requirements = config.get_uncoupled_atmosphere_data_requirements(stepper_config)
    assert requirements.n_timesteps == 3
    assert set(_ATMOSPHERE_IN_NAMES) <= set(requirements.names)


def test_uncoupled_atmosphere_batch_trains_atmosphere_alone():
    train_stepper, _, _, _ = _train_stepper(_D2D)
    out = train_stepper.train_on_batch(_uncoupled_batch(), NullOptimization())
    assert out.ocean.metrics == {} and out.ocean.gen_data == {}
    metrics = out.get_metrics()
    assert torch.isfinite(metrics["loss"])
    assert metrics["uncoupled_atmosphere/loss"] == metrics["loss"]
    assert not any(k.startswith("loss/") for k in metrics)
    assert "surface_temperature" in out.atmosphere.gen_data


def test_uncoupled_atmosphere_batch_without_config_raises():
    train_stepper, _, _, _ = get_train_stepper_and_batch(
        ocean_in_names=["sst"],
        ocean_out_names=["sst"],
        atmosphere_in_names=_ATMOSPHERE_IN_NAMES,
        atmosphere_out_names=["surface_temperature"],
        n_forward_times_ocean=1,
        n_forward_times_atmosphere=2,
        n_samples=2,
    )
    with pytest.raises(ValueError, match="not set"):
        train_stepper.train_on_batch(_uncoupled_batch(), NullOptimization())


def _params(train_stepper) -> dict[str, list[torch.Tensor]]:
    return {
        key: [p.detach().clone() for p in modules.parameters()]
        for key, modules in train_stepper.optimization_modules.items()
    }


def _changed(before, after) -> dict[str, bool]:
    return {
        key: any(not torch.equal(a, b) for a, b in zip(before[key], after[key]))
        for key in before
    }


def _per_component_optimization(train_stepper):
    configs = {
        key: OptimizationConfig(optimizer_type="AdamW", lr=0.01)
        for key in train_stepper.optimization_modules
    }
    return build_optimization_dict(
        configs, train_stepper.optimization_modules, max_epochs=1
    )


def _optimizer_steps(optimization) -> dict[str, int]:
    """Parameters with optimizer state, per component."""
    return {
        key: len(state["optimizer_state_dict"]["state"])
        for key, state in optimization.get_state()["components"].items()
    }


def test_d2d_atmosphere_trains_on_uncoupled_batches_only():
    """objective_per_batch {ocean, uncoupled_atmosphere} with per-component
    optimization: coupled batches step only the ocean, uncoupled atmosphere
    batches only the atmosphere."""
    train_stepper, coupled_data, _, _ = _train_stepper(_D2D)
    optimization = _per_component_optimization(train_stepper)

    before = _params(train_stepper)
    train_stepper.train_on_batch(coupled_data.data, optimization)
    assert _changed(before, _params(train_stepper)) == {
        "atmosphere": False,
        "ocean": True,
    }
    assert _optimizer_steps(optimization)["atmosphere"] == 0

    before = _params(train_stepper)
    train_stepper.train_on_batch(_uncoupled_batch(), optimization)
    assert _changed(before, _params(train_stepper)) == {
        "atmosphere": True,
        "ocean": False,
    }
    assert _optimizer_steps(optimization)["atmosphere"] > 0

    # atmosphere AdamW state (momentum, weight decay) must not move it on coupled
    before = _params(train_stepper)
    train_stepper.train_on_batch(coupled_data.data, optimization)
    assert _changed(before, _params(train_stepper)) == {
        "atmosphere": False,
        "ocean": True,
    }


def test_uncoupled_atmosphere_batch_single_optimizer_leaves_ocean():
    train_stepper, _, _, _ = _train_stepper(
        ObjectivePerBatchConfig(atmosphere=1.0, ocean=1.0, uncoupled_atmosphere=1.0)
    )
    optimization = OptimizationConfig(optimizer_type="AdamW", lr=0.01).build(
        train_stepper.optimization_modules, max_epochs=1
    )
    before = _params(train_stepper)
    train_stepper.train_on_batch(_uncoupled_batch(), optimization)
    assert _changed(before, _params(train_stepper)) == {
        "atmosphere": True,
        "ocean": False,
    }


def test_uncoupled_atmosphere_absent_leaves_config_unchanged():
    config = CoupledTrainStepperConfig(
        n_coupled_steps=1,
        ocean=_mse(),
        atmosphere=_mse(),
        objective_per_batch=ObjectivePerBatchConfig(atmosphere=1.0, ocean=1.0),
    )
    assert config.uncoupled_atmosphere is None
    assert dataclasses.replace(config).uncoupled_atmosphere is None


# ---------------------------------------------------------------------------
# MixedGriddedData (ported from ace#1627)
# ---------------------------------------------------------------------------


class _FakeGriddedData:
    """Enough of GriddedData for MixedGriddedData: a loader of labelled batches."""

    def __init__(self, label: str, n_batches: int, batch_size: int = 2):
        self.label = label
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_samples = n_batches * batch_size

    def subset_loader(self, start_batch=None, stop_batch=None):
        return [(self.label, i) for i in range(self.n_batches)][start_batch:stop_batch]

    @property
    def loader(self):
        return self.subset_loader()

    def set_epoch(self, epoch: int):
        pass

    def alternate_shuffle(self):
        pass

    def log_info(self, name: str):
        pass


def _mixed(fraction=0.25, n_coupled=12, n_uncoupled=100):
    return MixedGriddedData(
        _FakeGriddedData("coupled", n_coupled),  # type: ignore[arg-type]
        _FakeGriddedData("uncoupled", n_uncoupled),  # type: ignore[arg-type]
        fraction=fraction,
    )


@pytest.mark.parametrize(
    "weights, n_coupled, n_uncoupled",
    [
        ({"ocean": 1.0, "uncoupled_atmosphere": 1.0}, 10, 10),
        ({"atmosphere": 1.0, "ocean": 1.0, "uncoupled_atmosphere": 2 / 3}, 12, 4),
        (
            {"atmosphere": 1.0, "ocean": 1.0, "joint": 1.0, "uncoupled_atmosphere": 1},
            9,
            3,
        ),
    ],
)
def test_uncoupled_count_from_weights(weights, n_coupled, n_uncoupled):
    share = ObjectivePerBatchConfig(**weights).uncoupled_atmosphere_share()
    kinds = [kind for kind, _ in _mixed(share, n_coupled=n_coupled).loader]
    assert kinds.count("coupled") == n_coupled
    assert kinds.count("uncoupled") == n_uncoupled


def test_mixed_data_interleaves_the_configured_share():
    data = _mixed(fraction=0.25, n_coupled=12)
    batches = list(data.loader)
    assert data.n_batches == len(batches) == 16
    kinds = [kind for kind, _ in batches]
    assert kinds.count("coupled") == 12 and kinds.count("uncoupled") == 4
    assert kinds[0] == "coupled"
    for label in ("coupled", "uncoupled"):
        assert [i for kind, i in batches if kind == label] == list(
            range(kinds.count(label))
        )


def test_mixed_data_sequence_is_reproducible_and_changes_with_epoch():
    data = _mixed()
    first = [kind for kind, _ in data.loader]
    assert [kind for kind, _ in data.loader] == first
    data.set_epoch(1)
    assert [kind for kind, _ in data.loader] != first


def test_mixed_data_subset_resumes_the_same_sequence():
    data = _mixed()
    full = list(data.loader)
    assert list(data.subset_loader(start_batch=5, stop_batch=11)) == full[5:11]
    assert list(data.subset_loader(start_batch=5)) == full[5:]
    restarted = _mixed()
    assert list(restarted.subset_loader(start_batch=5)) == full[5:]
    # a stop-only subset (the trainer's post-epoch evaluation pass) is coupled-only
    assert all(kind == "coupled" for kind, _ in data.subset_loader(stop_batch=3))


def test_mixed_data_rejects_too_few_uncoupled_batches():
    with pytest.raises(ValueError, match=r"has 3 batches, fewer than the 10 "):
        _mixed(fraction=0.5, n_coupled=10, n_uncoupled=3)
    assert _mixed(fraction=0.5, n_coupled=10, n_uncoupled=10).n_batches == 20


def test_mixed_data_rejects_fraction_one():
    with pytest.raises(ValueError, match="fraction"):
        _mixed(fraction=1.0)
