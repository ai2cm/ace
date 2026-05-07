from collections.abc import Callable, Generator
from unittest.mock import MagicMock, Mock

import pytest
import torch

from fme.ace.stepper.time_length_probabilities import (
    TimeLengthProbabilities,
    TimeLengthProbability,
)
from fme.core.loss import LossOutput, StepLoss
from fme.core.typing_ import EnsembleTensorDict, TensorMapping

from .loss import LossContributionsConfig, StepLossABC, StepPredictionABC
from .stepper import ComponentEnsembleStepPrediction, CoupledStepperTrainLoss


def _wrap_as_loss_output(value: torch.Tensor) -> LossOutput:
    """Wrap a scalar tensor as a LossOutput for mocking StepLoss."""
    return LossOutput(loss=value, channel_dim=0, channel_names=["mock"])


def step_and_target_gen(
    n_atmos_per_ocean=2,
) -> Generator[tuple[ComponentEnsembleStepPrediction, TensorMapping], None, None]:
    torch.manual_seed(0)
    atmos_step = 0
    ocean_step = 0
    while True:
        yield (
            ComponentEnsembleStepPrediction(
                realm="atmosphere",
                data=EnsembleTensorDict(
                    {"a": torch.rand(1, 1, 1, 3), "b": torch.rand(1, 1, 1, 3)}
                ),
                step=atmos_step,
            ),
            {"a": torch.rand(1, 1, 1, 3), "b": torch.zeros(1, 1, 1, 3)},
        )
        if atmos_step % n_atmos_per_ocean == 1:
            yield (
                ComponentEnsembleStepPrediction(
                    realm="ocean",
                    data=EnsembleTensorDict(
                        {"o": torch.rand(1, 1, 1, 3), "c": torch.rand(1, 1, 1, 3)}
                    ),
                    step=ocean_step,
                ),
                {"o": torch.rand(1, 1, 1, 3), "c": torch.zeros(1, 1, 1, 3)},
            )
            ocean_step += 1
        atmos_step += 1


@pytest.fixture(scope="module")
def steps_thru_atmos_7() -> list[tuple[ComponentEnsembleStepPrediction, TensorMapping]]:
    """
    Fixture to generate a sequence of steps and targets
    """
    out = []
    for prediction, target_data in step_and_target_gen():
        out.append((prediction, target_data))
        if prediction.realm == "atmosphere" and prediction.step >= 7:
            break
    return out


class _StepLoss(StepLossABC):
    def __init__(
        self,
        loss_obj: Callable[[TensorMapping, TensorMapping, int], torch.Tensor],
        time_dim: int = 1,
    ):
        self._loss_obj = loss_obj
        self._time_dim = time_dim

    @property
    def effective_loss_scaling(self):
        raise NotImplementedError()

    def step_is_optimized(self, step: int, n_total_steps: int | None = None) -> bool:
        return step < 2

    def n_required_forward_steps(self) -> int:
        return 2

    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor:
        return self._loss_obj(prediction.data, target_data, prediction.step)


def assert_tensor_dicts_close(
    x: dict[str, torch.Tensor | None], y: dict[str, torch.Tensor | None]
):
    assert x.keys() == y.keys()
    for key in x.keys():
        if x[key] is None:
            assert y[key] is None
        torch.testing.assert_close(x[key], y[key])


def test_coupled_stepper_train_loss(steps_thru_atmos_7):
    ocean_loss_obj = _StepLoss(loss_obj=lambda *_, **__: torch.tensor(2.0))
    atmos_loss_obj = _StepLoss(loss_obj=lambda *_, **__: torch.tensor(1.0))
    loss_obj = CoupledStepperTrainLoss(
        ocean_loss=ocean_loss_obj, atmosphere_loss=atmos_loss_obj
    )
    metrics = {}
    for prediction, target_data in steps_thru_atmos_7:
        metrics[f"{prediction.realm}_{prediction.step}"] = loss_obj(
            prediction, target_data
        )
    expected_metrics = {
        "atmosphere_0": torch.tensor(1.0),
        "atmosphere_1": torch.tensor(1.0),
        "ocean_0": torch.tensor(2.0),
        "atmosphere_2": None,
        "atmosphere_3": None,
        "ocean_1": torch.tensor(2.0),
        "atmosphere_4": None,
        "atmosphere_5": None,
        "ocean_2": None,
        "atmosphere_6": None,
        "atmosphere_7": None,
    }
    assert_tensor_dicts_close(metrics, expected_metrics)


def test_loss_contributions(steps_thru_atmos_7):
    def mae_loss(gen, target, step: int):
        loss = torch.tensor(0.0)
        for key in gen:
            loss += (gen[key] - target[key]).abs().mean() / (step + 1)
        return loss

    def mae_loss_as_output(gen, target, step: int):
        return _wrap_as_loss_output(mae_loss(gen, target, step))

    atmos_loss_config = LossContributionsConfig(
        n_steps=6,
        weight=1 / 3,
    )
    mock_step_loss = Mock(spec=StepLoss, side_effect=mae_loss_as_output)
    atmosphere_loss = atmos_loss_config.build(
        loss_obj=mock_step_loss,
        time_dim=1,
        max_n_steps=20,
    )
    ocean_loss = _StepLoss(loss_obj=mae_loss)
    loss_obj = CoupledStepperTrainLoss(
        ocean_loss=ocean_loss,
        atmosphere_loss=atmosphere_loss,
    )
    metrics = {}
    expected_metrics: dict[str, torch.Tensor | None] = {}
    for prediction, target_data in steps_thru_atmos_7:
        label = f"{prediction.realm}_{prediction.step}"
        metrics[label] = loss_obj(prediction, target_data)
        if prediction.realm == "atmosphere":
            if prediction.step >= 6:
                expected_metrics[label] = None
            else:
                expected_metrics[label] = (
                    mae_loss(prediction.data, target_data, step=prediction.step) / 3
                )
        elif prediction.realm == "ocean":
            if prediction.step >= 2:
                expected_metrics[label] = None
            else:
                expected_metrics[label] = mae_loss(
                    prediction.data, target_data, step=prediction.step
                )
    assert_tensor_dicts_close(metrics, expected_metrics)


def test_loss_contributions_optimize_last_step_only(steps_thru_atmos_7):
    def mae_loss(gen, target, step: int):
        loss = torch.tensor(0.0)
        for key in gen:
            loss += (gen[key] - target[key]).abs().mean() / (step + 1)
        return loss

    def mae_loss_as_output(gen, target, step: int):
        return _wrap_as_loss_output(mae_loss(gen, target, step))

    n_total_atmos = 8
    n_total_ocean = 4
    atmos_loss_config = LossContributionsConfig(
        n_steps=6,
        weight=1 / 3,
        optimize_last_step_only=True,
    )
    mock_step_loss = Mock(spec=StepLoss, side_effect=mae_loss_as_output)
    atmosphere_loss = atmos_loss_config.build(
        loss_obj=mock_step_loss,
        time_dim=1,
        max_n_steps=n_total_atmos,
    )
    ocean_loss_config = LossContributionsConfig(
        n_steps=3,
        optimize_last_step_only=True,
    )
    ocean_loss = ocean_loss_config.build(
        loss_obj=Mock(spec=StepLoss, side_effect=mae_loss_as_output),
        time_dim=1,
        max_n_steps=n_total_ocean,
    )
    loss_obj = CoupledStepperTrainLoss(
        ocean_loss=ocean_loss,
        atmosphere_loss=atmosphere_loss,
    )
    metrics = {}
    expected_metrics: dict[str, torch.Tensor | None] = {}
    for prediction, target_data in steps_thru_atmos_7:
        label = f"{prediction.realm}_{prediction.step}"
        metrics[label] = loss_obj(prediction, target_data)
        if prediction.realm == "atmosphere":
            # n_steps=6, n_total=8 → last optimized step = min(6,8)-1 = 5
            if prediction.step == 5:
                expected_metrics[label] = (
                    mae_loss(prediction.data, target_data, step=prediction.step) / 3
                )
            else:
                expected_metrics[label] = None
        elif prediction.realm == "ocean":
            # n_steps=3, n_total=4 → last optimized step = min(3,4)-1 = 2
            if prediction.step == 2:
                expected_metrics[label] = mae_loss(
                    prediction.data, target_data, step=prediction.step
                )
            else:
                expected_metrics[label] = None
    assert_tensor_dicts_close(metrics, expected_metrics)


@pytest.mark.parametrize(
    "n_steps, n_total_steps, expected_optimized_step",
    [
        (6, 8, 5),
        (10, 8, 7),
        (float("inf"), 8, 7),
        (1, 1, 0),
        (3, 3, 2),
    ],
)
def test_step_is_optimized_last_step_only(
    n_steps, n_total_steps, expected_optimized_step
):
    config = LossContributionsConfig(n_steps=n_steps, optimize_last_step_only=True)
    loss = config.build(
        loss_obj=Mock(spec=StepLoss),
        time_dim=1,
        max_n_steps=n_total_steps,
    )
    for step in range(n_total_steps):
        result = loss.step_is_optimized(step)
        if step == expected_optimized_step:
            assert result, f"step {step} should be optimized"
        else:
            assert not result, f"step {step} should not be optimized"


def test_step_is_optimized_last_step_only_weight_zero():
    config = LossContributionsConfig(optimize_last_step_only=True, weight=0.0)
    loss = config.build(
        loss_obj=Mock(spec=StepLoss),
        time_dim=1,
        max_n_steps=5,
    )
    # weight=0 → NullLossContributions, always returns False
    assert not loss.step_is_optimized(0)


def test_stochastic_n_steps_sample_changes_step_is_optimized():
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=2, probability=1.0),
        ]
    )
    config = LossContributionsConfig(n_steps=sampler)
    loss = config.build(loss_obj=Mock(spec=StepLoss), time_dim=1, max_n_steps=4)
    # before sampling, _n_steps is max_n_forward_steps = 2
    assert loss.step_is_optimized(0)
    assert loss.step_is_optimized(1)
    assert not loss.step_is_optimized(2)

    # after sampling (deterministic: always 2), same behavior
    loss.sample_n_steps()
    assert loss.step_is_optimized(0)
    assert loss.step_is_optimized(1)
    assert not loss.step_is_optimized(2)


def test_stochastic_n_steps_deterministic_outcome():
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=3, probability=1.0),
        ]
    )
    config = LossContributionsConfig(n_steps=sampler)
    loss = config.build(loss_obj=Mock(spec=StepLoss), time_dim=1, max_n_steps=4)
    loss.sample_n_steps()
    assert loss.step_is_optimized(0)
    assert loss.step_is_optimized(1)
    assert loss.step_is_optimized(2)
    assert not loss.step_is_optimized(3)


def test_stochastic_n_steps_samples_vary():
    """With multiple outcomes, repeated sampling should eventually produce
    different effective n_steps values."""
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=1, probability=0.5),
            TimeLengthProbability(steps=4, probability=0.5),
        ]
    )
    config = LossContributionsConfig(n_steps=sampler)
    loss = config.build(loss_obj=Mock(spec=StepLoss), time_dim=1, max_n_steps=5)
    seen_optimized_step_3 = False
    seen_not_optimized_step_1 = False
    for _ in range(20):  # about 1 in a million prob of test failure
        loss.sample_n_steps()
        if loss.step_is_optimized(3):
            seen_optimized_step_3 = True
        if not loss.step_is_optimized(1):
            seen_not_optimized_step_1 = True
        if seen_optimized_step_3 and seen_not_optimized_step_1:
            break
    assert seen_optimized_step_3, "should sometimes sample n_steps=4"
    assert seen_not_optimized_step_1, "should sometimes sample n_steps=1"


class TestOptimizeLastStepOnlyStochastic:
    def _build(self, sampler, max_n_steps=6):
        config = LossContributionsConfig(n_steps=sampler, optimize_last_step_only=True)
        return config.build(
            loss_obj=Mock(spec=StepLoss), time_dim=1, max_n_steps=max_n_steps
        )

    def _sampler(self, outcomes):
        return TimeLengthProbabilities(
            outcomes=[
                TimeLengthProbability(steps=s, probability=p) for s, p in outcomes
            ]
        )

    def test_before_sampling(self):
        sampler = self._sampler([(2, 0.5), (5, 0.5)])
        loss = self._build(sampler, max_n_steps=6)
        # _n_steps = max_n_forward_steps = 5; last optimized = min(5,6)-1 = 4
        for step in range(6):
            if step == 4:
                assert loss.step_is_optimized(step), f"step {step} should be optimized"
            else:
                assert not loss.step_is_optimized(
                    step
                ), f"step {step} should not be optimized"

    def test_deterministic_sample(self):
        sampler = self._sampler([(3, 1.0)])
        loss = self._build(sampler, max_n_steps=6)
        loss.sample_n_steps()
        # _n_steps = 3; last optimized = min(3,6)-1 = 2
        for step in range(6):
            if step == 2:
                assert loss.step_is_optimized(step), f"step {step} should be optimized"
            else:
                assert not loss.step_is_optimized(
                    step
                ), f"step {step} should not be optimized"

    def test_varying_samples(self):
        sampler = self._sampler([(2, 0.5), (5, 0.5)])
        loss = self._build(sampler, max_n_steps=6)
        seen_step_1 = False  # min(2,6)-1 = 1
        seen_step_4 = False  # min(5,6)-1 = 4
        for _ in range(20):  # about 1 in a million prob of test failure
            loss.sample_n_steps()
            if loss.step_is_optimized(1) and not loss.step_is_optimized(4):
                seen_step_1 = True
            if loss.step_is_optimized(4) and not loss.step_is_optimized(1):
                seen_step_4 = True
            if seen_step_1 and seen_step_4:
                break
        assert seen_step_1, "should sometimes optimize only step 1 (n_steps=2)"
        assert seen_step_4, "should sometimes optimize only step 4 (n_steps=5)"


def test_sample_n_steps_noop_for_float_config():
    config = LossContributionsConfig(n_steps=5.0)
    loss = config.build(loss_obj=Mock(spec=StepLoss), time_dim=1, max_n_steps=5)
    loss.sample_n_steps()
    assert loss.step_is_optimized(4)
    assert not loss.step_is_optimized(5)


def test_coupled_stepper_train_loss_sample_n_steps_delegates():
    ocean_loss = MagicMock(spec=StepLossABC)
    atmos_loss = MagicMock(spec=StepLossABC)
    coupled_loss = CoupledStepperTrainLoss(
        ocean_loss=ocean_loss, atmosphere_loss=atmos_loss
    )
    coupled_loss.sample_n_steps()
    ocean_loss.sample_n_steps.assert_called_once()
    atmos_loss.sample_n_steps.assert_called_once()


@pytest.mark.parametrize(
    "n_steps, max_n_steps, expected",
    [
        (3, 10, 3),
        (10, 3, 3),
        (float("inf"), 5, 5),
        (5.0, 5, 5),
    ],
)
def test_loss_contributions_n_required_forward_steps(n_steps, max_n_steps, expected):
    config = LossContributionsConfig(n_steps=n_steps)
    loss = config.build(
        loss_obj=Mock(spec=StepLoss), time_dim=1, max_n_steps=max_n_steps
    )
    assert loss.n_required_forward_steps() == expected


@pytest.mark.parametrize(
    "n_steps, max_n_steps, expected",
    [
        (3, 10, 3),
        (10, 3, 3),
        (float("inf"), 5, 5),
    ],
)
def test_loss_contributions_n_required_forward_steps_optimize_last_step_only(
    n_steps, max_n_steps, expected
):
    config = LossContributionsConfig(n_steps=n_steps, optimize_last_step_only=True)
    loss = config.build(
        loss_obj=Mock(spec=StepLoss), time_dim=1, max_n_steps=max_n_steps
    )
    assert loss.n_required_forward_steps() == expected


def test_loss_contributions_n_required_forward_steps_after_sampling():
    sampler = TimeLengthProbabilities(
        outcomes=[TimeLengthProbability(steps=2, probability=1.0)]
    )
    config = LossContributionsConfig(n_steps=sampler)
    loss = config.build(loss_obj=Mock(spec=StepLoss), time_dim=1, max_n_steps=5)
    # before sampling, _n_steps == max_n_forward_steps == 2
    assert loss.n_required_forward_steps() == 2
    loss.sample_n_steps()
    assert loss.n_required_forward_steps() == 2


@pytest.mark.parametrize("config_kwargs", [{"n_steps": 0}, {"weight": 0.0}])
def test_null_loss_contributions_n_required_forward_steps(config_kwargs):
    config = LossContributionsConfig(**config_kwargs)
    loss = config.build(loss_obj=Mock(spec=StepLoss), time_dim=1, max_n_steps=10)
    assert loss.n_required_forward_steps() == 0


@pytest.mark.parametrize(
    "ocean_required, atmos_required, n_inner_steps, expected_outer",
    [
        (4, 0, 2, 4),
        (0, 5, 3, 2),
        (0, 6, 3, 2),
        (0, 7, 3, 3),
        (3, 5, 2, 3),
        (1, 8, 2, 4),
        (0, 0, 2, 0),
    ],
)
def test_coupled_stepper_train_loss_n_required_outer_steps(
    ocean_required, atmos_required, n_inner_steps, expected_outer
):
    ocean_loss = MagicMock(spec=StepLossABC)
    atmos_loss = MagicMock(spec=StepLossABC)
    ocean_loss.n_required_forward_steps.return_value = ocean_required
    atmos_loss.n_required_forward_steps.return_value = atmos_required
    coupled_loss = CoupledStepperTrainLoss(
        ocean_loss=ocean_loss, atmosphere_loss=atmos_loss
    )
    assert coupled_loss.n_required_outer_steps(n_inner_steps) == expected_outer


@pytest.mark.parametrize(
    "config_kwargs, expected_is_null",
    [
        ({}, False),
        ({"n_steps": 0}, True),
        ({"weight": 0.0}, True),
        ({"n_steps": 5.0}, False),
        ({"n_steps": 0.0}, True),
    ],
)
def test_loss_contributions_config_is_null(config_kwargs, expected_is_null):
    config = LossContributionsConfig(**config_kwargs)
    assert config.is_null is expected_is_null


@pytest.mark.parametrize("ocean_config_kwargs", [{"n_steps": 0}, {"weight": 0.0}])
def test_null_loss_contributions(steps_thru_atmos_7, ocean_config_kwargs):
    atmos_loss_config = LossContributionsConfig()
    atmosphere_loss = atmos_loss_config.build(
        loss_obj=Mock(
            spec=StepLoss,
            return_value=_wrap_as_loss_output(torch.tensor(5.25)),
        ),
        time_dim=1,
        max_n_steps=10,
    )
    ocean_loss_config = LossContributionsConfig(**ocean_config_kwargs)
    ocean_loss = ocean_loss_config.build(
        loss_obj=Mock(
            spec=StepLoss,
            return_value=_wrap_as_loss_output(torch.tensor(42.0)),
        ),
        time_dim=1,
        max_n_steps=10,
    )
    loss_obj = CoupledStepperTrainLoss(
        ocean_loss=ocean_loss,
        atmosphere_loss=atmosphere_loss,
    )
    for prediction, target_data in steps_thru_atmos_7:
        loss = loss_obj(prediction, target_data)
        if prediction.realm == "atmosphere":
            torch.testing.assert_close(loss, torch.tensor(5.25))
        elif prediction.realm == "ocean":
            assert loss is None
