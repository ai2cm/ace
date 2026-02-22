from collections.abc import Callable, Generator
from unittest.mock import Mock

import pytest
import torch

from fme.core.loss import StepLoss
from fme.core.typing_ import TensorMapping

from .loss import LossContributionsConfig, StepLossABC, StepPredictionABC
from .stepper import ComponentStepPrediction, CoupledStepperTrainLoss


def step_and_target_gen(
    n_atmos_per_ocean=2,
) -> Generator[tuple[ComponentStepPrediction, TensorMapping], None, None]:
    torch.manual_seed(0)
    atmos_step = 0
    ocean_step = 0
    while True:
        yield (
            ComponentStepPrediction(
                realm="atmosphere",
                data={"a": torch.rand(1, 1, 3), "b": torch.rand(1, 1, 3)},
                step=atmos_step,
            ),
            {"a": torch.rand(1, 1, 3), "b": torch.zeros(1, 1, 3)},
        )
        if atmos_step % n_atmos_per_ocean == 1:
            yield (
                ComponentStepPrediction(
                    realm="ocean",
                    data={"o": torch.rand(1, 1, 3), "c": torch.rand(1, 1, 3)},
                    step=ocean_step,
                ),
                {"o": torch.rand(1, 1, 3), "c": torch.zeros(1, 1, 3)},
            )
            ocean_step += 1
        atmos_step += 1


@pytest.fixture(scope="module")
def steps_thru_atmos_7() -> list[tuple[ComponentStepPrediction, TensorMapping]]:
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

    atmos_loss_config = LossContributionsConfig(
        n_steps=6,
        weight=1 / 3,
    )
    mock_step_loss = Mock(spec=StepLoss, side_effect=mae_loss)
    atmosphere_loss = atmos_loss_config.build(
        loss_obj=mock_step_loss,
        time_dim=1,
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

    n_total_atmos = 8
    n_total_ocean = 4
    atmos_loss_config = LossContributionsConfig(
        n_steps=6,
        weight=1 / 3,
        optimize_last_step_only=True,
    )
    mock_step_loss = Mock(spec=StepLoss, side_effect=mae_loss)
    atmosphere_loss = atmos_loss_config.build(
        loss_obj=mock_step_loss,
        time_dim=1,
    )
    ocean_loss_config = LossContributionsConfig(
        n_steps=3,
        optimize_last_step_only=True,
    )
    ocean_loss = ocean_loss_config.build(
        loss_obj=Mock(spec=StepLoss, side_effect=mae_loss),
        time_dim=1,
    )
    loss_obj = CoupledStepperTrainLoss(
        ocean_loss=ocean_loss,
        atmosphere_loss=atmosphere_loss,
    )
    metrics = {}
    expected_metrics: dict[str, torch.Tensor | None] = {}
    for prediction, target_data in steps_thru_atmos_7:
        label = f"{prediction.realm}_{prediction.step}"
        if prediction.realm == "atmosphere":
            n_total = n_total_atmos
        else:
            n_total = n_total_ocean
        metrics[label] = loss_obj(prediction, target_data, n_total_steps=n_total)
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
    )
    for step in range(n_total_steps):
        result = loss.step_is_optimized(step, n_total_steps)
        if step == expected_optimized_step:
            assert result, f"step {step} should be optimized"
        else:
            assert not result, f"step {step} should not be optimized"


def test_step_is_optimized_last_step_only_requires_n_total_steps():
    config = LossContributionsConfig(optimize_last_step_only=True)
    loss = config.build(
        loss_obj=Mock(spec=StepLoss),
        time_dim=1,
    )
    with pytest.raises(ValueError, match="n_total_steps is required"):
        loss.step_is_optimized(0)


def test_step_is_optimized_last_step_only_weight_zero():
    config = LossContributionsConfig(optimize_last_step_only=True, weight=0.0)
    loss = config.build(
        loss_obj=Mock(spec=StepLoss),
        time_dim=1,
    )
    # weight=0 → NullLossContributions, always returns False
    assert not loss.step_is_optimized(0, n_total_steps=5)


@pytest.mark.parametrize("ocean_config_kwargs", [{"n_steps": 0}, {"weight": 0.0}])
def test_null_loss_contributions(steps_thru_atmos_7, ocean_config_kwargs):
    # test LossContributionsConfig with n_steps = 0
    atmos_loss_config = LossContributionsConfig()
    atmosphere_loss = atmos_loss_config.build(
        loss_obj=Mock(spec=StepLoss, return_value=torch.tensor(5.25)),
        time_dim=1,
    )
    ocean_loss_config = LossContributionsConfig(**ocean_config_kwargs)
    ocean_loss = ocean_loss_config.build(
        loss_obj=Mock(spec=StepLoss, return_value=torch.tensor(42.0)),
        time_dim=1,
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
