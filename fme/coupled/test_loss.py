from collections.abc import Generator
from typing import Literal
from unittest.mock import MagicMock, Mock

import pytest
import torch

from fme.ace.stepper.time_length_probabilities import (
    TimeLengthProbabilities,
    TimeLengthProbability,
)
from fme.core.loss import LossOutput, StandardLoss, StepLoss
from fme.core.typing_ import EnsembleTensorDict, TensorMapping

from .loss import ComponentLossSchedule
from .stepper import (
    ComponentEnsembleStepPrediction,
    ComponentTrainingConfig,
    CoupledStepperTrainLoss,
    ObjectivePerBatchConfig,
)


def _wrap_as_loss_output(value: torch.Tensor) -> LossOutput:
    """Wrap a scalar tensor as a LossOutput for mocking StepLoss."""
    return LossOutput(
        losses=[StandardLoss(value.unsqueeze(0).unsqueeze(0))],
        channel_names=["mock"],
    )


def _mock_step_loss(fn):
    """Create a Mock(spec=StepLoss) whose forward returns LossOutput."""
    mock = Mock(spec=StepLoss)
    mock.side_effect = lambda data, target, step: _wrap_as_loss_output(
        fn(data, target, step)
    )
    return mock


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


def assert_tensor_dicts_close(
    x: dict[str, torch.Tensor | None], y: dict[str, torch.Tensor | None]
):
    assert x.keys() == y.keys()
    for key in x.keys():
        if x[key] is None:
            assert y[key] is None
        torch.testing.assert_close(x[key], y[key])


def _build_coupled_loss(
    ocean_loss: StepLoss,
    atmosphere_loss: StepLoss,
    ocean_n_steps=None,
    atmos_n_steps=None,
    ocean_weight=1.0,
    atmos_weight=1.0,
    ocean_optimize_last_step_only=False,
    atmos_optimize_last_step_only=False,
    ocean_n_steps_limit=10,
    atmos_n_steps_limit=10,
    objective_per_batch: ObjectivePerBatchConfig | None = None,
) -> CoupledStepperTrainLoss:
    ocean_schedule = ComponentLossSchedule(
        n_steps=ocean_n_steps,
        optimize_last_step_only=ocean_optimize_last_step_only,
        loss_weight=ocean_weight,
        n_steps_limit=ocean_n_steps_limit,
    )
    atmos_schedule = ComponentLossSchedule(
        n_steps=atmos_n_steps,
        optimize_last_step_only=atmos_optimize_last_step_only,
        loss_weight=atmos_weight,
        n_steps_limit=atmos_n_steps_limit,
    )
    return CoupledStepperTrainLoss(
        ocean_loss=ocean_loss,
        atmosphere_loss=atmosphere_loss,
        ocean_schedule=ocean_schedule,
        atmosphere_schedule=atmos_schedule,
        objective_weights=objective_per_batch,
    )


def test_coupled_stepper_train_loss(steps_thru_atmos_7):
    ocean_loss_obj = _mock_step_loss(lambda *_, **__: torch.tensor(2.0))
    atmos_loss_obj = _mock_step_loss(lambda *_, **__: torch.tensor(1.0))
    loss_obj = _build_coupled_loss(
        ocean_loss=ocean_loss_obj,
        atmosphere_loss=atmos_loss_obj,
        ocean_n_steps=2,
        atmos_n_steps=2,
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

    atmos_loss_obj = _mock_step_loss(mae_loss)
    ocean_loss_obj = _mock_step_loss(mae_loss)
    loss_obj = _build_coupled_loss(
        ocean_loss=ocean_loss_obj,
        atmosphere_loss=atmos_loss_obj,
        ocean_n_steps=2,
        atmos_n_steps=6,
        atmos_weight=1 / 3,
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
    atmos_loss_obj = _mock_step_loss(mae_loss)
    ocean_loss_obj = _mock_step_loss(mae_loss)
    loss_obj = _build_coupled_loss(
        ocean_loss=ocean_loss_obj,
        atmosphere_loss=atmos_loss_obj,
        ocean_n_steps=3,
        atmos_n_steps=6,
        atmos_weight=1 / 3,
        ocean_optimize_last_step_only=True,
        atmos_optimize_last_step_only=True,
        ocean_n_steps_limit=n_total_ocean,
        atmos_n_steps_limit=n_total_atmos,
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
    schedule = ComponentLossSchedule(
        n_steps=n_steps,
        optimize_last_step_only=True,
        loss_weight=1.0,
        n_steps_limit=n_total_steps,
    )
    for step in range(n_total_steps):
        result = schedule.step_is_optimized(step)
        if step == expected_optimized_step:
            assert result, f"step {step} should be optimized"
        else:
            assert not result, f"step {step} should not be optimized"


def test_step_is_optimized_last_step_only_weight_zero():
    schedule = ComponentLossSchedule(
        n_steps=None,
        optimize_last_step_only=True,
        loss_weight=0.0,
        n_steps_limit=5,
    )
    assert not schedule.step_is_optimized(0)


def test_stochastic_n_steps_sample_changes_step_is_optimized():
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=2, probability=1.0),
        ]
    )
    schedule = ComponentLossSchedule(
        n_steps=sampler,
        optimize_last_step_only=False,
        loss_weight=1.0,
        n_steps_limit=4,
    )
    # before sampling, _n_steps is max_n_forward_steps = 2
    assert schedule.step_is_optimized(0)
    assert schedule.step_is_optimized(1)
    assert not schedule.step_is_optimized(2)

    # after sampling (deterministic: always 2), same behavior
    schedule.sample_n_steps()
    assert schedule.step_is_optimized(0)
    assert schedule.step_is_optimized(1)
    assert not schedule.step_is_optimized(2)


def test_stochastic_n_steps_deterministic_outcome():
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=3, probability=1.0),
        ]
    )
    schedule = ComponentLossSchedule(
        n_steps=sampler,
        optimize_last_step_only=False,
        loss_weight=1.0,
        n_steps_limit=4,
    )
    schedule.sample_n_steps()
    assert schedule.step_is_optimized(0)
    assert schedule.step_is_optimized(1)
    assert schedule.step_is_optimized(2)
    assert not schedule.step_is_optimized(3)


def test_stochastic_n_steps_samples_vary():
    """With multiple outcomes, repeated sampling should eventually produce
    different effective n_steps values."""
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=1, probability=0.5),
            TimeLengthProbability(steps=4, probability=0.5),
        ]
    )
    schedule = ComponentLossSchedule(
        n_steps=sampler,
        optimize_last_step_only=False,
        loss_weight=1.0,
        n_steps_limit=5,
    )
    seen_optimized_step_3 = False
    seen_not_optimized_step_1 = False
    for _ in range(20):  # about 1 in a million prob of test failure
        schedule.sample_n_steps()
        if schedule.step_is_optimized(3):
            seen_optimized_step_3 = True
        if not schedule.step_is_optimized(1):
            seen_not_optimized_step_1 = True
        if seen_optimized_step_3 and seen_not_optimized_step_1:
            break
    assert seen_optimized_step_3, "should sometimes sample n_steps=4"
    assert seen_not_optimized_step_1, "should sometimes sample n_steps=1"


class TestOptimizeLastStepOnlyStochastic:
    def _build(self, sampler, n_steps_limit=6):
        return ComponentLossSchedule(
            n_steps=sampler,
            optimize_last_step_only=True,
            loss_weight=1.0,
            n_steps_limit=n_steps_limit,
        )

    def _sampler(self, outcomes):
        return TimeLengthProbabilities(
            outcomes=[
                TimeLengthProbability(steps=s, probability=p) for s, p in outcomes
            ]
        )

    def test_before_sampling(self):
        sampler = self._sampler([(2, 0.5), (5, 0.5)])
        schedule = self._build(sampler, n_steps_limit=6)
        # _n_steps = max_n_forward_steps = 5; last optimized = min(5,6)-1 = 4
        for step in range(6):
            if step == 4:
                assert schedule.step_is_optimized(
                    step
                ), f"step {step} should be optimized"
            else:
                assert not schedule.step_is_optimized(
                    step
                ), f"step {step} should not be optimized"

    def test_deterministic_sample(self):
        sampler = self._sampler([(3, 1.0)])
        schedule = self._build(sampler, n_steps_limit=6)
        schedule.sample_n_steps()
        # _n_steps = 3; last optimized = min(3,6)-1 = 2
        for step in range(6):
            if step == 2:
                assert schedule.step_is_optimized(
                    step
                ), f"step {step} should be optimized"
            else:
                assert not schedule.step_is_optimized(
                    step
                ), f"step {step} should not be optimized"

    def test_varying_samples(self):
        sampler = self._sampler([(2, 0.5), (5, 0.5)])
        schedule = self._build(sampler, n_steps_limit=6)
        seen_step_1 = False  # min(2,6)-1 = 1
        seen_step_4 = False  # min(5,6)-1 = 4
        for _ in range(20):  # about 1 in a million prob of test failure
            schedule.sample_n_steps()
            if schedule.step_is_optimized(1) and not schedule.step_is_optimized(4):
                seen_step_1 = True
            if schedule.step_is_optimized(4) and not schedule.step_is_optimized(1):
                seen_step_4 = True
            if seen_step_1 and seen_step_4:
                break
        assert seen_step_1, "should sometimes optimize only step 1 (n_steps=2)"
        assert seen_step_4, "should sometimes optimize only step 4 (n_steps=5)"


def test_sample_n_steps_noop_for_int_config():
    schedule = ComponentLossSchedule(
        n_steps=5,
        optimize_last_step_only=False,
        loss_weight=1.0,
        n_steps_limit=5,
    )
    schedule.sample_n_steps()
    assert schedule.step_is_optimized(4)
    assert not schedule.step_is_optimized(5)


def test_sample_n_steps_noop_for_none_config():
    schedule = ComponentLossSchedule(
        n_steps=None,
        optimize_last_step_only=False,
        loss_weight=1.0,
        n_steps_limit=5,
    )
    schedule.sample_n_steps()
    assert schedule.step_is_optimized(4)
    assert schedule.step_is_optimized(100)


def test_n_steps_max_property():
    assert (
        ComponentTrainingConfig(
            loss=Mock(),
        ).n_steps_max
        is None
    )
    assert (
        ComponentTrainingConfig(
            loss=Mock(),
            n_steps=None,
        ).n_steps_max
        is None
    )
    assert (
        ComponentTrainingConfig(
            loss=Mock(),
            n_steps=5,
        ).n_steps_max
        == 5
    )
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=2, probability=0.5),
            TimeLengthProbability(steps=4, probability=0.5),
        ]
    )
    assert (
        ComponentTrainingConfig(
            loss=Mock(),
            n_steps=sampler,
        ).n_steps_max
        == 4
    )


def test_coupled_stepper_train_loss_sample_n_steps_delegates():
    ocean_schedule = MagicMock(spec=ComponentLossSchedule)
    atmos_schedule = MagicMock(spec=ComponentLossSchedule)
    coupled_loss = CoupledStepperTrainLoss(
        ocean_loss=Mock(spec=StepLoss),
        atmosphere_loss=Mock(spec=StepLoss),
        ocean_schedule=ocean_schedule,
        atmosphere_schedule=atmos_schedule,
    )
    coupled_loss._sample_n_steps()
    ocean_schedule.sample_n_steps.assert_called_once()
    atmos_schedule.sample_n_steps.assert_called_once()


def test_seed_rng_does_not_corrupt_training_sampler():
    """seed_rng must only affect the eval sampler, leaving the training
    sampler's RNG untouched across eval/train cycles."""
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=1, probability=0.5),
            TimeLengthProbability(steps=4, probability=0.5),
        ]
    )
    loss_sched = ComponentLossSchedule(
        n_steps=sampler,
        loss_weight=1.0,
        optimize_last_step_only=False,
        n_steps_limit=5,
    )
    assert loss_sched._n_steps_sampler is not None
    loss_sched._n_steps_sampler.seed_rng(42)
    loss_sched.set_train()
    train_first_20 = []
    for _ in range(20):
        loss_sched.sample_n_steps()
        train_first_20.append(loss_sched._n_steps)
    loss_sched.set_eval()
    loss_sched.seed_rng(0)
    for _ in range(10):
        loss_sched.sample_n_steps()
    loss_sched.set_train()
    train_next_20 = []
    for _ in range(20):
        loss_sched.sample_n_steps()
        train_next_20.append(loss_sched._n_steps)
    loss_sched._n_steps_sampler.seed_rng(42)
    reference_40 = []
    for _ in range(40):
        loss_sched.sample_n_steps()
        reference_40.append(loss_sched._n_steps)
    assert train_first_20 + train_next_20 == reference_40


def test_coupled_stepper_train_loss_set_train_eval_delegates():
    ocean_schedule = MagicMock(spec=ComponentLossSchedule)
    atmos_schedule = MagicMock(spec=ComponentLossSchedule)
    coupled_loss = CoupledStepperTrainLoss(
        ocean_loss=MagicMock(spec=StepLoss),
        atmosphere_loss=MagicMock(spec=StepLoss),
        ocean_schedule=ocean_schedule,
        atmosphere_schedule=atmos_schedule,
    )
    coupled_loss.set_eval()
    ocean_schedule.set_eval.assert_called_once()
    atmos_schedule.set_eval.assert_called_once()
    coupled_loss.set_train()
    ocean_schedule.set_train.assert_called_once()
    atmos_schedule.set_train.assert_called_once()


@pytest.mark.parametrize(
    "n_steps, n_steps_limit, expected",
    [
        (3, 10, 3),
        (10, 3, 3),
        (float("inf"), 5, 5),
        (5.0, 5, 5),
    ],
)
def test_schedule_n_required_forward_steps(n_steps, n_steps_limit, expected):
    schedule = ComponentLossSchedule(
        n_steps=n_steps,
        optimize_last_step_only=False,
        loss_weight=1.0,
        n_steps_limit=n_steps_limit,
    )
    assert schedule.n_required_forward_steps() == expected


@pytest.mark.parametrize(
    "n_steps, n_steps_limit, expected",
    [
        (3, 10, 3),
        (10, 3, 3),
        (float("inf"), 5, 5),
    ],
)
def test_schedule_n_required_forward_steps_optimize_last_step_only(
    n_steps, n_steps_limit, expected
):
    schedule = ComponentLossSchedule(
        n_steps=n_steps,
        optimize_last_step_only=True,
        loss_weight=1.0,
        n_steps_limit=n_steps_limit,
    )
    assert schedule.n_required_forward_steps() == expected


def test_schedule_n_required_forward_steps_after_sampling():
    sampler = TimeLengthProbabilities(
        outcomes=[TimeLengthProbability(steps=2, probability=1.0)]
    )
    schedule = ComponentLossSchedule(
        n_steps=sampler,
        optimize_last_step_only=False,
        loss_weight=1.0,
        n_steps_limit=5,
    )
    # before sampling, _n_steps == max_n_forward_steps == 2
    assert schedule.n_required_forward_steps() == 2
    schedule.sample_n_steps()
    assert schedule.n_required_forward_steps() == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 0, "loss_weight": 1.0},
        {"n_steps": None, "loss_weight": 0.0},
    ],
)
def test_null_schedule_n_required_forward_steps(kwargs):
    schedule = ComponentLossSchedule(
        optimize_last_step_only=False,
        n_steps_limit=10,
        **kwargs,
    )
    assert schedule.n_required_forward_steps() == 0


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
    ocean_schedule = MagicMock(spec=ComponentLossSchedule)
    atmos_schedule = MagicMock(spec=ComponentLossSchedule)
    ocean_schedule.n_required_forward_steps.return_value = ocean_required
    atmos_schedule.n_required_forward_steps.return_value = atmos_required
    coupled_loss = CoupledStepperTrainLoss(
        ocean_loss=Mock(spec=StepLoss),
        atmosphere_loss=Mock(spec=StepLoss),
        ocean_schedule=ocean_schedule,
        atmosphere_schedule=atmos_schedule,
    )
    assert coupled_loss.n_required_outer_steps(n_inner_steps) == expected_outer


@pytest.mark.parametrize(
    "config_kwargs, expected_is_null",
    [
        ({}, False),
        ({"n_steps": 0}, True),
        ({"loss_weight": 0.0}, True),
        ({"n_steps": 5}, False),
        ({"n_steps": 0}, True),
    ],
)
def test_component_training_config_loss_is_null(config_kwargs, expected_is_null):
    config = ComponentTrainingConfig(loss=Mock(), **config_kwargs)
    assert config.loss_is_null is expected_is_null


@pytest.mark.parametrize(
    "ocean_kwargs",
    [
        {"ocean_n_steps": 0},
        {"ocean_weight": 0.0},
    ],
)
def test_null_loss_contributions(steps_thru_atmos_7, ocean_kwargs):
    atmos_loss_obj = _mock_step_loss(lambda *_, **__: torch.tensor(5.25))
    ocean_loss_obj = _mock_step_loss(lambda *_, **__: torch.tensor(42.0))
    loss_obj = _build_coupled_loss(
        ocean_loss=ocean_loss_obj,
        atmosphere_loss=atmos_loss_obj,
        ocean_n_steps_limit=10,
        atmos_n_steps_limit=10,
        **ocean_kwargs,
    )
    for prediction, target_data in steps_thru_atmos_7:
        loss = loss_obj(prediction, target_data)
        if prediction.realm == "atmosphere":
            torch.testing.assert_close(loss, torch.tensor(5.25))
        elif prediction.realm == "ocean":
            assert loss is None


# ---------------------------------------------------------------------------
# Per-batch objective draw (objective_per_batch)
# ---------------------------------------------------------------------------

_BOTH_REALMS = ObjectivePerBatchConfig(atmosphere=1.0, ocean=1.0)
_ZERO_STEPS = TimeLengthProbabilities(
    outcomes=[TimeLengthProbability(steps=0, probability=1.0)]
)


def _objective_loss(objective_per_batch=_BOTH_REALMS, **kwargs):
    """Build a coupled loss with both realms non-null (n_steps=2 each by
    default) and an objective draw, for objective tests."""
    ocean_loss_obj = _mock_step_loss(lambda *_, **__: torch.tensor(2.0))
    atmos_loss_obj = _mock_step_loss(lambda *_, **__: torch.tensor(1.0))
    kwargs.setdefault("ocean_n_steps", 2)
    kwargs.setdefault("atmos_n_steps", 2)
    return _build_coupled_loss(
        ocean_loss=ocean_loss_obj,
        atmosphere_loss=atmos_loss_obj,
        objective_per_batch=objective_per_batch,
        **kwargs,
    )


def _optimized_realms(loss_obj, n_steps_each=2):
    """Realms with at least one optimized step under the current objective."""
    return {
        realm
        for realm in ("ocean", "atmosphere")
        for step in range(n_steps_each)
        if loss_obj.step_is_optimized(realm, step)
    }


def _draw_objectives(loss_obj, n_draws):
    """Optimized-realm set of each of n_draws per-batch draws."""
    draws = []
    for _ in range(n_draws):
        loss_obj.sample_from_rng()
        draws.append(frozenset(_optimized_realms(loss_obj)))
    return draws


def test_objective_exactly_one_realm_optimized():
    loss_obj = _objective_loss()
    loss_obj.sample_from_rng()
    assert _optimized_realms(loss_obj) in ({"ocean"}, {"atmosphere"})


def test_objective_varies_across_draws():
    loss_obj = _objective_loss()
    draws = _draw_objectives(loss_obj, 50)  # ~1e-15 prob of missing a realm
    assert set(draws) == {frozenset({"ocean"}), frozenset({"atmosphere"})}


def test_objective_deterministic_for_fixed_seed():
    # Two losses built under the same (distributed) seed lazily initialize the
    # train objective RNG identically, so their draw sequences match.
    seq_a = _draw_objectives(_objective_loss(), 30)
    seq_b = _draw_objectives(_objective_loss(), 30)
    assert seq_a == seq_b
    assert len(set(seq_a)) == 2


@pytest.mark.parametrize(
    "atmosphere, ocean",
    [(1.0, 1.0), (3.0, 1.0), (1.0, 0.0), (0.0, 2.0)],
)
def test_objective_draw_follows_weights(atmosphere, ocean):
    loss_obj = _objective_loss(
        ObjectivePerBatchConfig(atmosphere=atmosphere, ocean=ocean)
    )
    n_draws = 4000
    draws = _draw_objectives(loss_obj, n_draws)
    assert all(len(draw) == 1 for draw in draws)
    atmosphere_fraction = draws.count(frozenset({"atmosphere"})) / n_draws
    expected = atmosphere / (atmosphere + ocean)
    # binomial std <= 0.008 at n_draws=4000; 0.03 is ~4 sigma
    assert atmosphere_fraction == pytest.approx(expected, abs=0.03)


def test_objective_null_realm_drops_out():
    # Ocean is null only for this batch (sampled n_steps 0 from a sampler that
    # is not statically null), so atmosphere takes the whole draw.
    loss_obj = _objective_loss(
        ObjectivePerBatchConfig(atmosphere=1.0, ocean=5.0),
        ocean_n_steps=_ZERO_STEPS,
        atmos_n_steps=2,
    )
    assert set(_draw_objectives(loss_obj, 20)) == {frozenset({"atmosphere"})}


def test_objective_no_eligible_objective_optimizes_nothing():
    # Only atmosphere is weighted, and its sampled window is empty this batch:
    # the non-null ocean schedule is still never optimized.
    loss_obj = _objective_loss(
        ObjectivePerBatchConfig(atmosphere=1.0),
        ocean_n_steps=2,
        atmos_n_steps=_ZERO_STEPS,
    )
    loss_obj.sample_from_rng()
    assert all(
        not loss_obj.step_is_optimized(realm, step)
        for realm in ("ocean", "atmosphere")
        for step in range(10)
    )


def test_objective_absent_matches_today(steps_thru_atmos_7):
    # Regression guard: no objective_per_batch reproduces
    # test_coupled_stepper_train_loss.
    loss_obj = _objective_loss(objective_per_batch=None)
    loss_obj.sample_from_rng()
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


def test_objective_active_in_eval():
    # In eval the objective STILL gates step_is_optimized to one realm (so the
    # validation loss follows the training mix), but compute_loss stays
    # ungated, preserving per-step diagnostics for the other realm.
    loss_obj = _objective_loss()
    loss_obj.set_eval()
    loss_obj.seed_rng(0)
    loss_obj.sample_from_rng()
    (selected,) = _optimized_realms(loss_obj)
    other: Literal["ocean", "atmosphere"] = (
        "ocean" if selected == "atmosphere" else "atmosphere"
    )
    assert loss_obj.step_is_optimized(selected, 0)
    assert not loss_obj.step_is_optimized(other, 0)
    other_pred = ComponentEnsembleStepPrediction(
        realm=other,
        data=EnsembleTensorDict({"x": torch.zeros(1, 1, 1, 1)}),
        step=0,
    )
    diag = loss_obj.compute_loss(other_pred, {"x": torch.zeros(1, 1, 1, 1)})
    expected = 2.0 if other == "ocean" else 1.0
    torch.testing.assert_close(diag, torch.tensor(expected))
    assert loss_obj(other_pred, {"x": torch.zeros(1, 1, 1, 1)}) is None


def test_objective_eval_reproducible_after_seed_eval():
    loss_obj = _objective_loss()

    def _eval_sequence():
        loss_obj.set_eval()
        loss_obj.seed_rng(0)
        return _draw_objectives(loss_obj, 30)

    first = _eval_sequence()
    # Advance the (free-running) train RNG to prove eval is independent of it.
    loss_obj.set_train()
    _draw_objectives(loss_obj, 7)
    second = _eval_sequence()
    assert first == second
    assert len(set(first)) == 2
