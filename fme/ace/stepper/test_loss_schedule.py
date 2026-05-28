import pytest

from fme.ace.stepper.loss_schedule import EpochNotProvidedError, LossSchedule
from fme.ace.stepper.time_length_probabilities import (
    TimeLengthMilestone,
    TimeLengthProbabilities,
    TimeLengthProbability,
    TimeLengthSchedule,
)


def test_no_schedule_no_sampler():
    schedule = LossSchedule(
        n_forward_steps_schedule=None, optimize_last_step_only=False
    )
    schedule.init_for_epoch(0)
    assert not schedule.has_sampler
    schedule.sample(5)
    assert schedule.n_loss_steps == 5


def test_constant_schedule_creates_sampler():
    ts = TimeLengthSchedule.from_constant(3)
    schedule = LossSchedule(n_forward_steps_schedule=ts, optimize_last_step_only=False)
    schedule.init_for_epoch(0)
    assert schedule.has_sampler
    schedule.sample(5)
    assert schedule.n_loss_steps == 3


def test_epoch_schedule_updates_on_epoch_change():
    ts = TimeLengthSchedule(
        start_value=2,
        milestones=[TimeLengthMilestone(epoch=1, value=4)],
    )
    schedule = LossSchedule(n_forward_steps_schedule=ts, optimize_last_step_only=False)
    schedule.init_for_epoch(0)
    schedule.sample(5)
    assert schedule.n_loss_steps == 2

    schedule.init_for_epoch(1)
    schedule.sample(5)
    assert schedule.n_loss_steps == 4


def test_epoch_not_provided_with_milestones():
    ts = TimeLengthSchedule(
        start_value=2,
        milestones=[TimeLengthMilestone(epoch=1, value=4)],
    )
    schedule = LossSchedule(n_forward_steps_schedule=ts, optimize_last_step_only=False)
    with pytest.raises(EpochNotProvidedError):
        schedule.init_for_epoch(None)


def test_epoch_none_without_milestones():
    ts = TimeLengthSchedule(start_value=2, milestones=[])
    schedule = LossSchedule(n_forward_steps_schedule=ts, optimize_last_step_only=False)
    schedule.init_for_epoch(None)


def test_sample_raises_when_exceeds_data():
    ts = TimeLengthSchedule.from_constant(10)
    schedule = LossSchedule(n_forward_steps_schedule=ts, optimize_last_step_only=False)
    schedule.init_for_epoch(0)
    with pytest.raises(RuntimeError, match="greater than"):
        schedule.sample(5)


def test_step_is_optimized_all_steps():
    schedule = LossSchedule(
        n_forward_steps_schedule=None, optimize_last_step_only=False
    )
    schedule.init_for_epoch(0)
    schedule.sample(4)
    assert all(schedule.step_is_optimized(i) for i in range(4))
    assert not schedule.step_is_optimized(4)


def test_step_is_optimized_last_only():
    schedule = LossSchedule(n_forward_steps_schedule=None, optimize_last_step_only=True)
    schedule.init_for_epoch(0)
    schedule.sample(4)
    assert not schedule.step_is_optimized(0)
    assert not schedule.step_is_optimized(2)
    assert schedule.step_is_optimized(3)


def test_n_forward_steps_evaluate_all():
    ts = TimeLengthSchedule.from_constant(3)
    schedule = LossSchedule(n_forward_steps_schedule=ts, optimize_last_step_only=False)
    schedule.init_for_epoch(0)
    schedule.sample(5)
    assert schedule.n_forward_steps(5, evaluate_all_steps=True) == 5
    assert schedule.n_forward_steps(5, evaluate_all_steps=False) == 3


def test_n_forward_steps_no_sampler():
    schedule = LossSchedule(
        n_forward_steps_schedule=None, optimize_last_step_only=False
    )
    schedule.init_for_epoch(0)
    schedule.sample(5)
    assert schedule.n_forward_steps(5, evaluate_all_steps=True) == 5
    assert schedule.n_forward_steps(5, evaluate_all_steps=False) == 5


def test_train_eval_samplers_are_independent():
    ts = TimeLengthSchedule(
        start_value=TimeLengthProbabilities(
            outcomes=[
                TimeLengthProbability(steps=5, probability=0.5),
                TimeLengthProbability(steps=10, probability=0.5),
            ]
        ),
        milestones=[],
    )
    schedule = LossSchedule(n_forward_steps_schedule=ts, optimize_last_step_only=False)
    schedule.init_for_epoch(0)
    assert schedule._train_sampler is not None
    schedule._train_sampler.seed_rng(42)
    train_before = [schedule._train_sampler.sample() for _ in range(20)]

    schedule.set_eval()
    schedule.seed_eval(seed=0)
    [schedule._eval_sampler.sample() for _ in range(10)]  # type: ignore

    schedule.set_train()
    schedule._train_sampler.seed_rng(42)
    train_after = [schedule._train_sampler.sample() for _ in range(20)]
    assert train_before == train_after


def test_n_loss_steps_raises_before_sample():
    schedule = LossSchedule(
        n_forward_steps_schedule=None, optimize_last_step_only=False
    )
    schedule.init_for_epoch(0)
    with pytest.raises(RuntimeError, match="Must call sample"):
        _ = schedule.n_loss_steps
