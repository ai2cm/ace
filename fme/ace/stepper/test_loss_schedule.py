import pytest

from fme.ace.stepper.loss_schedule import EpochNotProvidedError, LossSchedule
from fme.ace.stepper.time_length_probabilities import (
    TimeLengthMilestone,
    TimeLengthProbabilities,
    TimeLengthProbability,
    TimeLengthSchedule,
)


def test_no_schedule_no_sampler():
    schedule = LossSchedule(n_forward_steps_schedule=None)
    schedule.init_for_epoch(0)
    assert not schedule.has_sampler
    assert schedule.sample(5) == 5


def test_constant_schedule_creates_sampler():
    ts = TimeLengthSchedule.from_constant(3)
    schedule = LossSchedule(n_forward_steps_schedule=ts)
    schedule.init_for_epoch(0)
    assert schedule.has_sampler
    assert schedule.sample(5) == 3


def test_epoch_schedule_updates_on_epoch_change():
    ts = TimeLengthSchedule(
        start_value=2,
        milestones=[TimeLengthMilestone(epoch=1, value=4)],
    )
    schedule = LossSchedule(n_forward_steps_schedule=ts)
    schedule.init_for_epoch(0)
    assert schedule.sample(5) == 2

    schedule.init_for_epoch(1)
    assert schedule.sample(5) == 4


def test_epoch_not_provided_with_milestones():
    ts = TimeLengthSchedule(
        start_value=2,
        milestones=[TimeLengthMilestone(epoch=1, value=4)],
    )
    schedule = LossSchedule(n_forward_steps_schedule=ts)
    with pytest.raises(EpochNotProvidedError):
        schedule.init_for_epoch(None)


def test_epoch_none_without_milestones():
    ts = TimeLengthSchedule(start_value=2, milestones=[])
    schedule = LossSchedule(n_forward_steps_schedule=ts)
    schedule.init_for_epoch(None)


def test_sample_raises_when_exceeds_data():
    ts = TimeLengthSchedule.from_constant(10)
    schedule = LossSchedule(n_forward_steps_schedule=ts)
    schedule.init_for_epoch(0)
    with pytest.raises(RuntimeError, match="greater than"):
        schedule.sample(5)


def test_sample_dispatches_to_eval_sampler_after_set_eval():
    ts = TimeLengthSchedule(
        start_value=TimeLengthProbabilities(
            outcomes=[
                TimeLengthProbability(steps=5, probability=0.5),
                TimeLengthProbability(steps=10, probability=0.5),
            ]
        ),
        milestones=[],
    )
    schedule = LossSchedule(n_forward_steps_schedule=ts)
    schedule.init_for_epoch(0)
    assert schedule._train_sampler is not None
    assert schedule._eval_sampler is not None

    schedule._train_sampler.seed_rng(1)
    schedule._eval_sampler.seed_rng(2)
    expected_eval = [schedule._eval_sampler.sample() for _ in range(20)]

    schedule._train_sampler.seed_rng(1)
    schedule._eval_sampler.seed_rng(2)
    schedule.set_eval()
    sampled = [schedule.sample(10) for _ in range(20)]
    assert sampled == expected_eval


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
    schedule = LossSchedule(n_forward_steps_schedule=ts)
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
