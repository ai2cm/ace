import numpy as np
import pytest

from fme.ace.stepper.time_length_probabilities import (
    TimeLengthMilestone,
    TimeLengthProbabilities,
    TimeLengthProbability,
    TimeLengthSchedule,
)


def test_time_length_probabilities_constant():
    sampler = TimeLengthProbabilities([TimeLengthProbability(10, 1.0)])
    assert sampler.sample() == 10


@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param(
            TimeLengthProbabilities(
                [TimeLengthProbability(10, 0.5), TimeLengthProbability(20, 0.5)]
            ),
            id="normalized",
        ),
        pytest.param(
            TimeLengthProbabilities(
                [TimeLengthProbability(10, 5), TimeLengthProbability(20, 5)]
            ),
            id="non_normalized",
        ),
    ],
)
def test_time_length_probabilities_random(sampler: TimeLengthProbabilities):
    np.random.seed(0)
    samples = [sampler.sample() for _ in range(20)]
    assert not all(sample == 10 for sample in samples)
    assert not all(sample == 20 for sample in samples)
    assert all(sample in [10, 20] for sample in samples)


def test_time_length_schedule_constant_int():
    schedule = TimeLengthSchedule.from_constant(5)
    assert schedule.get_value(0) == 5
    assert schedule.get_value(100) == 5


def test_time_length_schedule_constant_probabilities():
    probs = TimeLengthProbabilities([TimeLengthProbability(10, 1.0)])
    schedule = TimeLengthSchedule.from_constant(probs)
    assert schedule.get_value(0) is probs


def test_time_length_schedule_with_milestones():
    probs1 = TimeLengthProbabilities([TimeLengthProbability(5, 1.0)])
    schedule = TimeLengthSchedule(
        start_value=probs1, milestones=[TimeLengthMilestone(epoch=2, value=10)]
    )
    assert schedule.get_value(0) is probs1
    assert schedule.get_value(1) is probs1
    assert schedule.get_value(2) == 10
    assert schedule.get_value(5) == 10


def test_time_length_schedule_max_n_forward_steps():
    probs = TimeLengthProbabilities(
        [TimeLengthProbability(5, 0.5), TimeLengthProbability(10, 0.5)]
    )
    schedule = TimeLengthSchedule(
        start_value=probs, milestones=[TimeLengthMilestone(epoch=2, value=15)]
    )
    max_schedule = schedule.max_n_forward_steps
    assert max_schedule.get_value(0) == 10  # max of probabilities
    assert max_schedule.get_value(2) == 15


def test_time_length_schedule_sorts_milestones():
    schedule = TimeLengthSchedule(
        start_value=1,
        milestones=[
            TimeLengthMilestone(epoch=5, value=3),
            TimeLengthMilestone(epoch=2, value=2),
        ],
    )
    assert schedule.get_value(0) == 1
    assert schedule.get_value(2) == 2
    assert schedule.get_value(5) == 3
