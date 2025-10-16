import numpy as np
import pytest

from fme.ace.stepper.time_length_probabilities import (
    TimeLengthProbabilities,
    TimeLengthProbability,
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
