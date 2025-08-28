import dataclasses

import numpy as np

from fme.core.distributed import Distributed


@dataclasses.dataclass
class TimeLengthProbability:
    steps: int
    probability: float


@dataclasses.dataclass
class TimeLengthProbabilities:
    outcomes: list[TimeLengthProbability]

    def __post_init__(self):
        self._n_times = np.asarray([p.steps for p in self.outcomes])
        self._probabilities = np.asarray(
            [p.probability for p in self.outcomes], dtype=np.float64
        )
        self._probabilities /= self._probabilities.sum()
        self._max_n_timesteps = int(max(self._n_times))
        self._rng = np.random.RandomState(
            Distributed.get_instance().get_seed()
            + 684  # don't use this number anywhere else
        )  # must be the same across all processes
        self.sample()  # check for errors

    @classmethod
    def from_constant(cls, n_steps: int) -> "TimeLengthProbabilities":
        return cls(outcomes=[TimeLengthProbability(steps=n_steps, probability=1.0)])

    @property
    def max_n_forward_steps(self) -> int:
        return int(max(self._n_times))

    def sample(self) -> int:
        """
        Update the current number of timesteps to sample based on
        the probabilities of sampling each number of timesteps.
        """
        return self._rng.choice(self._n_times, p=self._probabilities)
