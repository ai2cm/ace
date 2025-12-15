import dataclasses

import numpy as np

from fme.core.dataset.schedule import IntMilestone, IntSchedule, ValidatedMilestones
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
        self._rng = None

    def initialize_rng(self):
        """Set the rng at runtime. This helps guarantee that the distributed
        seed has already been set.

        """
        if self._rng is None:
            self._rng = np.random.RandomState(
                Distributed.get_instance().get_seed()
                + 684  # don't use this number anywhere else
            )  # must be the same across all processes

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
        self.initialize_rng()  # jit, if not called externally
        assert self._rng is not None
        return self._rng.choice(self._n_times, p=self._probabilities)


TimeLength = TimeLengthProbabilities | int


@dataclasses.dataclass
class TimeLengthMilestone:
    """
    A milestone for a time length schedule.
    """

    epoch: int
    value: TimeLength


@dataclasses.dataclass
class TimeLengthSchedule:
    """
    A schedule for a time length value.
    """

    start_value: TimeLength
    milestones: list[TimeLengthMilestone]

    def __post_init__(self):
        self._validated_milestones = ValidatedMilestones(
            start_value=self.start_value, milestones=self.milestones
        )

    @classmethod
    def from_constant(cls, value: TimeLength) -> "TimeLengthSchedule":
        """
        Create a TimeLengthSchedule that always returns the same value.

        Parameters:
            value: The constant value.

        Returns:
            A TimeLengthSchedule instance.
        """
        return cls(start_value=value, milestones=[])

    def get_value(self, epoch: int) -> TimeLength:
        return self._validated_milestones.get_value(epoch)

    @property
    def max_n_forward_steps(self) -> IntSchedule:
        """
        Get a schedule of the maximum number of forward steps.
        """
        if isinstance(self.start_value, int):
            max_start = self.start_value
        else:
            max_start = self.start_value.max_n_forward_steps
        max_milestones = []
        for milestone in self.milestones:
            if isinstance(milestone.value, int):
                max_value = milestone.value
            else:
                max_value = milestone.value.max_n_forward_steps
            max_milestones.append(IntMilestone(epoch=milestone.epoch, value=max_value))
        return IntSchedule(start_value=max_start, milestones=max_milestones)
