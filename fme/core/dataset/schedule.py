import dataclasses
from bisect import bisect_right
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class _Milestone(Protocol, Generic[T]):
    """
    A milestone for a generic schedule.
    """

    epoch: int
    value: T


class ValidatedMilestones(Generic[T]):
    def __init__(self, start_value: T, milestones: Sequence[_Milestone[T]]):
        if len(milestones) > 0:
            milestones = sorted(milestones, key=lambda x: x.epoch)
            if milestones[0].epoch <= 0:
                raise ValueError(
                    "The first milestone epoch must be greater than 0, "
                    f"got {milestones[0].epoch}"
                )
        self._values = [start_value] + [m.value for m in milestones]
        self._milestones = [0] + [m.epoch for m in milestones]
        for i in range(1, len(self._milestones)):
            if self._milestones[i] == self._milestones[i - 1]:
                raise ValueError(
                    "Milestones must have unique epochs, "
                    f"but found more than one at epoch {self._milestones[i]}"
                )

    def get_value(self, epoch: int) -> T:
        if epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}")
        idx = bisect_right(self._milestones, epoch) - 1
        return self._values[idx]


@dataclasses.dataclass
class IntMilestone:
    """
    A milestone for an integer schedule.
    """

    epoch: int
    value: int


@dataclasses.dataclass
class IntSchedule:
    """
    A schedule for an integer value.
    """

    start_value: int
    milestones: list[IntMilestone]

    def __post_init__(self):
        self._validated_milestones = ValidatedMilestones(
            start_value=self.start_value, milestones=self.milestones
        )

    @classmethod
    def from_constant(cls, value: int) -> "IntSchedule":
        """
        Create an IntSchedule that always returns the same value.

        Parameters:
            value: The constant value.

        Returns:
            An IntSchedule instance.
        """
        return cls(start_value=value, milestones=[])

    def get_value(self, epoch: int) -> int:
        return self._validated_milestones.get_value(epoch)

    @property
    def max_value(self) -> int:
        if len(self.milestones) == 0:
            return self.start_value
        return max(
            self.start_value, max(milestone.value for milestone in self.milestones)
        )

    def add(self, constant: int) -> "IntSchedule":
        new_milestones = [
            IntMilestone(epoch=milestone.epoch, value=milestone.value + constant)
            for milestone in self.milestones
        ]
        return IntSchedule(
            start_value=self.start_value + constant, milestones=new_milestones
        )
