import dataclasses


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
        # ensure milestones are sorted by epoch
        self.milestones.sort(key=lambda x: x.epoch)
        for i in range(1, len(self.milestones)):
            if self.milestones[i].epoch == self.milestones[i - 1].epoch:
                raise ValueError(
                    "Milestones must have unique epochs, "
                    f"but found more than one at epoch {self.milestones[i].epoch}"
                )

    def get_value(self, epoch: int) -> int:
        if len(self.milestones) == 0:
            return self.start_value
        if epoch < self.milestones[0].epoch:
            return self.start_value
        for i, milestone in enumerate(self.milestones):
            if epoch < milestone.epoch:
                return self.milestones[i - 1].value
        return self.milestones[-1].value

    def add(self, constant: int) -> "IntSchedule":
        new_milestones = [
            IntMilestone(epoch=milestone.epoch, value=milestone.value + constant)
            for milestone in self.milestones
        ]
        return IntSchedule(
            start_value=self.start_value + constant, milestones=new_milestones
        )
