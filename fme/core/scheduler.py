import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import SequentialLR


@dataclasses.dataclass
class SchedulerConfig:
    """
    Configuration for a scheduler to use during training.

    Parameters:
        type: Name of scheduler class from torch.optim.lr_scheduler,
            no scheduler is used by default.
        kwargs: Keyword arguments to pass to the scheduler constructor.
        steps_per_iteration: If true, step after each batch. Otherwise,
            just step at the end of each epoch. Schedulers that step with
            every iteration won't be passed the validation loss.
    """

    type: str | None = None
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    steps_per_iteration: bool = False

    def build(
        self, optimizer, max_epochs
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """
        Build the scheduler.
        """
        if self.type is None:
            return None

        build_kwargs = {**self.kwargs}
        # work-around so we don't need to specify T_max
        # in the yaml file for this scheduler
        if self.type == "CosineAnnealingLR" and "T_max" not in self.kwargs:
            build_kwargs["T_max"] = max_epochs

        scheduler_class = getattr(torch.optim.lr_scheduler, self.type)
        return scheduler_class(optimizer=optimizer, **build_kwargs)


@dataclasses.dataclass
class SequentialSchedulerConfig:
    """
    Configuration for using torch.optim.SequentialLR to build a sequence of LR
    schedulers that run one after the other.

    Parameters:
        schedulers: Ordered sequence of SchedulerConfigs to define the schedulers
            for the SequentialLR. Note that all schedulers in the sequence must
            have the same value for steps_per_iteration.
        milestones: Sequence of integers that reflects milestone points, where
            milestones[i] corresponds to the last epoch or iteration where
            schedulers[i] is active before switching to schedulers[i+1]. For example,
            with two schedulers and milestones=[10] the first 10 epochs will use the
            first scheduler and then switch to the second scheduler for epoch 11.
        last_epoch: The index of last epoch. Default: -1.
    """

    schedulers: Sequence[SchedulerConfig]
    milestones: Sequence[int]
    last_epoch: int = -1

    def __post_init__(self):
        valid_steps_per_iteration = all(
            [
                x.steps_per_iteration == self.schedulers[0].steps_per_iteration
                for x in self.schedulers
            ]
        )
        if not valid_steps_per_iteration:
            raise ValueError(
                "All SchedulerConfigs in the SequentialSchedulerConfig must have "
                "identical values for steps_per_iteration."
            )

    @property
    def steps_per_iteration(self) -> bool:
        return self.schedulers[0].steps_per_iteration

    def build(
        self, optimizer, max_epochs
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """
        Build the SequentialLR scheduler.
        """
        schedulers = [x.build(optimizer, max_epochs) for x in self.schedulers]
        return SequentialLR(
            optimizer=optimizer,
            schedulers=schedulers,
            milestones=self.milestones,
            last_epoch=self.last_epoch,
        )
