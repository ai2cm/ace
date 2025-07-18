import dataclasses
from collections.abc import Mapping
from typing import Any

import torch.optim.lr_scheduler


@dataclasses.dataclass
class SchedulerConfig:
    """
    Configuration for a scheduler to use during training.

    Parameters:
        type: Name of scheduler class from torch.optim.lr_scheduler,
            no scheduler is used by default.
        kwargs: Keyword arguments to pass to the scheduler constructor.
    """

    type: str | None = None
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

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
