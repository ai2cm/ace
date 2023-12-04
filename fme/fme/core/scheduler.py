import dataclasses
from typing import Any, Mapping, Optional

import torch.optim.lr_scheduler


@dataclasses.dataclass
class SchedulerConfig:
    """
    Configuration for a scheduler to use during training.

    Attributes:
        type: Name of scheduler class from torch.optim.lr_scheduler,
            no scheduler is used by default.
        kwargs: Keyword arguments to pass to the scheduler constructor.
    """

    type: Optional[str] = None
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def build(
        self, optimizer, max_epochs
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Build the scheduler.
        """
        raise NotImplementedError()
