import dataclasses
from typing import Any, Literal, Mapping

import torch


@dataclasses.dataclass
class LossConfig:
    type: Literal["L1Loss"] = "L1Loss"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})

    def build(self) -> torch.nn.Module:
        return torch.nn.L1Loss(**self.kwargs)
