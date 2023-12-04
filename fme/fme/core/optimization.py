import contextlib
import dataclasses
from typing import Any, Literal, Mapping

import torch
from torch import nn

from fme.core.scheduler import SchedulerConfig


@dataclasses.dataclass
class DisabledOptimizationConfig:
    """
    Configuration for optimization, kept only for backwards compatibility when
    loading configuration. Cannot be used to build, will raise an exception.
    """

    optimizer_type: Literal["Adam", "FusedAdam"] = "Adam"
    lr: float = 0.001
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    enable_automatic_mixed_precision: bool = True
    scheduler: SchedulerConfig = dataclasses.field(
        default_factory=lambda: SchedulerConfig()
    )

    def build(self, parameters, max_epochs: int):
        raise RuntimeError("Cannot build DisabledOptimizationConfig")

    def get_state(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DisabledOptimizationConfig":
        return cls(**state)


class NullOptimization:
    @contextlib.contextmanager
    def autocast(self):
        yield

    @property
    def learning_rate(self) -> float:
        return float("nan")

    def step_scheduler(self, valid_loss: float):
        return

    def step_weights(self, loss: torch.Tensor):
        return

    def get_state(self):
        return {}

    def load_state(self, state):
        return

    def set_mode(self, module: nn.Module):
        """
        Sets the mode of the module to eval.
        """
        module.eval()
