import contextlib
from typing import Any, Literal, Mapping, Optional
from fme.core.scheduler import SchedulerConfig

import torch.cuda.amp as amp
from apex import optimizers
import dataclasses
from torch import nn
import torch


class Optimization:
    def __init__(
        self,
        parameters,
        optimizer_type: Literal["Adam", "FusedAdam"],
        lr: float,
        scheduler: SchedulerConfig,
        enable_automatic_mixed_precision: bool,
    ):
        if optimizer_type == "FusedAdam":
            self.optimizer = optimizers.FusedAdam(parameters, lr=lr)
        elif optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(parameters, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        if enable_automatic_mixed_precision:
            self.gscaler: Optional[amp.GradScaler] = amp.GradScaler()
        else:
            self.gscaler = None
        self.scheduler = scheduler.build(self.optimizer)

    @contextlib.contextmanager
    def autocast(self):
        with amp.autocast(enabled=self.gscaler is not None):
            yield

    def set_mode(self, module: nn.Module):
        """
        Sets the mode of the module to train.
        """
        module.train()

    def step_scheduler(self, valid_loss: float):
        """
        Step the scheduler.

        Args:
            valid_loss: The validation loss. Used in schedulers which change the
                learning rate based on whether the validation loss is decreasing.
        """
        if self.scheduler is not None:
            try:
                self.scheduler.step(metrics=valid_loss)
            except TypeError:
                self.scheduler.step()

    def step_weights(self, loss: torch.Tensor):
        if self.gscaler is not None:
            self.gscaler.scale(loss).backward()
            self.gscaler.step(self.optimizer)
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        if self.gscaler is not None:
            self.gscaler.update()

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "gscaler_state_dict": self.gscaler.state_dict()
            if self.gscaler is not None
            else None,
        }
        return state

    def load_state(self, state):
        """
        Loads state from a serializable data structure.
        """
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.gscaler is not None:
            self.gscaler.load_state_dict(state["gscaler_state_dict"])


@dataclasses.dataclass
class OptimizationConfig:
    """
    Configuration for optimization.

    Attributes:
        optimizer_type: The type of optimizer to use.
        lr: The learning rate.
        enable_automatic_mixed_precision: Whether to use automatic mixed
            precision.
        scheduler: The type of scheduler to use. If none is given, no scheduler
            will be used.
    """

    optimizer_type: Literal["Adam", "FusedAdam"] = "Adam"
    lr: float = 0.001
    enable_automatic_mixed_precision: bool = True
    scheduler: SchedulerConfig = dataclasses.field(
        default_factory=lambda: SchedulerConfig()
    )

    def build(self, parameters, max_epochs: int) -> Optimization:
        return Optimization(
            parameters=parameters,
            optimizer_type=self.optimizer_type,
            lr=self.lr,
            scheduler=self.scheduler,
            enable_automatic_mixed_precision=self.enable_automatic_mixed_precision,
        )

    def get_state(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "OptimizationConfig":
        return cls(**state)


class NullOptimization:
    @contextlib.contextmanager
    def autocast(self):
        yield

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
