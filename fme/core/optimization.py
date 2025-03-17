import contextlib
import dataclasses
import itertools
from typing import Any, Iterable, Literal, Mapping, Optional

import numpy as np
import torch
from torch import nn

from fme.core.device import get_device
from fme.core.generics.optimization import OptimizationABC
from fme.core.scheduler import SchedulerConfig
from fme.core.typing_ import TensorDict, TensorMapping


class Optimization(OptimizationABC):
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        optimizer_type: Literal["Adam", "FusedAdam"],
        lr: float,
        max_epochs: int,
        scheduler: SchedulerConfig,
        enable_automatic_mixed_precision: bool,
        kwargs: Mapping[str, Any],
        use_gradient_accumulation: bool = False,
    ):
        if optimizer_type == "FusedAdam":
            self.optimizer = torch.optim.AdamW(parameters, lr=lr, fused=True, **kwargs)
        elif optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(parameters, lr=lr, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        if enable_automatic_mixed_precision:
            self.gscaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda")
        else:
            self.gscaler = None
        self.scheduler = scheduler.build(self.optimizer, max_epochs)
        self._accumulated_loss = torch.tensor(0.0, device=get_device())
        self._use_gradient_accumulation = use_gradient_accumulation

    @contextlib.contextmanager
    def autocast(self):
        enabled = self.gscaler is not None
        dtype = torch.bfloat16 if enabled else None
        with torch.amp.autocast("cuda", enabled=enabled, dtype=dtype):
            yield

    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def set_mode(self, modules: nn.ModuleList):
        """
        Sets the mode of the module to train.
        """
        for m in modules:
            m.train()

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

    def detach_if_using_gradient_accumulation(self, state: TensorMapping) -> TensorDict:
        if self._use_gradient_accumulation:
            return {k: v.detach() for k, v in state.items()}
        return dict(state)

    def accumulate_loss(self, loss: torch.Tensor):
        self._validate_loss(loss)
        self._accumulated_loss += loss
        if self._use_gradient_accumulation:
            self._backward(loss)

    def get_accumulated_loss(self) -> torch.Tensor:
        return self._accumulated_loss

    def _backward(self, loss: torch.Tensor):
        if self.gscaler is not None:
            self.gscaler.scale(loss).backward()
        else:
            loss.backward()

    def _step_weights(self):
        if self.gscaler is not None:
            self.gscaler.step(self.optimizer)
        else:
            self.optimizer.step()

    def step_weights(self):
        if not self._use_gradient_accumulation:
            self._backward(self._accumulated_loss)
        self._step_weights()
        self.optimizer.zero_grad()
        if self.gscaler is not None:
            self.gscaler.update()
        self._accumulated_loss = torch.tensor(0.0, device=get_device())

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "gscaler_state_dict": (
                self.gscaler.state_dict() if self.gscaler is not None else None
            ),
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

    def _validate_loss(self, loss: torch.Tensor):
        with torch.no_grad():
            if torch.isnan(loss):
                raise ValueError("Loss is NaN-valued during training.")


@dataclasses.dataclass
class OptimizationConfig:
    """
    Configuration for optimization.

    Parameters:
        optimizer_type: The type of optimizer to use.
        lr: The learning rate.
        kwargs: Additional keyword arguments to pass to the optimizer.
        enable_automatic_mixed_precision: Whether to use automatic mixed
            precision.
        scheduler: The type of scheduler to use. If none is given, no scheduler
            will be used.
        use_gradient_accumulation: Whether to use gradient accumulation. This must be
            supported by the stepper being optimized, which may accumulate gradients
            from separate losses to reduce memory consumption. The stepper may choose
            to accumulate gradients differently when this is enabled, such as by
            detaching the computational graph between steps. See the documentation of
            your stepper (e.g. Stepper) for more details.
    """

    optimizer_type: Literal["Adam", "FusedAdam"] = "Adam"
    lr: float = 0.001
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    enable_automatic_mixed_precision: bool = False
    scheduler: SchedulerConfig = dataclasses.field(
        default_factory=lambda: SchedulerConfig()
    )
    use_gradient_accumulation: bool = False

    def build(self, modules: torch.nn.ModuleList, max_epochs: int) -> Optimization:
        parameters = itertools.chain(*[module.parameters() for module in modules])
        return Optimization(
            parameters=parameters,
            optimizer_type=self.optimizer_type,
            lr=self.lr,
            max_epochs=max_epochs,
            scheduler=self.scheduler,
            enable_automatic_mixed_precision=self.enable_automatic_mixed_precision,
            kwargs=self.kwargs,
            use_gradient_accumulation=self.use_gradient_accumulation,
        )

    def get_state(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "OptimizationConfig":
        return cls(**state)


class NullOptimization(OptimizationABC):
    def __init__(self):
        self._accumulated_loss = torch.tensor(0.0, device=get_device())

    @contextlib.contextmanager
    def autocast(self):
        yield

    @property
    def learning_rate(self) -> float:
        return float("nan")

    def step_scheduler(self, valid_loss: float):
        return

    def detach_if_using_gradient_accumulation(self, state: TensorMapping) -> TensorDict:
        return dict(state)

    def accumulate_loss(self, loss: torch.Tensor):
        self._accumulated_loss += loss

    def get_accumulated_loss(self) -> torch.Tensor:
        return self._accumulated_loss

    def step_weights(self):
        self._accumulated_loss = torch.tensor(0.0, device=get_device())
        return

    def get_state(self):
        return {}

    def load_state(self, state):
        return

    def set_mode(self, modules: nn.ModuleList):
        """
        Sets the mode of the module to eval.
        """
        for m in modules:
            m.eval()


@dataclasses.dataclass
class ActivationCheckpointingConfig:
    """
    Trade increased computation in exchange for lowered memory consumption during
    training by recomputing activations in the backward pass.

    after_n_forward_steps: Number of forward steps to generate before activation
        checkpointing is applied. Activation checkpointing is not used unless this
        number is less than the number of forward steps in the optimization.
    kwargs: Keyword arguments to pass to torch.utils.checkpoint.checkpoint. Note that
        use_reentrant=False is always explicitly passed as recommended by the docs.
    """

    after_n_forward_steps: float = np.inf
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
