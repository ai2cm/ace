import abc
import contextlib

import torch
from torch import nn

from fme.core.typing_ import TensorDict, TensorMapping


class OptimizationABC(abc.ABC):
    @contextlib.contextmanager
    @abc.abstractmethod
    def autocast(self): ...

    @property
    @abc.abstractmethod
    def learning_rate(self) -> float: ...

    @abc.abstractmethod
    def set_mode(self, modules: nn.ModuleList):
        """
        Sets the mode of the module to train.
        """
        ...

    @abc.abstractmethod
    def step_scheduler(self, valid_loss: float):
        """
        Step the scheduler.

        Args:
            valid_loss: The validation loss. Used in schedulers which change the
                learning rate based on whether the validation loss is decreasing.
        """
        ...

    @abc.abstractmethod
    def detach_if_using_gradient_accumulation(self, state: TensorMapping) -> TensorDict:
        """
        Detaches the state if using gradient accumulation.
        """
        ...

    @abc.abstractmethod
    def accumulate_loss(self, loss: torch.Tensor):
        """
        Accumulate the loss.

        In order to support gradient accumulation, loss values accumulated after
        a call to `accumulate_loss` must not depend on any parameter interactions that
        occurred before the call. For example, if this is called at the end of a model
        step, later steps must have their graph detached from previous timesteps. This
        can be done only when gradient accumulation is enabled using the
        `detach_if_using_gradient_accumulation` method on this object.
        """
        ...

    @abc.abstractmethod
    def checkpoint(self, module: nn.Module, step: int) -> nn.Module:
        """
        Applies activation checkpointing to the module if configured to do so,
        otherwise returns the module unchanged.

        Args:
            module: The module to checkpoint.
            step: The current step number.
        """
        ...

    @abc.abstractmethod
    def get_accumulated_loss(self) -> torch.Tensor:
        """
        Get the accumulated loss.
        """
        ...

    @abc.abstractmethod
    def step_weights(self):
        """
        Step the weights.

        Resets the accumulated loss to zero.
        """
        ...

    @abc.abstractmethod
    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        ...

    @abc.abstractmethod
    def load_state(self, state):
        """
        Loads state from a serializable data structure.
        """
        ...
