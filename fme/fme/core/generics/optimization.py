import abc
import contextlib

import torch
from torch import nn


class OptimizationABC(abc.ABC):
    @contextlib.contextmanager
    @abc.abstractmethod
    def autocast(self): ...

    @property
    @abc.abstractmethod
    def learning_rate(self) -> float: ...

    @abc.abstractmethod
    def set_mode(self, module: nn.Module):
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
    def step_weights(self, loss: torch.Tensor): ...

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
