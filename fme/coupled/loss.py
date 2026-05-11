import abc

import torch

from fme.core.loss import StepLoss
from fme.core.typing_ import TensorDict, TensorMapping


class StepPredictionABC(abc.ABC):
    @property
    @abc.abstractmethod
    def data(self) -> TensorMapping: ...

    @property
    @abc.abstractmethod
    def step(self) -> int: ...


class StepLossABC(abc.ABC):
    """
    Abstract base class for step loss functions.

    """

    @property
    @abc.abstractmethod
    def effective_loss_scaling(self) -> TensorDict: ...

    @abc.abstractmethod
    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor:
        """Computes the loss for a given prediction and target data."""
        ...


class StepLossAdapter(StepLossABC):
    """Adapts a ``StepLoss`` (core) into the coupled ``StepLossABC`` interface."""

    def __init__(self, loss_obj: StepLoss):
        self._loss = loss_obj

    @property
    def effective_loss_scaling(self) -> TensorDict:
        return self._loss.effective_loss_scaling

    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor:
        loss_output = self._loss(prediction.data, target_data, prediction.step)
        return loss_output.total()
