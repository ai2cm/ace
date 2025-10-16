import abc
import dataclasses
from collections.abc import Callable

import torch

from fme.core.device import get_device
from fme.core.typing_ import TensorMapping


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

    @abc.abstractmethod
    def step_is_optimized(self, step: int) -> bool:
        """Returns True if the step is less than to the number of
        steps contributing to the loss.
        """
        ...

    @abc.abstractmethod
    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor:
        """Computes the loss for a given prediction and target data."""
        ...


@dataclasses.dataclass
class LossContributionsConfig:
    """
    Configuration for loss contributions.

    Parameters:
        n_steps: (optional) The number of consecutive steps contributing to the loss,
            starting from the first.
        weight: (optional) Weight applied to each step loss for the given realm.
            Each step contributes equally to the total loss.

    """

    n_steps: float = float("inf")
    weight: float = 1.0

    def build(
        self,
        loss_obj: Callable[[TensorMapping, TensorMapping, int], torch.Tensor],
        time_dim: int,
    ) -> StepLossABC:
        if self.n_steps == 0 or self.weight == 0.0:
            return NullLossContributions()
        return LossContributions(
            n_steps=self.n_steps,
            weight=self.weight,
            loss_obj=loss_obj,
            time_dim=time_dim,
        )


class NullLossContributions(StepLossABC):
    """
    Loss that always returns zero and an empty dictionary regardless of inputs.

    """

    def step_is_optimized(self, step: int) -> bool:
        return False

    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor:
        return torch.tensor(0.0, device=get_device())


class LossContributions(StepLossABC):
    def __init__(
        self,
        n_steps: float,
        weight: float,
        loss_obj: Callable[[TensorMapping, TensorMapping, int], torch.Tensor],
        time_dim: int,
    ):
        self._loss = loss_obj
        self._n_steps = n_steps
        self._weight = weight
        self._time_dim = time_dim

    def step_is_optimized(self, step: int) -> bool:
        """Returns True if the step is less than to the number of steps and
        weight is != 0. The first step number is assumed to be 0.

        """
        return step < self._n_steps and self._weight != 0.0

    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor:
        if self.step_is_optimized(prediction.step):
            return self._weight * self._loss(
                prediction.data, target_data, prediction.step
            )
        return torch.tensor(0.0, device=get_device())
