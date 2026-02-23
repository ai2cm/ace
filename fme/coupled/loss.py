import abc
import dataclasses

import torch

from fme.ace.stepper.time_length_probabilities import TimeLengthProbabilities
from fme.core.device import get_device
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

    def sample_n_steps(self) -> None:
        """Sample a new effective n_steps for the current batch.

        No-op by default; override in subclasses that support stochastic
        n_steps via ``TimeLengthProbabilities``.
        """
        pass

    @abc.abstractmethod
    def step_is_optimized(self, step: int, n_total_steps: int | None = None) -> bool:
        """Returns True if the given step should contribute to the loss.

        Args:
            step: The step index to check.
            n_total_steps: The total number of steps for this component. Required
                when ``optimize_last_step_only`` is True.
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
        n_steps: The number of consecutive steps contributing to the loss,
            starting from the first. Can be a float (including ``inf`` for all
            steps) or a ``TimeLengthProbabilities`` for stochastic per-batch
            sampling.
        weight: (optional) Weight applied to each step loss for the given realm.
            Each step contributes equally to the total loss.
        optimize_last_step_only: If True, only the last step within the training
            horizon defined by ``n_steps`` is optimized (i.e. contributes to the
            loss and has gradients enabled). The optimized step index is
            ``min(n_steps, n_total_steps) - 1``.

    """

    n_steps: TimeLengthProbabilities | float = float("inf")
    weight: float = 1.0
    optimize_last_step_only: bool = False

    def build(
        self,
        loss_obj: StepLoss,
        time_dim: int,
    ) -> StepLossABC:
        if self.weight == 0.0:
            return NullLossContributions(loss_obj)
        if isinstance(self.n_steps, int | float) and self.n_steps == 0:
            return NullLossContributions(loss_obj)
        return LossContributions(
            n_steps=self.n_steps,
            weight=self.weight,
            optimize_last_step_only=self.optimize_last_step_only,
            loss_obj=loss_obj,
            time_dim=time_dim,
        )


class NullLossContributions(StepLossABC):
    """
    Loss that always returns zero and an empty dictionary regardless of inputs.

    """

    def __init__(
        self,
        loss_obj: StepLoss,
    ):
        self._loss = loss_obj

    @property
    def effective_loss_scaling(self) -> TensorDict:
        return self._loss.effective_loss_scaling

    def step_is_optimized(self, step: int, n_total_steps: int | None = None) -> bool:
        return False

    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor:
        return torch.tensor(0.0, device=get_device())


class LossContributions(StepLossABC):
    def __init__(
        self,
        n_steps: TimeLengthProbabilities | float,
        weight: float,
        optimize_last_step_only: bool,
        loss_obj: StepLoss,
        time_dim: int,
    ):
        self._loss = loss_obj
        if isinstance(n_steps, TimeLengthProbabilities):
            self._n_steps_sampler: TimeLengthProbabilities | None = n_steps
            self._n_steps: float = float(n_steps.max_n_forward_steps)
        else:
            self._n_steps_sampler = None
            self._n_steps = n_steps
        self._weight = weight
        self._optimize_last_step_only = optimize_last_step_only
        self._time_dim = time_dim

    def sample_n_steps(self) -> None:
        if self._n_steps_sampler is not None:
            self._n_steps = float(self._n_steps_sampler.sample())

    @property
    def effective_loss_scaling(self) -> TensorDict:
        return self._loss.effective_loss_scaling

    def step_is_optimized(self, step: int, n_total_steps: int | None = None) -> bool:
        """Returns True if the step should contribute to the loss.

        When ``optimize_last_step_only`` is False (default), returns True for
        steps ``0`` through ``n_steps - 1``. When True, returns True only for
        the step at index ``min(n_steps, n_total_steps) - 1``.
        """
        if self._weight == 0.0:
            return False
        if self._optimize_last_step_only:
            if n_total_steps is None:
                raise ValueError(
                    "n_total_steps is required when optimize_last_step_only is True"
                )
            last_optimized_step = min(self._n_steps, n_total_steps) - 1
            return step == last_optimized_step
        return step < self._n_steps

    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor:
        return self._weight * self._loss(prediction.data, target_data, prediction.step)
