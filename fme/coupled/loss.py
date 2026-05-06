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
    def step_is_optimized(self, step: int) -> bool:
        """Returns True if the given step should contribute to the loss.

        Args:
            step: The step index to check.
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
            starting from the first. Can be an int, ``None`` (the default,
            meaning all available steps), or a ``TimeLengthProbabilities`` for
            stochastic per-batch sampling.
        weight: (optional) Weight applied to each step loss for the given realm.
            Each step contributes equally to the total loss.
        optimize_last_step_only: If True, only the last step within the training
            horizon defined by ``n_steps`` is optimized (i.e. contributes to the
            loss and has gradients enabled). The optimized step index is
            ``min(n_steps, n_total_steps) - 1``.

    """

    n_steps: TimeLengthProbabilities | int | None = None
    weight: float = 1.0
    optimize_last_step_only: bool = False

    @property
    def is_null(self) -> bool:
        """True when this config produces a ``NullLossContributions`` (i.e.
        contributes nothing to the loss).
        """
        if self.weight == 0.0:
            return True
        return self.n_steps == 0

    @property
    def n_steps_max(self) -> int | None:
        """Upper bound on the number of consecutive steps that can contribute
        to the loss, or ``None`` if unbounded (``n_steps=None``).

        For ``TimeLengthProbabilities`` this is the largest value the sampler
        can produce.
        """
        if self.n_steps is None:
            return None
        if isinstance(self.n_steps, TimeLengthProbabilities):
            return self.n_steps.max_n_forward_steps
        return self.n_steps

    def build(
        self,
        loss_obj: StepLoss,
        time_dim: int,
        n_steps_limit: int,
    ) -> StepLossABC:
        """Build the ``StepLossABC`` for this configuration.

        Args:
            loss_obj: The underlying step loss applied at each optimized step.
            time_dim: Time dimension index of the prediction tensors.
            n_steps_limit: The total number of forward steps available in the
                training window (i.e. the upper bound imposed by the rollout
                length, distinct from ``self.n_steps_max`` which is the
                user-configured bound on this config). The effective number of
                optimized steps is ``min(self.n_steps_max, n_steps_limit)`` when
                ``self.n_steps_max`` is not ``None``, else ``n_steps_limit``.
        """
        if self.is_null:
            return NullLossContributions(loss_obj)
        return LossContributions(
            n_steps=self.n_steps,
            weight=self.weight,
            optimize_last_step_only=self.optimize_last_step_only,
            loss_obj=loss_obj,
            time_dim=time_dim,
            n_steps_limit=n_steps_limit,
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

    def step_is_optimized(self, step: int) -> bool:
        return False

    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor:
        return torch.tensor(0.0, device=get_device())


class LossContributions(StepLossABC):
    def __init__(
        self,
        n_steps: TimeLengthProbabilities | int | None,
        weight: float,
        optimize_last_step_only: bool,
        loss_obj: StepLoss,
        time_dim: int,
        n_steps_limit: int,
    ):
        self._loss = loss_obj
        if isinstance(n_steps, TimeLengthProbabilities):
            self._n_steps_sampler: TimeLengthProbabilities | None = n_steps
            self._n_steps: float = float(n_steps.max_n_forward_steps)
        else:
            # Coalesce ``None`` to ``inf`` so downstream arithmetic
            # (``min(...)``, ``step < self._n_steps``) handles the unbounded
            # case without explicit branches.
            self._n_steps_sampler = None
            self._n_steps = float("inf") if n_steps is None else float(n_steps)
        self._weight = weight
        self._optimize_last_step_only = optimize_last_step_only
        self._time_dim = time_dim
        self._n_steps_limit = n_steps_limit

    def sample_n_steps(self) -> None:
        if self._n_steps_sampler is not None:
            self._n_steps = float(self._n_steps_sampler.sample())

    @property
    def effective_loss_scaling(self) -> TensorDict:
        return self._loss.effective_loss_scaling

    def step_is_optimized(self, step: int) -> bool:
        """Returns True if the step should contribute to the loss.

        When ``optimize_last_step_only`` is False (default), returns True for
        steps ``0`` through ``n_steps - 1``. When True, returns True only for
        the step at index ``min(n_steps, n_total_steps) - 1``.
        """
        if self._weight == 0.0:
            return False
        if self._optimize_last_step_only:
            last_optimized_step = min(self._n_steps, self._n_steps_limit) - 1
            return step == last_optimized_step
        return step < self._n_steps

    def __call__(
        self, prediction: StepPredictionABC, target_data: TensorMapping
    ) -> torch.Tensor | None:
        if self.step_is_optimized(prediction.step):
            loss_output = self._loss(prediction.data, target_data, prediction.step)
            return self._weight * loss_output.total()
        return torch.tensor(0.0, device=get_device())
