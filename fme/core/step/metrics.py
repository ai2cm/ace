"""Per-step metrics payloads carried on prediction batches.

A ``Step`` implementation that exposes extra training-time quantities
(quantities that only exist inside that step, e.g. the pre-correction values at
a corrector boundary) returns them as a :class:`StepMetrics` from ``.step``.

Two layers, mirroring ``StepperState``:

- :class:`StepMetrics` — the *step-focused*, polymorphic payload a step
  produces. Concrete subclasses carry whatever that step's loss and aggregator
  consume. Generic stepper / batch code only ever calls its tensor-container
  lifecycle methods (``to_device`` / ``to_cpu`` / ``pin_memory`` /
  ``broadcast_ensemble``) and :meth:`StepMetrics.cat` to assemble a rollout
  window; only the step's own loss and aggregator introspect concrete fields.
- :class:`StepperMetrics` — the concrete container riding on ``BatchData``
  alongside ``StepperState``, holding the step-focused payload (and a home for
  any future stepper-level metrics). Generic code moves it around via the same
  lifecycle methods without inspecting the payload.

A step that exposes nothing returns :class:`NullStepMetrics`, so generic call
sites need no ``None`` / ``isinstance`` branches. The associated
``StepMetricsLoss`` / ``StepMetricsAggregator`` classes (generic over the
concrete ``StepMetrics`` type) are introduced alongside the first features that
consume these payloads.
"""

import abc
import dataclasses
from collections.abc import Sequence
from typing import Generic, TypeVar


class StepMetrics(abc.ABC):
    """The step-focused, polymorphic payload a Step produces each step.

    The abstract interface mirrors ``StepperState`` (the per-sample object this
    rides beside on ``BatchData``): the tensor-container lifecycle methods plus
    :meth:`cat` to assemble a per-step sequence into one rollout window.
    """

    @abc.abstractmethod
    def to_device(self) -> "StepMetrics": ...

    @abc.abstractmethod
    def to_cpu(self) -> "StepMetrics": ...

    @abc.abstractmethod
    def pin_memory(self) -> "StepMetrics": ...

    @abc.abstractmethod
    def broadcast_ensemble(self, n_ensemble: int) -> "StepMetrics": ...

    @classmethod
    @abc.abstractmethod
    def cat(cls, items: "Sequence[StepMetrics]", dim: int) -> "StepMetrics":
        """Concatenate a per-step sequence along the time dim ``dim``.

        Called by the rollout to assemble per-timestep payloads into one
        window-aligned payload. Implementations may assume ``items`` are all of
        this concrete type and non-empty.
        """


@dataclasses.dataclass(frozen=True)
class NullStepMetrics(StepMetrics):
    """Payload for a step that exposes no extra quantities."""

    def to_device(self) -> "NullStepMetrics":
        return self

    def to_cpu(self) -> "NullStepMetrics":
        return self

    def pin_memory(self) -> "NullStepMetrics":
        return self

    def broadcast_ensemble(self, n_ensemble: int) -> "NullStepMetrics":
        return self

    @classmethod
    def cat(cls, items: "Sequence[StepMetrics]", dim: int) -> "NullStepMetrics":
        return cls()


M = TypeVar("M", bound=StepMetrics)


@dataclasses.dataclass
class StepperMetrics(Generic[M]):
    """Stepper-level metrics container carried on ``BatchData``.

    Concrete and parallel to ``StepperState``; generic code moves it around via
    the lifecycle methods without inspecting the step-focused ``step`` payload,
    which is the part that varies by Step implementation. A home for any future
    stepper-level (cross-step) metrics to live beside ``step``.
    """

    step: M

    def to_device(self) -> "StepperMetrics[M]":
        return StepperMetrics(self.step.to_device())  # type: ignore[arg-type]

    def to_cpu(self) -> "StepperMetrics[M]":
        return StepperMetrics(self.step.to_cpu())  # type: ignore[arg-type]

    def pin_memory(self) -> "StepperMetrics[M]":
        return StepperMetrics(self.step.pin_memory())  # type: ignore[arg-type]

    def broadcast_ensemble(self, n_ensemble: int) -> "StepperMetrics[M]":
        return StepperMetrics(self.step.broadcast_ensemble(n_ensemble))  # type: ignore[arg-type]

    @classmethod
    def cat(cls, items: "Sequence[StepperMetrics[M]]", dim: int) -> "StepperMetrics[M]":
        step_type = type(items[0].step)
        return cls(step_type.cat([item.step for item in items], dim))  # type: ignore[arg-type]


def null_stepper_metrics() -> StepperMetrics[NullStepMetrics]:
    """A :class:`StepperMetrics` carrying no step payload (the BatchData default)."""
    return StepperMetrics(NullStepMetrics())
