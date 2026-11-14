import dataclasses
from collections.abc import Sequence
from typing import Literal

from fme.ace.stepper.time_length_probabilities import TimeLengthProbabilities


@dataclasses.dataclass(frozen=True)
class JointPair:
    """One ocean-atmosphere step pair joined by a direct edge, 0-indexed.

    The ocean is later when the atmosphere step lies in the ocean step's
    forcing window; the atmosphere is later when the atmosphere step lies in
    the next window, whose forcing reads the ocean step's output.

    Parameters:
        ocean_step: 0-indexed ocean step.
        atmosphere_step: 0-indexed atmosphere step.
        n_inner_steps: Atmosphere steps per ocean step.
    """

    ocean_step: int
    atmosphere_step: int
    n_inner_steps: int

    def __post_init__(self):
        if self.atmosphere_step // self.n_inner_steps not in (
            self.ocean_step,
            self.ocean_step + 1,
        ):
            raise ValueError(
                f"Joint pair [{self.ocean_step + 1}, {self.atmosphere_step + 1}] "
                "has no direct edge: the atmosphere step must lie in the ocean "
                "step's forcing window or the next one."
            )

    @classmethod
    def from_config(cls, pair: Sequence[int], n_inner_steps: int) -> "JointPair":
        """Build from a 1-indexed ``[i_O, i_A]`` config entry."""
        ocean_step, atmosphere_step = pair
        return cls(
            ocean_step=ocean_step - 1,
            atmosphere_step=atmosphere_step - 1,
            n_inner_steps=n_inner_steps,
        )

    @property
    def ocean_is_later(self) -> bool:
        return self.atmosphere_step // self.n_inner_steps == self.ocean_step

    @property
    def n_outer_steps(self) -> int:
        """Outer (ocean) steps needed to reach both pair steps."""
        return self.atmosphere_step // self.n_inner_steps + 1

    def role(
        self, realm: Literal["ocean", "atmosphere"], step: int
    ) -> Literal["earlier", "later"] | None:
        if realm == "ocean":
            if step != self.ocean_step:
                return None
            return "later" if self.ocean_is_later else "earlier"
        if step != self.atmosphere_step:
            return None
        return "earlier" if self.ocean_is_later else "later"


class ComponentLossSchedule:
    """Mutable per-component schedule that tracks the current effective
    ``n_steps`` (which may change per batch when stochastic) and answers
    step-optimization queries.
    """

    def __init__(
        self,
        n_steps: TimeLengthProbabilities | int | None,
        optimize_last_step_only: bool,
        loss_weight: float,
        n_steps_limit: int,
    ):
        if isinstance(n_steps, TimeLengthProbabilities):
            self._n_steps_sampler: TimeLengthProbabilities | None = n_steps
            self._eval_n_steps_sampler: TimeLengthProbabilities | None = (
                TimeLengthProbabilities(outcomes=list(n_steps.outcomes))
            )
            self._n_steps: float = float(n_steps.max_n_forward_steps)
        else:
            self._n_steps_sampler = None
            self._eval_n_steps_sampler = None
            self._n_steps = float("inf") if n_steps is None else float(n_steps)
        self._is_training: bool = True
        self._optimize_last_step_only = optimize_last_step_only
        self._loss_weight = loss_weight
        self._n_steps_limit = n_steps_limit

    @property
    def loss_weight(self) -> float:
        return self._loss_weight

    def sample_n_steps(self) -> None:
        sampler = (
            self._n_steps_sampler if self._is_training else self._eval_n_steps_sampler
        )
        if sampler is not None:
            self._n_steps = float(sampler.sample())

    def seed_rng(self, seed: int) -> None:
        if self._eval_n_steps_sampler is not None:
            self._eval_n_steps_sampler.seed_rng(seed)

    def set_train(self) -> None:
        self._is_training = True

    def set_eval(self) -> None:
        self._is_training = False

    def n_required_forward_steps(self) -> int:
        if self.loss_weight == 0.0:
            return 0
        return int(min(self._n_steps, self._n_steps_limit))

    def step_is_optimized(self, step: int) -> bool:
        if self.loss_weight == 0.0:
            return False
        if self._optimize_last_step_only:
            last_optimized_step = min(self._n_steps, self._n_steps_limit) - 1
            return step == last_optimized_step
        return step < self._n_steps
