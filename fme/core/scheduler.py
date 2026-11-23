import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import SequentialLR


@dataclasses.dataclass
class SchedulerConfig:
    """
    Configuration for a scheduler to use during training.

    Parameters:
        type: Name of scheduler class from torch.optim.lr_scheduler,
            no scheduler is used by default.
        kwargs: Keyword arguments to pass to the scheduler constructor.
        step_each_iteration: If true, step after each batch. Otherwise,
            just step at the end of each epoch. Schedulers that step with
            every iteration won't be passed the validation loss.
    """

    type: str | None = None
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    step_each_iteration: bool = False

    @property
    def has_schedule(self) -> bool:
        return self.type is not None

    def build(self, optimizer, max_epochs) -> "LRScheduler":
        """
        Build the scheduler.
        """
        if self.type is None:
            return LRScheduler()

        build_kwargs = {**self.kwargs}
        # work-around so we don't need to specify T_max
        # in the yaml file for this scheduler
        if self.type == "CosineAnnealingLR" and "T_max" not in self.kwargs:
            build_kwargs["T_max"] = max_epochs

        scheduler_class = getattr(torch.optim.lr_scheduler, self.type)
        return LRScheduler(
            scheduler_obj=scheduler_class(optimizer=optimizer, **build_kwargs),
            step_each_iteration=self.step_each_iteration,
        )


@dataclasses.dataclass
class SequentialSchedulerConfig:
    """
    Configuration for using torch.optim.SequentialLR to build a sequence of LR
    schedulers that run one after the other.

    Parameters:
        schedulers: Ordered sequence of SchedulerConfigs to define the schedulers
            for the SequentialLR. Note that all schedulers in the sequence must
            have the same value for steps_per_iteration.
        milestones: Sequence of integers that reflects milestone points, where
            milestones[i] corresponds to the last epoch or iteration where
            schedulers[i] is active before switching to schedulers[i+1]. For example,
            with two schedulers and milestones=[10] the first 10 epochs will use the
            first scheduler and then switch to the second scheduler for epoch 11.
        last_epoch: The index of last epoch. Default: -1.
    """

    schedulers: Sequence[SchedulerConfig]
    milestones: Sequence[int]
    last_epoch: int = -1

    def __post_init__(self):
        valid_steps_per_iteration = all(
            [
                x.step_each_iteration == self.schedulers[0].step_each_iteration
                for x in self.schedulers
            ]
        )
        if not valid_steps_per_iteration:
            raise ValueError(
                "All SchedulerConfigs in the SequentialSchedulerConfig must have "
                "identical values for steps_per_iteration."
            )

    @property
    def step_each_iteration(self) -> bool:
        return self.schedulers[0].step_each_iteration

    @property
    def has_schedule(self) -> bool:
        return True

    def build(self, optimizer, max_epochs) -> "LRScheduler":
        """
        Build the SequentialLR scheduler.
        """
        schedulers = [
            x.build(optimizer, max_epochs).scheduler_obj for x in self.schedulers
        ]
        return LRScheduler(
            scheduler_obj=SequentialLR(
                optimizer=optimizer,
                schedulers=schedulers,
                milestones=self.milestones,
                last_epoch=self.last_epoch,
            ),
            step_each_iteration=self.step_each_iteration,
        )


@dataclasses.dataclass
class HelicalSchedulerConfig:
    """
    Configuration for a periodic cosine learning rate schedule with an optional
    linear warmup and an optional linear decay of the cosine to zero.

    Each parameter group's learning rate is its initial learning rate times
    ``envelope(t) * cycle(t)``, with ``t`` the scheduler step::

        W = warmup_iters, P = period, m = min_factor, s0 = warmup_start_factor
        cycle(t) = s0 + (1 - s0) * t / W                    for t < W
                 = m + (1 - m) * (1 + cos(2 pi (t - W) / P)) / 2  for t >= W,
                                                    cosine_start = "peak"
                 = m + (1 - m) * (1 - cos(2 pi (t - W) / P)) / 2  for t >= W,
                                                    cosine_start = "trough"
        t_peak   = W (cosine_start = "peak") or W + P / 2 ("trough"),
                   the first time cycle(t) = 1
        envelope(t) = 1                                     decay_end is None
                    = 1                         for t <= t_peak
                    = max(0, (T - t) / (T - t_peak))  for t > t_peak, T = decay_end

    Parameters:
        period: Cosine period (peak to peak), in scheduler steps.
        min_factor: Cosine minimum, as a factor of the initial learning rate.
        cosine_start: Phase of the cosine at the end of the warmup: ``"peak"``
            starts it at the initial learning rate, ``"trough"`` at
            ``min_factor``.
        warmup_iters: Length of the linear warmup, in scheduler steps.
        warmup_start_factor: Factor of the initial learning rate at step 0 of
            the warmup.
        decay_end: Scheduler step at which the linear envelope reaches zero.
            If None, the cosine repeats with constant amplitude.
        step_each_iteration: If true, step after each batch. Otherwise, step at
            the end of each epoch.
    """

    period: int
    min_factor: float = 0.0
    cosine_start: Literal["peak", "trough"] = "peak"
    warmup_iters: int = 0
    warmup_start_factor: float = 0.0
    decay_end: int | None = None
    step_each_iteration: bool = True

    def __post_init__(self):
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.warmup_iters < 0:
            raise ValueError(
                f"warmup_iters must be non-negative, got {self.warmup_iters}"
            )
        for name in ("min_factor", "warmup_start_factor"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.cosine_start not in ("peak", "trough"):
            raise ValueError(
                "cosine_start must be 'peak' or 'trough', " f"got {self.cosine_start!r}"
            )
        if self.decay_end is not None and self.decay_end <= self.first_peak:
            raise ValueError(
                f"decay_end ({self.decay_end}) must be after the first peak "
                f"of the schedule (step {self.first_peak})"
            )

    @property
    def first_peak(self) -> float:
        """Scheduler step of the first maximum of the cycle."""
        if self.cosine_start == "peak":
            return float(self.warmup_iters)
        return self.warmup_iters + self.period / 2

    @property
    def has_schedule(self) -> bool:
        return True

    def factor(self, step: int) -> float:
        """Learning rate at ``step`` as a factor of the initial learning rate."""
        return self._envelope(step) * self._cycle(step)

    def _cycle(self, step: int) -> float:
        if step < self.warmup_iters:
            return self.warmup_start_factor + (1 - self.warmup_start_factor) * (
                step / self.warmup_iters
            )
        cos = math.cos(2 * math.pi * (step - self.warmup_iters) / self.period)
        half_wave = (1 + cos) / 2 if self.cosine_start == "peak" else (1 - cos) / 2
        return self.min_factor + (1 - self.min_factor) * half_wave

    def _envelope(self, step: int) -> float:
        if self.decay_end is None or step <= self.first_peak:
            return 1.0
        return max(0.0, (self.decay_end - step) / (self.decay_end - self.first_peak))

    def build(self, optimizer, max_epochs) -> "LRScheduler":
        return LRScheduler(
            scheduler_obj=HelicalLR(optimizer, self.factor),
            step_each_iteration=self.step_each_iteration,
        )


class HelicalLR(torch.optim.lr_scheduler.LRScheduler):
    """Sets each group's learning rate to its initial learning rate times
    ``factor(step)``. The factor is evaluated in closed form from the step
    count, so a scheduler resumed from ``state_dict`` continues the curve.
    """

    def __init__(self, optimizer, factor, last_epoch: int = -1):
        self._factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        factor = self._factor(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]

    def state_dict(self) -> dict[str, Any]:
        return {k: v for k, v in super().state_dict().items() if k != "_factor"}


class LRScheduler:
    """Thin wrapper around torch.optim.lr_scheduler._LRScheduler."""

    def __init__(
        self,
        scheduler_obj: torch.optim.lr_scheduler._LRScheduler | None = None,
        step_each_iteration: bool = False,
    ):
        self._scheduler_obj = scheduler_obj
        self._step_each_iteration = step_each_iteration

    @property
    def scheduler_obj(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        return self._scheduler_obj

    def should_step(self, is_iteration: bool) -> bool:
        """Determine whether the scheduler should be stepped based on
        configuration and context.
        """
        if self._scheduler_obj is None:
            return False
        do_iter_step = self._step_each_iteration and is_iteration
        do_epoch_step = not self._step_each_iteration and not is_iteration
        return do_iter_step or do_epoch_step

    def step(self, *args, **kwargs):
        if self._scheduler_obj is not None:
            self._scheduler_obj.step(*args, **kwargs)

    def state_dict(self):
        if self._scheduler_obj is None:
            return None
        return self._scheduler_obj.state_dict()

    def load_state_dict(self, state):
        if self._scheduler_obj is not None and state is not None:
            self._scheduler_obj.load_state_dict(state)
