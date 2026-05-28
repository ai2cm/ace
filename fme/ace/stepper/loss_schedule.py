from fme.ace.stepper.time_length_probabilities import (
    TimeLength,
    TimeLengthProbabilities,
    TimeLengthSchedule,
)


class EpochNotProvidedError(ValueError):
    pass


def probabilities_from_time_length(value: TimeLength) -> TimeLengthProbabilities:
    if isinstance(value, TimeLengthProbabilities):
        return value
    else:
        return TimeLengthProbabilities.from_constant(value)


class LossSchedule:
    """Encapsulates which forward steps contribute to loss.

    Manages stochastic n_forward_steps sampling (including epoch-based
    schedules) and the ``optimize_last_step_only`` policy, answering
    per-step optimization queries via ``step_is_optimized``.
    """

    def __init__(
        self,
        n_forward_steps_schedule: TimeLengthSchedule | None,
        optimize_last_step_only: bool,
    ):
        self._schedule = n_forward_steps_schedule
        self._optimize_last_step_only = optimize_last_step_only
        self._train_sampler: TimeLengthProbabilities | None = None
        self._eval_sampler: TimeLengthProbabilities | None = None
        self._is_training: bool = True
        self._epoch: int | None = None

    def init_for_epoch(self, epoch: int | None) -> None:
        if (
            epoch is None
            and self._schedule is not None
            and len(self._schedule.milestones) > 0
        ):
            raise EpochNotProvidedError(
                "current configuration requires epoch to be provided "
                "on BatchData during training"
            )
        if self._epoch == epoch:
            return
        if self._schedule is not None:
            assert epoch is not None
            self._train_sampler = probabilities_from_time_length(
                self._schedule.get_value(epoch)
            )
            self._eval_sampler = TimeLengthProbabilities(
                outcomes=list(self._train_sampler.outcomes)
            )
        else:
            self._train_sampler = None
            self._eval_sampler = None
        self._epoch = epoch

    def sample(self, n_data_steps: int) -> int:
        """Sample and return the number of loss steps for the current batch."""
        sampler = self._train_sampler if self._is_training else self._eval_sampler
        if sampler is not None:
            sampled = sampler.sample()
            if sampled > n_data_steps:
                raise RuntimeError(
                    "The number of forward steps to train on "
                    f"({sampled}) is greater than the number of "
                    f"forward steps in the data ({n_data_steps}), "
                    "This is supposed to be ensured by the StepperConfig when train "
                    "data requirements are retrieved, so this is a bug."
                )
            return sampled
        else:
            return n_data_steps

    def n_forward_steps(
        self, n_data_steps: int, n_loss_steps: int, evaluate_all_steps: bool
    ) -> int:
        """Number of forward steps to iterate in the rollout loop."""
        if evaluate_all_steps:
            return n_data_steps
        return n_loss_steps

    def step_is_optimized(self, step: int, n_loss_steps: int) -> bool:
        if self._optimize_last_step_only:
            return step == n_loss_steps - 1
        return step < n_loss_steps

    @property
    def has_sampler(self) -> bool:
        return self._train_sampler is not None

    def seed_eval(self, seed: int) -> None:
        if self._eval_sampler is not None:
            self._eval_sampler.seed_rng(seed)

    def set_train(self) -> None:
        self._is_training = True

    def set_eval(self) -> None:
        self._is_training = False
