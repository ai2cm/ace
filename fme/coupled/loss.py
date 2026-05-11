from fme.ace.stepper.time_length_probabilities import TimeLengthProbabilities


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
            self._n_steps: float = float(n_steps.max_n_forward_steps)
        else:
            self._n_steps_sampler = None
            self._n_steps = float("inf") if n_steps is None else float(n_steps)
        self._optimize_last_step_only = optimize_last_step_only
        self.loss_weight = loss_weight
        self._n_steps_limit = n_steps_limit

    def sample_n_steps(self) -> None:
        if self._n_steps_sampler is not None:
            self._n_steps = float(self._n_steps_sampler.sample())

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
