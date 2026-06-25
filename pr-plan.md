# Add `use_single_component_optim` flag to `CoupledTrainStepperConfig`

Adds an opt-in `use_single_component_optim` flag to `CoupledTrainStepperConfig` so
that each `train_on_batch` call optimizes exactly one of {ocean, atmosphere}, chosen
by a per-batch coin owned by `CoupledStepperTrainLoss`. Default off reproduces today's
behavior exactly. All production code lives in `fme/coupled/stepper.py`
(`CoupledStepperTrainLoss`, `CoupledTrainStepperConfig`, `CoupledTrainStepper`);
`ComponentLossSchedule` in `fme/coupled/loss.py` is reused unchanged.

---

## `fme/coupled/stepper.py` (modified)

New import (currently absent):

```python
from fme.core.distributed import Distributed  # NEW — for distributed-consistent coin seed
# numpy (`np`) is already imported.
```

### `CoupledStepperTrainLoss` — owns the coin

```python
class CoupledStepperTrainLoss:
    def __init__(
        self,
        ocean_loss: StepLoss,
        atmosphere_loss: StepLoss,
        ocean_schedule: ComponentLossSchedule,
        atmosphere_schedule: ComponentLossSchedule,
        single_component_optim: bool = False,  # NEW — enable the per-batch coin
    ):
        ...
        # NEW coin state:
        self._single_component_optim: bool = single_component_optim
        self._coin_rng: np.random.RandomState | None = None  # lazy, like TimeLengthProbabilities
        self._selected_realm: Literal["ocean", "atmosphere"] | None = None  # None == both eligible

    def _initialize_coin_rng(self) -> None:  # NEW — private; lazy-init, mirrors TimeLengthProbabilities.initialize_rng
        # if self._coin_rng is None:
        #     self._coin_rng = np.random.RandomState(
        #         Distributed.get_instance().get_seed() + <UNIQUE_OFFSET>  # != 684
        #     )  # identical across ranks -> identical coin decisions, reproducible from run seed
        ...

    def flip_optimization_coin(self) -> None:  # NEW — re-draw selected realm; no-op when disabled
        # When disabled: leave _selected_realm = None (both eligible) and return.
        # When enabled: among realms with n_required_forward_steps() > 0 (non-null THIS batch),
        #   - 0 non-null  -> should be impossible (post_init guarantees >=1); leave None defensively
        #   - 1 non-null  -> select it deterministically
        #   - 2 non-null  -> fair 50/50 draw via self._coin_rng
        # MUST be called AFTER sample_n_steps() so the null check sees this batch's window.
        ...

    def step_is_optimized(
        self,
        realm: Literal["ocean", "atmosphere"],
        step: int,
    ) -> bool:  # CHANGED — coin short-circuit, then delegate to schedule
        # NEW guard (training only; coin disabled in eval, mirroring schedule._is_training):
        #   if self._single_component_optim and <training> and self._selected_realm is not None \
        #      and realm != self._selected_realm:
        #       return False
        # then existing behavior:
        return self._schedules[realm].step_is_optimized(step)

    def __call__(
        self,
        prediction: ComponentEnsembleStepPrediction,
        target_data: TensorMapping,
    ) -> torch.Tensor | None:  # CHANGED — route through the wrapper instead of the schedule
        realm = prediction.realm
        # CHANGED: was `self._schedules[realm].step_is_optimized(prediction.step)`
        if not self.step_is_optimized(realm, prediction.step):
            return None
        ...

    # def compute_loss(...)  # UNCHANGED — intentionally still calls the schedule directly,
    #   NOT the coin-gated wrapper, so evaluate_all_steps keeps per-step diagnostic metrics
    #   for the non-selected realm. Accumulation is gated separately by the
    #   wrapper in CoupledTrainStepper._accumulate_step_loss.
```

> **Eval gating detail.** `ComponentLossSchedule` tracks `_is_training`; the coin must read
> the same training/eval state. Resolve by having the loss query its schedules' training flag
> (e.g. a private `_is_training` property derived from the schedules) so `set_train()` /
> `set_eval()` already flip it — no new public surface. In eval, `step_is_optimized` ignores
> the coin and both realms stay eligible.

### `CoupledTrainStepperConfig` — flag, plumbing, validation

```python
@dataclasses.dataclass
class CoupledTrainStepperConfig:
    n_coupled_steps: int
    ocean: ComponentTrainingConfig
    atmosphere: ComponentTrainingConfig
    n_ensemble: int = -1
    use_single_component_optim: bool = False  # NEW — optimize exactly one realm per batch
    parameter_init: CoupledParameterInitConfig = ...

    def __post_init__(self):  # CHANGED — validate the new flag
        ...
        # Existing rule already guarantees >=1 non-null realm, so the coin always has a
        # selectable realm. Add a ValueError for any flag combination that cannot be
        # satisfied (e.g. a future state that would make both realms permanently null
        # while the flag is on). No new constraint on today's valid configs.

    def _build_loss(
        self, stepper: CoupledStepper, n_coupled_steps: int
    ) -> CoupledStepperTrainLoss:  # CHANGED — plumb the flag through
        ...
        return CoupledStepperTrainLoss(
            ocean_loss=ocean_step_loss,
            atmosphere_loss=atmos_step_loss,
            ocean_schedule=ocean_schedule,
            atmosphere_schedule=atmos_schedule,
            single_component_optim=self.use_single_component_optim,  # NEW
        )
```

### `CoupledTrainStepper` — flip once per batch

```python
class CoupledTrainStepper(...):
    def train_on_batch(self, ...):  # CHANGED — flip the coin once per batch
        ...
        self._loss.sample_n_steps()
        self._loss.flip_optimization_coin()  # NEW — MUST come after sample_n_steps()
        ...
```

### Critical detail — coin policy, ordering, and gating chokepoints

- **Policy.** Fair 50/50 among realms that are **non-null for this batch** — reuse
  `ComponentLossSchedule.n_required_forward_steps()` (already returns 0 when
  `loss_weight == 0.0` or the sampled window yields no optimized step). A single non-null
  realm is always selected.
- **Ordering.** `flip_optimization_coin()` runs **after** `sample_n_steps()` so the null
  determination reflects the current batch's stochastically-sampled `n_steps`.
- **Gating is two call sites, not one** (verified against code): the loss `__call__` and
  `compute_loss` previously called `self._schedules[realm].step_is_optimized(...)`
  **directly**, bypassing the `(realm, step)` wrapper used by `_accumulate_step_loss` for
  its `no_grad` decision. The coin gate goes in the wrapper; `__call__` is rerouted through
  it; `compute_loss` is deliberately left on the schedule so eval-only diagnostic metrics
  survive.
- **RNG.** Lazy `np.random.RandomState(Distributed.get_instance().get_seed() + <offset>)`
  with a unique offset `!= 684`; identical across ranks → identical decisions, reproducible
  from the run seed. Not routed through `seed_eval`/`ComponentLossSchedule.seed_rng` (those
  seed only the eval sampler).

---

## Tests

## `fme/coupled/test_loss.py` (modified)

Primary seam — pure, no model/GPU. Build on the existing `_build_coupled_loss` helper and
`steps_thru_atmos_7` fixture; mirror `test_step_is_optimized_last_step_only`,
`test_stochastic_n_steps_sample_changes_step_is_optimized`,
`test_stochastic_n_steps_deterministic_outcome`, and `test_stochastic_n_steps_samples_vary`.

```python
# _build_coupled_loss gains a `single_component_optim=False` pass-through kwarg (test helper change).

def test_single_component_exactly_one_realm_optimized():
    # GOAL: flag on + fixed seed -> after flip, exactly one realm reports
    # step_is_optimized == True for some step in its window; the other reports False
    # for every step. Use both realms non-null with multi-step windows.
    ...

def test_single_component_selection_varies_across_flips():
    # GOAL: over many flip_optimization_coin() calls both realms get selected at least once
    # (analogous to test_stochastic_n_steps_samples_vary). Both realms non-null, 50/50.
    ...

def test_single_component_deterministic_for_fixed_seed():
    # GOAL: two losses built under the same distributed seed produce identical selection
    # sequences across N flips (analogous to test_stochastic_n_steps_deterministic_outcome).
    ...

def test_single_component_null_realm_never_selected():
    # GOAL: when one realm is null, the other is selected on every flip.
    # PARAMETERIZE: null cause in {loss_weight == 0.0, sampled n_steps == 0}.
    ...

def test_single_component_flag_off_matches_today():
    # GOAL: regression guard — with the flag off, step_is_optimized (and the __call__ loss
    # path) reproduce current behavior exactly. Compare against the existing
    # test_coupled_stepper_train_loss expectations.
    ...

def test_single_component_disabled_in_eval():
    # GOAL: in eval mode (set_eval) the coin does not restrict either realm; both stay
    # eligible so evaluate_all_steps yields full per-step metrics.
    ...
```

## `fme/coupled/test_stepper.py` (modified)

Integration seam — build on `get_train_stepper_and_batch` with a small/cheap config
(`n_coupled_steps=1`, MSE losses, both realms non-null).

```python
def test_single_component_coin_reflipped_per_train_on_batch():
    # GOAL: with use_single_component_optim=True and a fixed seed, run several train_on_batch
    # calls; assert (a) the optimized realm varies across batches, and (b) within a single
    # batch only the selected realm carries non-zero accumulated loss / step metrics while
    # the other realm's contribute zero. Confirms the per-batch re-flip wiring.
    ...
```

---

## Open Questions

- When `use_single_component_optim=True` but one component is statically null (zero
  `loss_weight`, or `n_steps` whose max is 0), the coin degrades to always selecting the
  sole non-null realm — i.e. the flag has no observable effect. Should `__post_init__`
  (a) raise a `ValueError` (the flag is meaningless here), (b) emit a warning and proceed,
  or (c) silently allow it as graceful degradation? The plan currently assumes (c); (b)
  seems most useful for catching misconfigured experiments. Note this is distinct from the
  always-rejected case where *both* realms are null.
- Flag name: `use_single_component_optim` (proposed) vs `optimize_single_component_per_batch`?
  Method name: `flip_optimization_coin()` vs an alternative. Soft proposals only — `Config`
  suffix and `_`-private conventions hold either way.
- Coin policy is fixed at fair 50/50 among non-null realms; loss-weight-proportional
  probabilities are explicitly out of scope but could be a later extension.
