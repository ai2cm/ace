# Add `optimize_single_component_per_batch` flag to `CoupledTrainStepperConfig`

Adds an opt-in `optimize_single_component_per_batch` flag to `CoupledTrainStepperConfig`
so that each `train_on_batch` call optimizes exactly one of {ocean, atmosphere}, chosen
per batch by `CoupledStepperTrainLoss`. Default off reproduces today's behavior exactly.
All production code lives in `fme/coupled/stepper.py` (`CoupledStepperTrainLoss`,
`CoupledTrainStepperConfig`, `CoupledTrainStepper`); `ComponentLossSchedule` in
`fme/coupled/loss.py` is reused unchanged.

> **Naming note (settled in review).** The flag is `optimize_single_component_per_batch`
> (clearer than "single component optim", which is ambiguous vs. FTO/FTA). The per-batch
> selection method is `component_optimization_choice()`, the RNG attribute is
> `_component_choice_rng`, and its lazy initializer is `_init_component_choice_rng()`.
> "Coin" terminology is dropped throughout.

---

## `fme/coupled/stepper.py` (modified)

New import (currently absent):

```python
from fme.core.distributed import Distributed  # NEW — for distributed-consistent choice seed
# numpy (`np`) is already imported.
```

### `CoupledStepperTrainLoss` — owns the per-batch component choice

```python
class CoupledStepperTrainLoss:
    def __init__(
        self,
        ocean_loss: StepLoss,
        atmosphere_loss: StepLoss,
        ocean_schedule: ComponentLossSchedule,
        atmosphere_schedule: ComponentLossSchedule,
        optimize_single_component_per_batch: bool = False,  # NEW — enable per-batch selection
    ):
        ...
        # NEW selection state. Two RNGs, mirroring ComponentLossSchedule's
        # _n_steps_sampler / _eval_n_steps_sampler split, so validation is comparable
        # to training (single-component) yet reproducible (see "Eval behavior" below):
        self._optimize_single_component_per_batch: bool = optimize_single_component_per_batch
        self._component_choice_rng: np.random.RandomState | None = None       # train; lazy, distributed seed
        self._eval_component_choice_rng: np.random.RandomState | None = None  # eval; seeded via seed_eval
        self._selected_realm: Literal["ocean", "atmosphere"] | None = None    # None == both eligible

    def _init_component_choice_rng(self) -> None:  # NEW — private; lazy-init train RNG, mirrors TimeLengthProbabilities.initialize_rng
        # if self._component_choice_rng is None:
        #     self._component_choice_rng = np.random.RandomState(
        #         Distributed.get_instance().get_seed() + <UNIQUE_OFFSET>  # != 684
        #     )  # identical across ranks -> identical decisions, reproducible from run seed

    def seed_rng(self, seed: int) -> None:  # RENAMED from seed_step_sampler; also reseeds the EVAL choice RNG
        # Reseeds the EVAL RNGs only (training RNGs are distributed-seeded lazily). Renamed
        # because it no longer only touches the n_steps "step sampler", and to match the
        # ComponentLossSchedule.seed_rng it delegates to (which is likewise eval-only).
        # existing: for i, schedule in enumerate(self._schedules.values()): schedule.seed_rng(seed + i)
        # NEW: self._eval_component_choice_rng = np.random.RandomState(seed + <UNIQUE_OFFSET>)
        # The validation loop already drives this via CoupledTrainStepper.seed_eval(0) before
        # every pass; same place ComponentLossSchedule.seed_rng reseeds the eval n_steps sampler.
        ...

    def component_optimization_choice(self) -> None:  # NEW — re-draw selected realm; no-op when disabled
        # When disabled: leave _selected_realm = None (both eligible) and return.
        # When enabled: pick the RNG by training/eval state (mirrors sample_n_steps):
        #     rng = self._component_choice_rng (train) | self._eval_component_choice_rng (eval)
        #   then, among realms with n_required_forward_steps() > 0 (non-null THIS batch),
        #   - 1 non-null  -> select it deterministically
        #   - 2 non-null  -> fair 50/50 draw via rng
        #   (0 non-null is impossible: __post_init__ guarantees >=1, and the static-null
        #    case is rejected at construction — see Validation below.)
        # Runs in BOTH train and eval (validation goes through train_on_batch), so the
        # eval choice is freshly set per validation batch — no stale carryover.
        # MUST be called AFTER sample_n_steps() so the null check sees this batch's window.
        ...

    def step_is_optimized(
        self,
        realm: Literal["ocean", "atmosphere"],
        step: int,
    ) -> bool:  # CHANGED — selection short-circuit, then delegate to schedule
        # NEW guard — applies in BOTH train and eval (no _is_training gate here; that lives
        # in component_optimization_choice, which only picks the train vs eval RNG):
        #   if self._optimize_single_component_per_batch and self._selected_realm is not None \
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
    #   NOT the selection-gated wrapper, so evaluate_all_steps keeps per-step diagnostic
    #   metrics for the non-selected realm. Accumulation is gated separately by the
    #   wrapper in CoupledTrainStepper._accumulate_step_loss.
```

> **Two RNGs, train + eval — for comparability.** This mirrors `ComponentLossSchedule`'s
> `_n_steps_sampler` / `_eval_n_steps_sampler` split, which exists precisely so the validation
> loss stays on the same scale as the training batch loss. The train choice RNG is lazily
> seeded from the distributed seed; the **eval** choice RNG is reseeded deterministically
> inside `CoupledStepperTrainLoss.seed_rng(seed)` (renamed from `seed_step_sampler`; already
> iterates the schedules' `seed_rng`), so each validation pass replays the same selections
> regardless of training RNG position. The validation loop already invokes
> `CoupledTrainStepper.seed_eval(0)` → `seed_rng(0)` before every pass, so no trainer-side
> change is needed.
> We still do **not** add a shared `RandomChoice` abstraction — the small repetition with
> `TimeLengthProbabilities` is acceptable and `np.random.RandomState` already suffices.

> **Eval behavior — selection ACTIVE in eval (reverses the earlier "disabled in eval" plan).**
> The scalar loss reported for both training and validation is `optimization.get_accumulated_loss()`,
> which only counts steps where the `step_is_optimized` wrapper returns `True`. If the selection
> were disabled in eval, validation would accumulate **both** realms while a training batch
> accumulates **one** — so the validation loss would be ~2× the (mixture) training loss and the
> two curves would be incomparable (misleading for monitoring / checkpoint-on-val-loss). Keeping
> the selection active in eval makes the validation loss single-component-per-batch too, matching
> the training scale. The training/eval distinction is read **inside
> `component_optimization_choice()`** (to pick the train vs eval RNG, mirroring
> `ComponentLossSchedule.sample_n_steps`), **not** in the `step_is_optimized` wrapper. Full
> per-step diagnostic metrics for the non-selected realm are unaffected: they flow through
> `compute_loss` (ungated under `evaluate_all_steps`), not through the accumulated-loss gate.

### `CoupledTrainStepperConfig` — flag, plumbing, validation

```python
@dataclasses.dataclass
class CoupledTrainStepperConfig:
    n_coupled_steps: int
    ocean: ComponentTrainingConfig
    atmosphere: ComponentTrainingConfig
    n_ensemble: int = -1
    optimize_single_component_per_batch: bool = False  # NEW — optimize exactly one realm per batch
    parameter_init: CoupledParameterInitConfig = ...

    def __post_init__(self):  # CHANGED — validate the new flag
        ...
        # Existing rule already guarantees >=1 non-null realm. When the flag is on, raise
        # ValueError if either realm is STATICALLY null (loss_weight == 0.0, or an n_steps
        # schedule whose max yields no optimized step): in that case the choice degrades to
        # always selecting the sole non-null realm and the flag has no observable effect, so
        # the configuration is rejected rather than silently accepted. (Distinct from the
        # already-rejected both-null case.)

    def _build_loss(
        self, stepper: CoupledStepper, n_coupled_steps: int
    ) -> CoupledStepperTrainLoss:  # CHANGED — plumb the flag through
        ...
        return CoupledStepperTrainLoss(
            ocean_loss=ocean_step_loss,
            atmosphere_loss=atmos_step_loss,
            ocean_schedule=ocean_schedule,
            atmosphere_schedule=atmos_schedule,
            optimize_single_component_per_batch=self.optimize_single_component_per_batch,  # NEW
        )
```

### `CoupledTrainStepper` — choose once per batch

```python
class CoupledTrainStepper(...):
    def train_on_batch(self, ...):  # CHANGED — make the component choice once per batch
        ...
        self._loss.sample_n_steps()
        self._loss.component_optimization_choice()  # NEW — MUST come after sample_n_steps()
        ...
```

### Critical detail — selection policy, ordering, and gating chokepoints

- **Policy.** Fair 50/50 among realms that are **non-null for this batch** — reuse
  `ComponentLossSchedule.n_required_forward_steps()` (already returns 0 when
  `loss_weight == 0.0` or the sampled window yields no optimized step). A single non-null
  realm is always selected. Equal weighting only — no configurable probabilities (settled
  in review; loss-weight-proportional weighting is out of scope).
- **Ordering.** `component_optimization_choice()` runs **after** `sample_n_steps()` so the
  null determination reflects the current batch's stochastically-sampled `n_steps`.
- **Gating is two call sites, not one** (verified against code): the loss `__call__` and
  `compute_loss` previously called `self._schedules[realm].step_is_optimized(...)`
  **directly**, bypassing the `(realm, step)` wrapper used by `_accumulate_step_loss` for
  its `no_grad` decision. The selection gate goes in the wrapper; `__call__` is rerouted
  through it; `compute_loss` is deliberately left on the schedule so eval-only diagnostic
  metrics survive.
- **RNG (train).** Lazy `np.random.RandomState(Distributed.get_instance().get_seed() +
  <offset>)` with a unique offset `!= 684`; identical across ranks → identical decisions,
  reproducible from the run seed, free-running across training batches.
- **RNG (eval) + comparability.** A separate eval choice RNG, reseeded inside
  `seed_rng` (renamed from `seed_step_sampler`; driven by the validation loop's `CoupledTrainStepper.seed_eval(0)`
  before each pass). `component_optimization_choice()` picks the train vs eval RNG by
  training/eval state, and the `step_is_optimized` wrapper honors the selection in **both**
  modes — so the accumulated validation loss is single-component-per-batch, matching the
  training-loss scale. Per-step `compute_loss` metrics stay ungated, so both realms keep
  full per-step eval diagnostics.

---

## Tests

## `fme/coupled/test_loss.py` (modified)

Primary seam — pure, no model/GPU. Build on the existing `_build_coupled_loss` helper and
`steps_thru_atmos_7` fixture; mirror `test_step_is_optimized_last_step_only`,
`test_stochastic_n_steps_sample_changes_step_is_optimized`,
`test_stochastic_n_steps_deterministic_outcome`, and `test_stochastic_n_steps_samples_vary`.

```python
# _build_coupled_loss gains an `optimize_single_component_per_batch=False` pass-through kwarg
# (test helper change).

def test_single_component_exactly_one_realm_optimized():
    # GOAL: flag on + fixed seed -> after the choice, exactly one realm reports
    # step_is_optimized == True for some step in its window; the other reports False
    # for every step. Use both realms non-null with multi-step windows.
    ...

def test_single_component_selection_varies_across_choices():
    # GOAL: over many component_optimization_choice() calls both realms get selected at least
    # once (analogous to test_stochastic_n_steps_samples_vary). Both realms non-null, 50/50.
    ...

def test_single_component_deterministic_for_fixed_seed():
    # GOAL: two losses built under the same distributed seed produce identical selection
    # sequences across N choices (analogous to test_stochastic_n_steps_deterministic_outcome).
    ...

def test_single_component_null_realm_never_selected():
    # GOAL: when one realm is null only for the current batch (sampled n_steps == 0 with a
    # multi-outcome schedule, so it is NOT statically null), the other is selected on every
    # choice. (Statically-null configs are rejected at construction — see the validation test.)
    ...

def test_single_component_flag_off_matches_today():
    # GOAL: regression guard — with the flag off, step_is_optimized (and the __call__ loss
    # path) reproduce current behavior exactly. Compare against the existing
    # test_coupled_stepper_train_loss expectations.
    ...

def test_single_component_active_in_eval():
    # GOAL: in eval mode (set_eval) the selection STILL restricts step_is_optimized to one
    # realm per batch (so the accumulated/validation loss is single-component, comparable to
    # training) — but compute_loss remains ungated, so per-step diagnostic metrics are still
    # available for both realms. Contrast with the train-mode behavior.

def test_single_component_eval_reproducible_after_seed_eval():
    # GOAL: seed_eval(0) makes the eval selection sequence deterministic and independent of
    # the training RNG position — two passes after seed_eval(0) produce identical selections
    # (analogous to how the eval n_steps sampler is reseeded). The eval RNG is separate from
    # the train RNG.
    ...
```

## `fme/coupled/test_stepper.py` (modified)

Integration seam — build on `get_train_stepper_and_batch` with a small/cheap config
(`n_coupled_steps=1`, MSE losses, both realms non-null).

```python
def test_single_component_choice_remade_per_train_on_batch():
    # GOAL: with optimize_single_component_per_batch=True and a fixed seed, run several
    # train_on_batch calls; assert (a) the optimized realm varies across batches, and (b)
    # within a single batch only the selected realm carries non-zero accumulated loss / step
    # metrics while the other realm's contribute zero. Confirms the per-batch re-choice wiring.
    ...

def test_single_component_validation_loss_is_single_component_and_reproducible():
    # GOAL: with the flag on, run train_on_batch(evaluate_all_steps=True) (the validation
    # path) under NullOptimization. Assert (a) the accumulated "loss" reflects exactly one
    # realm per batch (single-component, comparable to the training batch loss) while per-step
    # loss/{realm}_step_{step} metrics are present for BOTH realms; and (b) after
    # stepper.seed_eval(0), the per-batch selection sequence is identical across two runs.
    ...
```

### Validation test (`__post_init__`)

```python
def test_optimize_single_component_rejects_static_null_realm():
    # GOAL: with the flag on and one realm statically null (loss_weight == 0.0, or an n_steps
    # schedule whose max yields no optimized step), CoupledTrainStepperConfig.__post_init__
    # raises ValueError. PARAMETERIZE over the two static-null causes.
    ...
```
