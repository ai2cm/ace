# Joint atmosphere-ocean pair optimization for coupled training

Coupled training gains a third per-batch objective, `joint`: for one drawn
pair of steps `(i_O, i_A)`, gradients flow between the ocean and the
atmosphere across the pair's one direct edge, optimizing `P(O, A)` beside the
conditional objectives `P(A|O)` and `P(O|A)`. `optimize_single_component_per_batch`
becomes `objective_per_batch`, a weighted draw over `{atmosphere, ocean, joint}`.

```
n          = dt_ocean / dt_atmos                # CoupledStepperConfig.n_inner_steps
i_O, i_A   : 1-indexed ocean / atmosphere steps (config); code steps are 0-indexed
W(i_O)     = {(i_O-1)*n + 1, ..., i_O*n}        # atmosphere steps averaged into ocean step i_O's forcing
x_O(i), x_A(i) : predicted ocean / atmosphere state at step i
θ_O, θ_A   : ocean / atmosphere module weights
L_O(i), L_A(i) : the realm's per-step loss at step i, loss_weight * StepLoss(...).total() as today
ocean later  ⇔ i_A ∈ W(i_O)       edge: L_O(i_O) → x_A(i_A) → θ_A   (x_A(i_A) in the ocean forcing mean)
atmos later  ⇔ i_A ∈ W(i_O + 1)   edge: L_A(i_A) → x_O(i_O) → θ_O   (x_O(i_O) in atmosphere step i_A's forcing)
earlier, later : the pair's two steps in causal order; L_earlier, L_later their per-step losses
L_joint    = L_earlier(i_earlier) + L_later(i_later)
```

```yaml
stepper_training:
  joint_pairs: [[1, 1], [1, 21], [1, 21]]   # [i_O, i_A]; repeats raise a pair's draw weight
  objective_per_batch:                       # optional
    atmosphere: 1.0                          # absent key = 0; weights normalized
    ocean: 1.0
    joint: 1.0                               # > 0 requires joint_pairs
```

Per batch:

```
objective_per_batch absent  -> every schedule's losses + L_joint(pair ~ U(joint_pairs))   (L_joint only if joint_pairs)
objective_per_batch present -> objective ~ normalized weights over objectives with weight > 0 and a loss this batch
                               atmosphere | ocean -> that schedule's losses (today's single-component batch)
                               joint              -> L_joint only
                               no such objective  -> no loss this batch (a weight-0 realm is never optimized)
rollout outer steps = max(today's schedule rollout, ceil(i_A / n))   # every batch, joint batches included
joint_pairs with objective_per_batch.joint absent or 0 -> accepted; the pairs go unused
```

The weighted draw replaces today's `randint` over non-null realms: the
deprecated flag keeps today's distribution, not today's per-batch sequence.

Gradient cuts are the existing `use_gradient_accumulation` detaches; joint mode
adds none. It keeps the pair's edge attached, runs both pair steps with grad
whatever the schedule, and holds `L_earlier` until the later step, so one
backward of `L_joint` runs through both components. With accumulation on (the
intended setting), no backward visits a component twice. With it off, the
cross-component backprop through every grad-enabled step is the same as today.

---

## `fme/coupled/loss.py` (modified)

```python
@dataclasses.dataclass(frozen=True)
class JointPair:  # NEW: one validated pair, 0-indexed
    ocean_step: int
    atmosphere_step: int
    n_inner_steps: int

    @classmethod
    def from_config(cls, pair: Sequence[int], n_inner_steps: int) -> "JointPair":  # [i_O, i_A] -> 0-indexed; raises off a direct edge
        ...

    @property
    def ocean_is_later(self) -> bool:  # atmosphere_step // n_inner_steps == ocean_step
        ...

    @property
    def n_outer_steps(self) -> int:  # atmosphere_step // n_inner_steps + 1
        ...

    def role(self, realm: Literal["ocean", "atmosphere"], step: int) -> Literal["earlier", "later"] | None:
        ...
```

## `fme/coupled/stepper.py` (modified)

```python
@dataclasses.dataclass
class ObjectivePerBatchConfig:  # NEW
    atmosphere: float = 0.0
    ocean: float = 0.0
    joint: float = 0.0

    def __post_init__(self): ...  # every weight >= 0, sum > 0


@dataclasses.dataclass
class CoupledTrainStepperConfig:
    joint_pairs: list[list[int]] | None = None                  # NEW
    objective_per_batch: ObjectivePerBatchConfig | None = None  # NEW
    optimize_single_component_per_batch: bool = False           # CHANGED: deprecated

    def __post_init__(self): ...  # CHANGED
    # flag and objective_per_batch both set -> ValueError
    # flag true -> DeprecationWarning; objective_per_batch = {atmosphere: 1, ocean: 1}; flag reset to False
    # objective_per_batch.<realm> > 0 for a statically null realm -> ValueError (today's flag check, per key)
    # objective_per_batch.joint > 0 without joint_pairs -> ValueError
    # joint_pairs: non-empty, pairs of two indices >= 1

    def get_train_window_data_requirements(self, stepper_config): ...  # CHANGED: atmosphere target horizon covers max i_A
    def _build_loss(self, stepper, n_coupled_steps): ...  # CHANGED: objective weights, JointPair list


class CoupledStepperTrainLoss:
    def __init__(
        self, ocean_loss, atmosphere_loss, ocean_schedule, atmosphere_schedule,
        objective_weights: ObjectivePerBatchConfig | None = None,  # CHANGED: replaces optimize_single_component_per_batch
        joint_pairs: Sequence[JointPair] = (),                     # NEW
    ): ...

    def sample_from_rng(self) -> None: ...  # CHANGED: n_steps, then objective, then pair; train/eval RNGs as today

    @property
    def joint_pair(self) -> JointPair | None: ...  # NEW: this batch's pair; None when L_joint is not in the batch

    def joint_role(self, realm, step) -> Literal["earlier", "later"] | None: ...  # NEW

    def n_required_outer_steps(self, n_inner_steps: int) -> int: ...  # CHANGED: covers joint_pair
    def step_is_optimized(self, realm, step) -> bool: ...  # CHANGED: pair steps always; objective gates the rest
    def compute_loss(self, prediction, target_data) -> torch.Tensor: ...  # CHANGED: pair steps regardless of schedule
    def __call__(self, prediction, target_data) -> torch.Tensor | None: ...  # CHANGED


class CoupledStepper:
    def get_prediction_generator(
        self, initial_condition, forcing_data, optimizer,
        joint_pair: JointPair | None = None,  # NEW: inference passes None
    ): ...  # CHANGED


class CoupledTrainStepper:
    def _accumulate_loss(self, ...): ...  # CHANGED: passes self._loss.joint_pair to the generator
    def _accumulate_step_loss(self, ...): ...  # CHANGED: holds L_earlier; accumulate_loss(L_earlier + L_later) once
```

### Critical detail: the pair's attached edge

- **Ocean later.** The pair's atmosphere step skips
  `detach_if_using_gradient_accumulation` before it joins the window stack, so
  it stays attached in the ocean forcing mean. Every other step of the window
  is unchanged.
- **Atmosphere later.** The ocean output at the pair's ocean step is kept
  attached beside the detached copy that becomes the next ocean IC. In the next
  window, only the pair's atmosphere step reads the attached copy: its
  ocean-derived forcings and, when it is the window's first step, the SST
  prescribed into its initial condition. Every other step reads the detached
  copy.
- **Grad context.** `_accumulate_loss` sets the grad context per `next()`, and
  generator code between yields runs in the context of the `next()` that
  resumes it. The window's forcings are built while resuming its first
  atmosphere step, which may run under `no_grad`. The pair step's attached
  forcing is therefore built while resuming the pair step itself, for example
  by splitting the window's atmosphere generator at that step. Never
  `torch.enable_grad()`: validation runs under `no_grad`.
- **Segment boundaries.** `single_module.py` `predict_generator` detaches its
  state only when resumed after a yield, so a segment that is not resumed
  hands its last state on attached. Each segment boundary therefore calls
  `optimizer.detach_if_using_gradient_accumulation` on the carried state: this
  reproduces today's detach, not a new cut. Each segment continues the
  atmosphere step count, so `optimizer.checkpoint(..., step=...)` sees the same
  step numbers as an unsplit window.
- **One backward.** Under accumulation `accumulate_loss` calls `backward` at
  once and frees the graph. Holding `L_earlier` keeps the earlier step's graph
  alive until the later step. No other backward reaches it, because every
  other path out of the earlier step is cut by an accumulation detach.

## `fme/coupled/train/train_config.py` (modified)

```python
def _validate_joint_pairs(
    joint_pairs: list[list[int]] | None, n_inner_steps: int, n_coupled_steps: int
) -> None:  # NEW; called in TrainConfig.__post_init__ beside _validate_n_steps
    # each pair: i_A ∈ W(i_O) ∪ W(i_O + 1)  (a direct edge)
    # each pair: i_O <= n_coupled_steps and ceil(i_A / n) <= n_coupled_steps
    # one ValueError naming every offending pair
    ...
```

---

## Tests

## `fme/coupled/test_loss.py` (modified)

```python
# Build on _build_coupled_loss, _single_component_loss, _optimized_realms.
# Port the test_single_component_* tests and the _build_coupled_loss
# optimize_single_component_per_batch kwarg to objective_per_batch {atmosphere: 1, ocean: 1}.

def test_objective_draw_follows_weights():
    # GOAL: draw frequencies follow normalized weights.
    # PARAMETERIZE: weights, including a zero key.
    ...

def test_objective_null_realm_drops_out():
    # GOAL: a realm with no loss this batch (sampled n_steps 0) is never drawn; the rest renormalize.
    ...

def test_objective_no_eligible_objective_optimizes_nothing():
    # GOAL: {atmosphere: 1} with a non-null ocean schedule, atmosphere n_steps 0 this batch
    # -> step_is_optimized false for every step of both realms.
    ...

def test_joint_objective_optimizes_only_pair_steps():
    # GOAL: objective joint -> step_is_optimized true only at the pair's two steps.
    ...

def test_no_objective_weights_schedules_plus_pair():
    # GOAL: objective_per_batch absent with joint_pairs -> schedule steps plus the pair's steps.
    ...

def test_joint_pair_draw_uniform_with_repeats():
    # GOAL: uniform over the list, so a repeated pair is drawn more often.
    ...

def test_n_required_outer_steps_covers_pair():
    # GOAL: rollout = max(schedules, pair). PARAMETERIZE: ocean later, atmosphere later.
    ...

def test_compute_loss_pair_step_outside_schedule():
    # GOAL: compute_loss at a pair step outside its schedule is non-zero.
    ...

def test_objective_eval_reproducible_after_seed_eval():
    # GOAL: eval objective and pair draws replay after seed_rng.
    ...
```

## `fme/coupled/test_stepper.py` (modified)

```python
# Build on get_train_stepper_and_batch, _single_component_stepper_and_batch,
# _trainable_train_stepper_and_batch. Port test_optimize_single_component_rejects_static_null_realm
# (it matches the error text "optimize_single_component_per_batch").

def test_objective_per_batch_config_validation():
    # GOAL: __post_init__ rejects invalid configs.
    # PARAMETERIZE: flag and mapping both set; all weights 0; a negative weight;
    # positive weight on a statically null realm; joint > 0 without pairs; malformed pair.
    ...

def test_deprecated_single_component_flag():
    # GOAL: DeprecationWarning; behaves as {atmosphere: 1, ocean: 1}; the flag reads False after
    # __post_init__, so dataclasses.replace on the config does not raise.
    ...

def test_joint_pair_cross_component_gradient():
    # GOAL: in a joint batch the earlier component's gradient differs from that of
    # L_earlier alone, and the later component gets a gradient.
    # PARAMETERIZE: ocean later / atmosphere later x use_gradient_accumulation true / false.
    ...

def test_joint_pair_with_schedules_accumulation():
    # GOAL: objective_per_batch absent, use_gradient_accumulation true, schedules optimizing
    # the steps next to the pair, pair atmosphere step not first in its window -> train step
    # runs (no backward through a freed graph) and both components get a gradient.
    # PARAMETERIZE: ocean later / atmosphere later.
    ...

def test_joint_pair_per_component_optimizers():
    # GOAL: per-component optimizers (OptimizationDict) with a joint batch -> one backward
    # reaches both components and each optimizer steps its own component.
    ...

def test_joint_pair_ocean_later_only_pair_step_attached():
    # GOAL: accumulation on -> in the ocean forcing mean only the pair's atmosphere step carries grad.
    ...

def test_joint_pair_atmos_later_only_pair_step_reads_attached_ocean():
    # GOAL: accumulation on -> only the pair's atmosphere step has a path to the ocean output.
    # PARAMETERIZE: pair step first in its window / not first, with the window's first step under no_grad.
    ...

def test_joint_pair_single_backward():
    # GOAL: Mock(wraps=Optimization(...)): one accumulate_loss call carries L_earlier + L_later;
    # L_earlier is never passed alone.
    ...

def test_joint_pair_beyond_atmosphere_n_steps():
    # GOAL: i_A past the atmosphere n_steps -> target data covers it and the loss is finite.
    ...

def test_joint_pair_validation_loss_reproducible():
    # GOAL: eval, NullOptimization -> per-step metrics for both realms; draws replay after seed_eval.
    # PARAMETERIZE: evaluate_all_steps True / False.
    ...
```

## `fme/coupled/train/test_train_config.py` (modified)

```python
def test_validate_joint_pairs():
    # GOAL: accept (1, 1), (1, n), (1, n + 1), (1, 2n); reject (1, 2n + 1) and (2, n), (2, 1)
    # (no direct edge), i_O > n_coupled_steps, ceil(i_A / n) > n_coupled_steps.
    # PARAMETERIZE: n.
    ...
```

---

## Open Questions

None. Settled above: a `joint` batch rolls out `max(schedules, pair)`;
`joint_pairs` with `joint` weight 0 or absent is accepted and unused; `L_joint`
sums each realm's weighted per-step loss, so a realm at `loss_weight: 0`
contributes only its gradient path through the other realm's loss.
