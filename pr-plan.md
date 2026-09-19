# Post-corrector residual tracking, presentation, and optimization for prescribed prognostics

For each variable in `prescribed_prognostic_names`, `step_with_adjustments`
overwrites the corrected output `x'` with the target `y` and discards `x'`.
This PR (stacked on `feature/corrector-loss-config`) records the
post-corrector residual `r' = y − x'` at that overwrite and uses it three
ways: (1) *tracking* — `r'` is carried per step on `StepOutput` and stacked
on `StepDiagnostics`, exported by the step-diagnostics writer; (2)
*presentation* — aggregators and writers see the model's own `x' = y − r'`
for prescribed names, while the value fed into the next step remains exactly
`y`; (3) *optimization* — an opt-in config selects prescribed names whose
training-loss prediction becomes `x'` (targets untouched), so the model
trains through the corrector on the fields it is prevented from drifting on.
The new config is grouped with `corrector_loss` under one `loss_features`
surface on `TrainStepperConfig`.

Three invariants define the optimization feature:

- (a) for each selected prescribed name the loss prediction is `x'`
  (algebraically exact: the prediction dict carries `y` after the overwrite,
  and `y − r' = x'`);
- (b) for unselected prescribed names today's behavior is preserved exactly —
  the loss sees `y` against `y`, a zero contribution (the loss keeps
  consuming the `y`-valued dict; presentation feeds aggregators from a
  separate dict);
- (c) the next-step input is always exactly `y` (the rollout state is never
  touched).

Where a name is both prescribed and ocean-written, the pre-overwrite value is
the ocean's blend, not the corrector's output — the residual optimizes a
target-vs-target comparison over ocean points and the network only elsewhere.
Selecting such a name is a build-time error.

---

## `fme/core/step/output.py` (modified)

```python
@dataclasses.dataclass
class StepOutput:
    # NEW — after the existing fields:
    prescribed_residual: TensorMapping = dataclasses.field(default_factory=dict)
    # r'[name] = next_step_input_data[name] − pre-overwrite output[name],
    # one entry per prescribed prognostic name; empty when none are configured.

    @property
    def presented_output(self) -> TensorDict:  # NEW
        # {**output, name: output[name] − prescribed_residual[name]} — since
        # output[name] is y after the overwrite, this recovers x' for
        # prescribed names and is the identity elsewhere.
        ...

    @classmethod
    def stack_diagnostics(cls, outputs) -> "StepDiagnostics | None":
        # CHANGED — also stacks prescribed_residual (same union/consistency
        # validation per field); returns None only when delta AND
        # prescribed_residual are empty across all steps.
        ...
```

A plain mapping rather than a wrapper dataclass: the residual is one
name→tensor dict with no behavior of its own; detach happens at the step seam
(below) and masking reuses the existing spatial-masking helper.

## `fme/core/step/step_diagnostics.py` (modified)

```python
PRESCRIBED_RESIDUALS = "prescribed_residuals"  # NEW — dataset key, alongside CORRECTION_DELTAS

@dataclasses.dataclass
class StepDiagnostics:
    prescribed_residual: TensorMapping = dataclasses.field(default_factory=dict)  # NEW
    # stacked (sample, time, ...) like delta

    # to_device / to_cpu / pin_memory / broadcast_ensemble — CHANGED, carry
    # prescribed_residual with the same treatment as delta.

    def to_datasets(self, time):  # CHANGED — adds PRESCRIBED_RESIDUALS when non-empty
        ...
```

## `fme/core/step/step.py` (modified)

```python
class StepABC(abc.ABC):
    # __init__ CHANGED — self._detach_prescribed_residuals = True, beside
    # _detach_corrector_deltas

    def set_detach_prescribed_residuals(self, detach: bool) -> None:  # NEW
        # mirrors set_detach_corrector_deltas; wrapping steps must forward
        ...
```

A parallel flag, not a reuse of `set_detach_corrector_deltas`: when only one
of `corrector_loss` / prescribed optimization needs gradients, the other's
tensors stay off the autograd graph. Both default detached; only
`Stepper.build_loss` flips them, independently.

## `fme/core/step/single_module.py` (modified)

```python
def step_with_adjustments(
    ...,
    detach_corrector_deltas: bool = True,
    detach_prescribed_residuals: bool = True,  # NEW
) -> StepOutput:
    # CHANGED — the prescribed-prognostic overwrite loop also records
    #   residual[name] = next_step_input_data[name] − output[name]
    # before the overwrite (output[name] is x' at that point: post-corrector,
    # and post-ocean for an ocean-written name — hence the build-time
    # exclusion of ocean-written names from optimization selection).
    # Residuals are detached per the flag, and returned as
    # StepOutput(..., prescribed_residual=residuals).


class SingleModuleStep(StepABC):
    def step(self, args, wrapper=...) -> StepOutput:
        # CHANGED — also passes
        # detach_prescribed_residuals=self._detach_prescribed_residuals
        ...
```

## `fme/core/step/secondary_module.py`, `fme/core/step/radiation.py`, `fme/ace/step/fcn3.py`, `fme/core/step/multi_call.py` (modified)

The remaining `step_with_adjustments` callers pass
`detach_prescribed_residuals=self._detach_prescribed_residuals`, the same
one-line change as `SingleModuleStep`. `MultiCallStep` forwards
`set_detach_prescribed_residuals` to the wrapped step (the
`set_detach_corrector_deltas` pattern) and its `step` carries the wrapped
step's `prescribed_residual` through, like `corrector_diagnostics`
(multi-call output names are disjoint from prescribed names, so
`presented_output` on the merged `StepOutput` is unaffected).

## `fme/core/loss.py` (modified)

```python
@dataclasses.dataclass
class StepLossPrescribedArgs:  # NEW — sibling of StepLossCorrectorArgs
    optimized_names: list[str]  # concrete names, resolved at build


class StepLoss(torch.nn.Module):
    def __init__(
        self,
        loss: WeightedMappingLoss,
        sqrt_loss_decay_constant: float = 0.0,
        corrector_args: StepLossCorrectorArgs | None = None,
        prescribed_args: StepLossPrescribedArgs | None = None,  # NEW
    ): ...

    @property
    def needs_prescribed_residuals(self) -> bool:  # NEW
        ...

    def forward(
        self,
        predict_dict: TensorMapping,
        target_dict: TensorMapping,
        step: int,
        data_mask: TensorMapping | None = None,
        deltas: TensorMapping | None = None,
        residuals: TensorMapping | None = None,  # NEW — appended, existing
    ) -> LossOutput: ...                         # positional callers untouched
```

### Critical detail — `forward` with residuals

- `residuals` `None`/empty or no `prescribed_args`: exactly today's behavior.
- Otherwise every name in `optimized_names` must be in `residuals`, else
  raise — prescribed names are known at build and the step produces a
  residual for each on every step, so this is a drift guard, mirroring the
  delta rule.
- Swap: prediction fed to the main loss is `predict_dict[k] − residuals[k]`
  for `k in optimized_names` (this *is* `x'`), plain `predict_dict[k]`
  otherwise; targets untouched. Composes with the pre-corrector swap: the two
  features' name sets are disjoint by construction (`loss_visible_names`
  subtracts `prescribed_prognostic_names` on the base branch), so at most one
  swap applies per name.
- Off-mask points: a masked residual is NaN-filled, so the swapped prediction
  is NaN off-mask — dropped by `WeightedMappingLoss`'s target-driven NaN
  zeroing, since the target is NaN at the same points in masked-output
  configs. Covered by the masked-output test below.
- No new metric: once presentation lands, the existing per-variable
  aggregator metrics report `x'` for prescribed names, which is the quantity
  a residual metric would monitor.

`StepLossConfig.build` gains a default-`None` `prescribed_args` parameter
passed into `StepLoss`, like `corrector_args`. `fme/core/loss.py` still
imports nothing from `fme/core/corrector/`.

## `fme/core/loss_features.py` (new)

```python
@dataclasses.dataclass
class PrescribedPrognosticOptimizationConfig:
    names_and_prefixes: list[str] | None = None

    def __post_init__(self):
        # None ⇒ error: configuring the feature while selecting nothing is a
        # contradiction, not a no-op (corrector_loss convention).
        # Docstring documents: the swap also applies to validation batches
        # (under no_grad), so selected names report nonzero validation loss
        # where today they report zero — intended, validation should measure
        # what training optimizes; and with no corrector configured the
        # feature optimizes the raw network output for selected names
        # (x' = x), which is supported.

    def build(
        self,
        prescribed_prognostic_names: Collection[str],
        ocean_written_names: Collection[str],
    ) -> StepLossPrescribedArgs:
        # Raises when prescribed_prognostic_names is empty (configured with
        # nothing to consume); when any entry matches no prescribed name
        # (NameAndPrefixSelection.unmatched_entries — all prescribed names
        # are known at build, no discovery pass); and when a selected name is
        # in ocean_written_names — there the pre-overwrite value is the
        # ocean's blend (a Prescriber keeps the generated field off the ocean
        # mask), so the residual loss compares target against target over
        # ocean points and reaches the network only elsewhere: muddled
        # semantics rejected up front, per the keep_gradient_names
        # philosophy. No corrector is required.


@dataclasses.dataclass
class LossFeaturesConfig:
    corrector: CorrectorLossConfig | None = None
    prescribed_prognostic_optimization: PrescribedPrognosticOptimizationConfig | None = None

    def __post_init__(self):
        # error when both are None: configuring loss_features while selecting
        # no feature is a contradiction, not a no-op.
```

### Critical detail — the config regrouping

`TrainStepperConfig.corrector_loss` (added on the base branch, unmerged) is
*replaced* by `loss_features`; in yaml, `corrector_loss: {...}` becomes
`loss_features: {corrector: {...}}`. No deprecation shim: the base branch has
not merged, so no released config carries `corrector_loss`, and training
configs have no checkpoint-compatibility constraint. The module lives in
`fme/core/` (importing `fme/core/corrector/loss.py` and `fme/core/loss.py`)
because the grouped surface is not corrector-owned — the prescribed feature
works with no corrector configured.

## `fme/ace/stepper/single_module.py` (modified)

```python
class Stepper:
    def build_loss(
        self,
        loss_config: StepLossConfig,
        loss_features: LossFeaturesConfig | None = None,  # CHANGED — replaces
    ) -> StepLoss:                                        # corrector_loss
        # CHANGED — corrector_args from loss_features.corrector.build(...) as
        # on the base branch; prescribed_args from
        # loss_features.prescribed_prognostic_optimization.build(
        #     prescribed_prognostic_names=self.get_prescribed_prognostic_names(),
        #     ocean_written_names={self._step_obj.surface_temperature_name}
        #         when an ocean is configured (prescribed or slab — the
        #         property is non-None exactly then), else frozenset())
        # then set_detach_corrector_deltas(False) iff needs_corrector_deltas
        # and set_detach_prescribed_residuals(False) iff
        # needs_prescribed_residuals — independent flips.

    def step(self, ...):
        # CHANGED — applies self._output_masking to prescribed_residual as it
        # does to corrector_diagnostics (NaN off-mask).

    def predict(self, ...):
        # CHANGED — presentation: in the final BatchData.new_on_device
        # rebuild (where stack_diagnostics already attaches), each prescribed
        # name in data.data is replaced by y − stacked residual (= x');
        # data.data holds exactly the forward steps at that point, matching
        # the residual's time dim. Ordering inside the existing function
        # body: prognostic_state = data.get_end(...) and
        # compute_derived_variables both run before the rebuild, so the next
        # window's initial condition and the derived variables stay on y.
        # predict_generator's state = result.output is untouched — the
        # rollout consumes exactly y (invariant (c)).


@dataclasses.dataclass
class TrainStepperConfig:
    loss_features: LossFeaturesConfig | None = None  # CHANGED — replaces
                                                     # corrector_loss


class TrainStepper(TrainStepperABC[...]):
    def _accumulate_loss(self, ...):
        # CHANGED — splits the one gen_step variable:
        #   loss consumes unfold_ensemble_dim(step_output.output, ...) — the
        #     y-valued dict, unchanged for unselected names (invariant (b));
        #   output_list (→ gen_data → training/validation aggregators)
        #     collects unfold_ensemble_dim(step_output.presented_output, ...).
        # When self._loss_obj.needs_prescribed_residuals, residuals =
        # unfold_ensemble_dim(dict(step_output.prescribed_residual), ...) is
        # passed down, beside deltas. The hard boundary extends to r':
        # training consumes residuals at the StepOutput level here, never via
        # the StepDiagnostics carriage.

    def _accumulate_step_loss(self, ..., deltas=None, residuals=None):  # CHANGED
        # forwards residuals into self._loss_obj(...)
        ...
```

### Critical detail — presentation scope and grounding

Presentation is unconditional: whenever `prescribed_prognostic_names` is
configured, aggregators and writers see `x'` — no opt-in switch. Rationale:
`y` is already in the target/reference data every aggregator compares
against; presenting `y` as the prediction makes those metrics vacuous for
prescribed names, and the step-diagnostics writer plus presented outputs
carry strictly more information than today.

The four consumer surfaces are covered by the two seams above, verified on
the base branch:

- training `gen_data` and validation metrics — the `_accumulate_loss` split
  (validation runs the same `train_on_batch` under `torch.no_grad()` via
  `run_validation_loop`, so the same presented dict reaches
  `record_batch`);
- inline-inference aggregators and inference writers — the `Stepper.predict`
  rebuild (writers read `batch.prediction` downstream of it; the restart
  write consumes `prognostic_state`, which stays on `y`).

No aggregator or writer implementation changes; none takes a new or
differently-shaped argument. `fme.coupled` is out of scope (its
`ComponentStepPrediction.data` serves prediction, next-step input, and
ocean-forcing roles at once and needs its own design); the coupled stepper's
loss path passes no `loss_features` and is unchanged.

Loss targets and the prescribed `y` are the same values for
`n_ic_timesteps == 1` (the single-module case) and names untouched by
`Stepper.forcing_deriver`; the swap itself is exact regardless, since it
cancels the same tensor the overwrite wrote.

---

## Tests

## `fme/core/step/test_step.py` (modified)

```python
# Deterministic corrector (fixed-offset correction) + one prescribed name.

def test_step_with_adjustments_prescribed_residual():
    # GOAL: residual equals next_step_input_data − pre-overwrite output for
    # each prescribed name; output carries y; delta drop unchanged; empty
    # residual dict when no names are prescribed.
    ...

def test_prescribed_residual_detach_flag():
    # GOAL: default detaches residuals; detach_prescribed_residuals=False
    # keeps them on the graph; the two detach flags act independently.
    # PARAMETERIZE: (detach_deltas, detach_residuals) truth table.
    ...

```

## `fme/core/step/test_output.py` (modified)

```python
def test_presented_output():
    # GOAL: presented_output is x' for prescribed names and identical to
    # output elsewhere; identity when no residuals.
    ...

def test_stack_diagnostics_prescribed_residual():
    # GOAL: residual series stacked alongside delta; None only when both are
    # empty; inconsistent residual key sets across steps raise.
    ...
```

## `fme/core/step/test_multi_call.py` (modified)

```python
def test_multi_call_forwards_prescribed_residual_surface():
    # GOAL: set_detach_prescribed_residuals reaches the wrapped step;
    # MultiCallStep.step carries the wrapped prescribed_residual and
    # presented_output reflects it.
    ...
```

## `fme/core/step/test_step_diagnostics.py` (modified)

```python
def test_step_diagnostics_ops_carry_prescribed_residual():
    # GOAL: to_device/to_cpu/pin_memory/broadcast_ensemble treat
    # prescribed_residual like delta.
    ...

def test_to_datasets_exports_prescribed_residuals():
    # GOAL: PRESCRIBED_RESIDUALS dataset present iff residuals non-empty;
    # CORRECTION_DELTAS unaffected.
    ...
```

## `fme/core/test_loss.py` (modified)

```python
def test_prescribed_swap_selected_only():
    # GOAL: loss prediction is predict − residual for optimized_names only;
    # unselected prescribed names keep the plain prediction (zero
    # contribution when prediction == target); targets untouched.
    ...

def test_prescribed_missing_residual_raises():
    # GOAL: a non-empty residual dict lacking an optimized name raises.
    ...

def test_prescribed_empty_residuals_inert():
    # GOAL: residuals None/empty, or no prescribed_args ⇒ today's behavior;
    # needs_prescribed_residuals truth table over config presence.
    ...

def test_prescribed_and_corrector_swaps_compose():
    # GOAL: both args configured with disjoint names in one forward — each
    # swap applies to its own names, penalty unaffected.
    ...
```

## `fme/core/test_loss_features.py` (new)

```python
def test_config_post_init_errors():
    # GOAL: LossFeaturesConfig with both None; optimization config with
    # names_and_prefixes=None — each raises in __post_init__.
    ...

def test_build_errors():
    # GOAL: empty prescribed_prognostic_names; an entry matching no
    # prescribed name; a selected name in ocean_written_names — each raises
    # at build with distinct messages.
    # PARAMETERIZE: entry ∈ {exact name, trailing-underscore prefix}.
    ...

def test_build_without_corrector():
    # GOAL: build succeeds with no corrector configured (x' = x case) and
    # packs exactly the matched prescribed names.
    ...
```

## `fme/ace/stepper/test_single_module.py` (modified)

```python
# Deterministic corrector + prescribed name via the existing stepper helpers.

def test_prescribed_optimization_invariants():
    # GOAL: (a) the selected name's loss equals the analytic loss of x'
    # against the target; (b) an unselected prescribed name contributes zero;
    # (c) the rollout's next-step input is exactly y (step-2 inputs match the
    # forcing series, with and without the feature).
    ...

def test_gradient_flows_through_residual_when_configured():
    # GOAL: with the feature on, parameter grads differ from the detached
    # baseline; with only corrector_loss on, residuals stay detached (and
    # vice versa — flag independence end to end).
    ...

def test_presentation_train_gen_data():
    # GOAL: TrainOutput.gen_data carries x' for prescribed names (feature
    # configured or not); the loss value is unchanged for unselected names.
    ...

def test_presentation_predict():
    # GOAL: predict's returned data carries x' for prescribed names; the
    # returned prognostic state (next window's initial condition) carries y;
    # derived variables computed from y.
    ...

# test_predict_with_prescribed_prognostic — CHANGED: it asserts predict
# returns exactly the forcing values for prescribed names; under
# presentation it asserts x' (and that the prognostic state stays on y).

def test_validation_batch_reports_prescribed_loss():
    # GOAL: under no_grad with NullOptimization the selected name's loss is
    # nonzero and finite — the documented validation-visible change.
    ...

def test_masked_output_prescribed_optimization_finite():
    # GOAL: masked-output config + prescribed optimization yields finite loss
    # and gradients through the NaN-filled residual path.
    ...

def test_loss_features_grouping():
    # GOAL: both features configured under one loss_features config in one
    # train_on_batch: pre-corrector swap and residual swap each apply to
    # their (disjoint) names; regularization metrics still appear.
    ...
```

---

## Open Questions

- Naming: `loss_features` (field and `LossFeaturesConfig`) vs
  `training_loss_features`; `prescribed_prognostic_optimization` vs the
  shorter `prescribed_optimization`; `presented_output` vs
  `output_without_prescription`.
- Presentation is proposed unconditional-when-prescribing; is an opt-out
  wanted for inference users who prefer writers to record the on-rails `y`?
- Is a dedicated residual-magnitude training metric wanted, or do the
  presentation-informed aggregator metrics suffice (proposed)?
