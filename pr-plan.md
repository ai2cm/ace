# Config-selected pre-corrector optimization and corrector regularization

Adds a training-only `corrector_loss` config that consumes the correction
deltas already carried on `StepOutput`: (1) *pre-corrector optimization* — for
selected corrector-modified variables the main loss sees the pre-corrector
prediction `output − delta`; (2) *corrector regularization* — a penalty pushing
`delta` toward zero in loss-normalized space. Field selection is pure opt-in
via `NameAndPrefixMatcher` entries. Implements the `StepOutputLoss` design of
#1273, with the divergences listed at the end.

---

## `fme/core/name_and_prefix_matcher.py` (modified)

```python
class NameAndPrefixMatcher:
    ...  # unchanged

@dataclasses.dataclass(frozen=True)
class NameAndPrefixSelection:  # NEW — matcher plus its entries, for validation/reporting
    entries: tuple[str, ...]

    @property
    def matcher(self) -> NameAndPrefixMatcher: ...

    def matched(self, names: Iterable[str]) -> list[str]:
        """Names (sorted) that match any entry."""

    def unmatched_entries(self, names: Iterable[str]) -> list[str]:
        """Entries that match none of ``names`` — the validation primitive."""
```

`NameAndPrefixMatcher` has no per-entry reporting today; validation needs it,
so the entry list is kept alongside the matcher rather than adding state to
the matcher itself.

## `fme/core/corrector/output.py` (modified)

```python
@dataclasses.dataclass
class CorrectorDiagnostics:
    def detach(self) -> "CorrectorDiagnostics":  # NEW
        return CorrectorDiagnostics(
            delta={k: v.detach() for k, v in self.delta.items()}
        )
```

## `fme/core/corrector/registry.py` (modified)

```python
class CorrectorABC(abc.ABC):
    @property
    def enabled(self) -> bool:  # NEW — whether this corrector can modify anything
        return True

    @property
    def keep_gradient_names(self) -> frozenset[str]:  # NEW — names corrected via
        return frozenset()                            # straight-through clamps

class CorrectionSequence(CorrectorABC):
    @property
    def enabled(self) -> bool:  # NEW — False for an empty sequence
        ...

    @property
    def keep_gradient_names(self) -> frozenset[str]:  # NEW — union over corrections
        ...

class EpochScheduledCorrector(CorrectorABC):
    # enabled / keep_gradient_names forwarded to the wrapped corrector,
    # independent of the current epoch's disabled state.  # NEW
```

Each `Correction` gains a `keep_gradient_names` property (default empty).
`ForcePositive` (`fme/core/corrector/utils.py`) returns its names when
`keep_gradient` is set; `SeaIceFractionCorrection`
(`fme/core/corrector/ocean.py`) returns the fields it clamps when
`keep_gradient` is set. This is the discovery mechanism the keep-gradient
guard below needs — today those names are unreachable from a corrector
instance (`CorrectionSequence._corrections` is private). No corrector's
numerical behavior changes.

## `fme/core/step/output.py` (modified)

```python
@dataclasses.dataclass
class StepOutput:
    @property
    def uncorrected(self) -> TensorDict:  # NEW — adopts #1273's step_output.uncorrected
        """output − delta for delta keys; output unchanged elsewhere."""
```

## `fme/core/loss.py` (modified)

`StepOutputLoss` lives here and imports no config from
`fme/core/corrector/loss.py` — the config's `build` assembles the primitives
and hands them in (one-way import: `corrector/loss.py` → `loss.py`).

```python
@dataclasses.dataclass
class StepOutputLossResult:  # NEW
    main: LossOutput
    regularization: torch.Tensor | None  # unweighted penalty, for metrics
    regularization_weight: float

    def total(self) -> torch.Tensor:
        """main.total() + regularization_weight * regularization (when present)."""


class StepOutputLoss:  # NEW — wraps StepLoss; pass-through when nothing configured
    def __init__(
        self,
        step_loss: StepLoss,
        precorrector_selection: NameAndPrefixSelection | None = None,
        regularization_selection: NameAndPrefixSelection | None = None,
        build_regularizer: Callable[[tuple[str, ...]], WeightedMappingLoss] | None = None,
        regularization_weight: float = 1.0,
    ): ...

    @property
    def needs_uncorrected_grad(self) -> bool:
        """True when either feature is configured — both differentiate through delta."""

    def __call__(
        self,
        step_output: StepOutput,
        target: EnsembleTensorDict,
        step: int,
        n_ensemble: int,
        data_mask: TensorMapping | None = None,
    ) -> StepOutputLossResult: ...
```

### Critical detail — `__call__` algorithm

- Prediction fed to the main loss: `step_output.uncorrected[k]` for keys
  matched by `precorrector_selection` and present in the deltas; plain
  `step_output.output[k]` otherwise. Targets untouched. The ensemble dim is
  unfolded internally (`unfold_ensemble_dim`, `fme/core/tensors.py`) before
  calling the wrapped `StepLoss` — callers pass the folded `StepOutput` as
  yielded.
- Regularization: penalty on `delta` toward zero in loss-normalized space,
  `regularizer(delta_selected, zeros_like)` — with an affine normalizer the
  means cancel, so this penalizes `delta/std`. Mean over the selected
  channels only: the `WeightedMappingLoss` is built lazily per tuple of
  *present-and-selected* names (small cache), since delta keys are only known
  at runtime and may vary.
- First-time runtime check (maintainer decision, mechanics proposed here):
  the first call whose `delta` is non-empty checks
  `selection.unmatched_entries(delta.keys())` for both features and raises on
  any unmatched entry, then never re-checks. Empty-delta calls do not count
  as "first" — `EpochScheduledCorrector` yields empty diagnostics across all
  disabled training epochs, so the check simply waits for the first enabled
  optimized step. Cost accepted per the maintainer decision: with an
  epoch-scheduled corrector this error can land epochs into a run.
- Warn-once: the first empty-delta call (with either feature configured)
  emits a single warning — "corrector produced no correction deltas; features
  inactive (expected while a corrector is epoch-scheduled off)" — and never
  repeats, so disabled epochs cannot spam.
- After the check passes (behavior proposed here; review settles it), a step
  where a selected variable is absent from the
  deltas: the pre-corrector swap no-ops for that variable; the regularization
  means over present-and-selected channels only (absent channels drop out
  rather than contributing zero). When none are present,
  `regularization=None` and the per-step metric is omitted. Metric caveat:
  the per-step penalty's channel set can vary, so `corrector_regularization`
  is comparable across epochs only while the corrector's modified set is
  stable — noted in the config docstring.

## `fme/core/corrector/loss.py` (new)

```python
@dataclasses.dataclass
class PreCorrectorOptimizationConfig:
    names_and_prefixes: list[str] | None = None


@dataclasses.dataclass
class CorrectorRegularizationConfig:
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    weight: float = 1.0
    names_and_prefixes: list[str] | None = None

    def __post_init__(self):
        # reject EnsembleLoss / NaN loss types and any global_mean_type (per #1273);
        # reject weight <= 0.
        # Config docstrings document (per #1273): the penalty has no per-step
        # decay — only `weight` scales it, unlike the main loss's
        # sqrt_loss_step_decay — and that configuring both features together
        # is first-class and composes.


@dataclasses.dataclass
class CorrectorLossConfig:
    precorrector_optimization: PreCorrectorOptimizationConfig | None = None
    regularization: CorrectorRegularizationConfig | None = None

    def __post_init__(self):
        # error when both are None: configuring corrector_loss while selecting
        # no feature is a contradiction, not a no-op.

    def build(
        self,
        step_loss: StepLoss,
        output_names: Collection[str],
        normalizer: StandardNormalizer,
        gridded_operations: GriddedOperations | None,
        channel_dim: int = -3,
    ) -> StepOutputLoss:
        # Build-time validation (see seam below), then assemble the
        # StepOutputLoss primitives: selections, the regularizer factory
        # (closes over self.regularization.loss.build(gridded_operations),
        # normalizer, channel_dim), and the weight.
```

### Critical detail — build-time validation seam (proposed; review settles it)

`__post_init__` cannot see the network output names, so name validation runs
in `build(...)`, called from `TrainStepper.__init__` (below) where the built
stepper's `out_names`, loss normalizer, and gridded operations are all in
hand. `build` raises when:

- a present feature config has `names_and_prefixes is None` — selecting
  nothing explicitly is an error, per the maintainer decision (opt-in
  inclusion via `NameAndPrefixMatcher` semantics, amending #1273's
  exclusion design);
- any entry matches no network output name
  (`NameAndPrefixSelection.unmatched_entries(output_names)`).

The network-output check is the mandated minimum: correctors can be
epoch-scheduled or dynamic, so their modified set is not checkable at build.
The runtime first-time check against the delta keys (above) covers the rest.

## `fme/core/step/step.py` (modified)

```python
class StepABC(abc.ABC):
    @property
    @abc.abstractmethod
    def corrector_enabled(self) -> bool:  # NEW — per #1273's build-time guard
        ...

    @property
    @abc.abstractmethod
    def corrector_keep_gradient_names(self) -> frozenset[str]:  # NEW
        ...

    def set_detach_corrector_deltas(self, detach: bool) -> None:  # NEW — concrete;
        self._detach_corrector_deltas = detach  # default True, set in __init__
```

### Critical detail — detach threading

The detach flag is step-level *state*, not a `step()` parameter: `StepABC`
stores `_detach_corrector_deltas = True` in its existing `__init__`, and each
concrete step passes it into `step_with_adjustments` — the minimal threading;
the `StepABC.step` / `StepArgs` signatures do not change. Wrapper steps
(`MultiCallStep`) forward the setter to the wrapped step, mirroring the
existing `train()` / `set_epoch()` forwarding pattern. Default is detached
everywhere; only `TrainStepper` flips it, once at build, exactly when
`StepOutputLoss.needs_uncorrected_grad`. Inference paths run under `no_grad`,
so an attached-mode stepper builds no graphs there.

## `fme/core/step/single_module.py` (modified)

```python
def step_with_adjustments(
    ...,
    stepper_state: StepperState | None = None,
    detach_corrector_deltas: bool = True,  # NEW
) -> StepOutput:
    # CHANGED — the unconditional per-tensor detach becomes:
    #   diagnostics = result.diagnostics.detach() if detach_corrector_deltas
    #   else CorrectorDiagnostics(delta=dict(result.diagnostics.delta))


class SingleModuleStep(StepABC):
    @property
    def corrector_enabled(self) -> bool:  # NEW — self._corrector.enabled
        ...

    @property
    def corrector_keep_gradient_names(self) -> frozenset[str]:  # NEW — forwards
        ...

    def step(self, args, wrapper=...) -> StepOutput:
        # CHANGED — passes detach_corrector_deltas=self._detach_corrector_deltas
        ...
```

## `fme/core/step/secondary_module.py`, `fme/core/step/radiation.py`, `fme/ace/step/fcn3.py`, `fme/core/step/multi_call.py` (modified)

The remaining `step_with_adjustments` callers — `secondary_module.py`,
`radiation.py`, `fcn3.py` — get the same three-line change as
`SingleModuleStep`: implement `corrector_enabled` /
`corrector_keep_gradient_names` from their `_corrector`, and pass
`detach_corrector_deltas=self._detach_corrector_deltas`. `MultiCallStep` does
not call `step_with_adjustments` (no corrector of its own; it already passes
the wrapped step's `corrector_diagnostics` through) — it forwards the two
properties to the wrapped step and overrides `set_detach_corrector_deltas` to
forward as well.

## `fme/ace/stepper/single_module.py` (modified)

```python
class Stepper:
    @property
    def corrector_enabled(self) -> bool:  # NEW — forwards self._step_obj
        ...

    def set_detach_corrector_deltas(self, detach: bool) -> None:  # NEW — forwards
        ...

    def build_step_output_loss(
        self,
        loss_config: StepLossConfig,
        corrector_loss: CorrectorLossConfig | None,
    ) -> StepOutputLoss:  # NEW — build_loss(...) unchanged; this wraps its result,
        ...               # pass-through StepOutputLoss when corrector_loss is None


@dataclasses.dataclass
class TrainStepperConfig:
    corrector_loss: CorrectorLossConfig | None = None  # NEW — default preserves
                                                       # current behavior


class TrainStepper(TrainStepperABC[...]):
    def __init__(self, stepper: Stepper, config: TrainStepperConfig):
        # CHANGED — always builds one StepOutputLoss so _accumulate_loss has a
        # single uniform call site:
        #   self._loss_obj = stepper.build_step_output_loss(config.loss, config.corrector_loss)
        # Build-time guards when corrector_loss is configured:
        #   - raise if not stepper.corrector_enabled (per #1273)
        #   - raise if a precorrector_optimization entry matches any name in
        #     stepper's corrector_keep_gradient_names (see divergence 7)
        #   - stepper.set_detach_corrector_deltas(False) when
        #     self._loss_obj.needs_uncorrected_grad
        ...

    def _accumulate_loss(self, ...):
        # CHANGED — keeps the yielded StepOutput instead of unwrapping .output
        # immediately; output_list still collects the unfolded .output.
        ...

    def _accumulate_step_loss(
        self,
        step_output: StepOutput,  # CHANGED — was gen_step: EnsembleTensorDict
        target_step: TensorMapping,
        step: int,
        ...
    ) -> torch.Tensor:
        # CHANGED — calls self._loss_obj(step_output, target_step, step,
        # n_ensemble, data_mask); returns result.total() so the penalty is
        # folded into the one per-step accumulate_loss call; records
        # metrics["corrector_regularization_step_{step}"] and a per-batch
        # metrics["corrector_regularization"] (mean over steps where present;
        # the trainer's existing metric averaging yields the epoch aggregate).
        ...
```

### Critical detail — accumulation seam and the hard boundary

- The penalty is folded into each optimized step's `result.total()` and
  bypasses the `get_regularizer_loss` accumulation in `train_on_batch`
  (which stays, unchanged, for the module regularizers): accumulating the
  penalty as a second `optimization.accumulate_loss` call double-backwards
  under gradient accumulation, per #1273.
- Hard boundary: training consumes deltas at the `StepOutput` level inside
  `_accumulate_loss`, never via the `StepDiagnostics` carriage
  (`fme/core/step/step_diagnostics.py`) — that carriage is detached,
  output-masked, and inference-only by design. Stated here and seam-commented
  in code.
- `Stepper.step` applies `apply_output_masking` to the diagnostics; with
  attached deltas the NaN fill now sits on a live graph. The existing loss
  path is mask-aware; the masked-output test below covers the new
  attached-delta case.

---

## Divergences from #1273

1. **Inclusion, not exclusion**: field selection is opt-in
   `names_and_prefixes` per feature (default selects nothing); #1273 used
   `exclude_names_and_prefixes` defaulting to all corrector-modified
   variables. A present config with `names_and_prefixes=None` is a build-time
   error.
2. **Error, not warn**: #1273 warn-onces unmatched matcher entries. Here,
   unmatched-vs-network-outputs is a build-time error and
   unmatched-vs-delta-keys a first-time runtime error.
3. **`CorrectorDiagnostics.detach()`**: #1273 threads a bare flag; this plan
   adds the helper and keeps the detach decision at the step seam.
4. **`StepOutput.uncorrected` added here**: #1273 assumed #1271 had delivered
   it; it does not exist on `main`.
5. **`n_ensemble` at call time**: #1273 fixes `n_ensemble` at build; here
   `StepOutputLoss.__call__` takes it as a parameter, matching how
   `_accumulate_loss` already holds it per batch.
6. **`StepOutputLossResult` carries the unweighted penalty**: #1273 folds the
   weight into `.regularization`; here `.regularization` is unweighted (for
   metrics) with `regularization_weight` applied in `total()`.
7. **keep-gradient guard made enforceable**: a 2026-07-10 maintainer decision
   held that selecting a `replace_value_keep_gradient` variable (e.g.
   `SeaIceFractionCorrection` under `keep_gradient_through_clamps`) for
   pre-corrector optimization is an error — two gradient mechanisms layered
   on one signal. Its premise has changed: under exclusion-default such a
   selection was always a deliberate per-name entry, whereas under opt-in a
   prefix entry can sweep one in unintentionally — which argues *for* keeping
   the error, since the layering can now happen silently. This plan upholds
   the error and adds the missing discovery surface
   (`keep_gradient_names` through `Correction` → `CorrectorABC` → `StepABC`);
   without it the variables are unreachable from a corrector instance and the
   decision is unenforceable.

---

## Tests

## `fme/core/test_loss.py` (modified)

```python
# Deterministic StepOutput helper: fixed output/delta tensors, a normalizer
# with known means/stds, MSE step loss — penalties computable by hand.

def test_step_output_loss_pass_through():
    # GOAL: with no features configured, result.main equals StepLoss on
    # step_output.output and total() == main.total(); needs_uncorrected_grad False.
    ...

def test_precorrector_swap_selected_only():
    # GOAL: main loss sees output − delta for selected keys only; unselected
    # corrector-modified keys and delta-absent keys use plain output.
    # PARAMETERIZE: selection entry ∈ {exact name, trailing-underscore prefix}.
    ...

def test_regularization_analytic_penalty():
    # GOAL: penalty equals the hand-computed mean of (delta/std)^2 — normalizer
    # means cancel against the zeros target.
    ...

def test_regularization_selected_channels_scale():
    # GOAL: penalty means over selected channels only; adding an unselected
    # delta key leaves it unchanged, dropping a selected key from the deltas
    # renormalizes over the present-and-selected set.
    ...

def test_total_decomposition():
    # GOAL: total() == main.total() + weight * regularization; result carries
    # the unweighted penalty.
    ...

def test_needs_uncorrected_grad_per_config():
    # GOAL: property truth table over {precorrector, regularization} presence.
    ...

def test_runtime_first_check_and_absent_key_behavior():
    # GOAL: first non-empty-delta call errors on an entry matching no delta
    # key; empty-delta calls warn once (no spam) and do not consume the check.
    ...
```

## `fme/core/corrector/test_loss.py` (new)

```python
def test_corrector_loss_config_both_none_raises():
    # GOAL: __post_init__ error when neither feature is configured.
    ...

def test_feature_config_names_none_is_build_error():
    # GOAL: a present feature config with names_and_prefixes=None fails build.
    ...

def test_regularization_config_rejects_loss_types():
    # GOAL: EnsembleLoss / NaN types and any global_mean_type are rejected.
    ...

def test_build_errors_on_entry_matching_no_output_name():
    # GOAL: build-time validation against network output names.
    ...
```

## `fme/core/step/test_output.py` (modified)

```python
def test_uncorrected_subtracts_delta_over_delta_keys():
    # GOAL: uncorrected == output − delta on delta keys, output elsewhere;
    # empty diagnostics returns output values unchanged.
    ...
```

## `fme/core/step/test_step.py` (modified)

```python
def test_step_with_adjustments_detach_flag():
    # GOAL: default detaches deltas (grad_fn is None, matching today);
    # detach_corrector_deltas=False leaves deltas on the graph while the
    # corrected output is unaffected either way.
    ...
```

## `fme/ace/stepper/test_single_module.py` (modified)

```python
# Deterministic corrector: a correction adding a fixed offset to one variable.

def test_train_on_batch_precorrector_equivalence():
    # GOAL: with pre-corrector optimization selecting the corrected variable,
    # metrics["loss"] equals the same stepper's loss with no corrector at all;
    # returned predictions stay fully corrected.
    ...

def test_gradient_flows_through_correction_when_configured():
    # GOAL: with corrector_loss configured, parameter grads differ from the
    # detached baseline (delta attached); without corrector_loss the plain
    # prediction path still detaches deltas.
    ...

def test_corrector_regularization_gradient_accumulation():
    # GOAL: penalty folded into per-step totals backpropagates under gradient
    # accumulation (no double-backward); one accumulate_loss per optimized step.
    ...

def test_corrector_loss_requires_enabled_corrector():
    # GOAL: TrainStepper build raises when corrector_loss is configured and
    # step.corrector_enabled is False.
    ...

def test_keep_gradient_selection_raises_at_build():
    # GOAL: a precorrector entry matching a keep_gradient name errors at build.
    # PARAMETERIZE: entry ∈ {exact name, prefix that sweeps it in}.
    ...

def test_masked_output_with_corrector_loss_finite():
    # GOAL: masked-output config + corrector_loss yields finite losses and
    # parameter gradients through the NaN-filling apply_output_masking path.
    ...

def test_epoch_scheduled_corrector_first_check():
    # GOAL: disabled epochs neither error nor repeat the empty-delta warning;
    # the delta-key check runs at the first enabled optimized step; per-step
    # regularization metrics appear only from that epoch on.
    ...

def test_corrector_regularization_metrics():
    # GOAL: corrector_regularization_step_{i} per optimized step plus the
    # per-batch corrector_regularization mean.
    ...

def test_both_features_compose_in_one_run():
    # GOAL: pre-corrector optimization and regularization configured together
    # in one train_on_batch: main loss sees the swapped targets, the penalty
    # is added, and both metric families appear (per #1273's composition
    # acceptance criterion).
    ...
```

---

## Open Questions

- Keep the build-time error for keep-gradient selections (divergence 5), or
  document-only to avoid growing the `Correction` API with
  `keep_gradient_names`?
- `Stepper.build_step_output_loss` as a second method beside `build_loss`,
  or fold `corrector_loss` into `build_loss` behind a default-`None`
  parameter?
