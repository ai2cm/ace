# Config-selected pre-corrector optimization and corrector regularization

Adds a training-only `corrector_loss` config that consumes the correction
deltas already carried on `StepOutput`: (1) *pre-corrector optimization* — for
selected corrector-modified variables the main loss sees the pre-corrector
prediction `output − delta`; (2) *corrector regularization* — a penalty pushing
`delta` toward zero in loss-normalized space. Field selection is pure opt-in
via `NameAndPrefixMatcher` entries. Both features extend the existing
`StepLoss`/`LossOutput` — no new loss classes. All name validation happens
when the run starts: at construction the corrector discovers the delta keys it
will produce by running once on fake data. Implements the design of #1273,
with the divergences listed at the end.

Throughout, "active" means the step's corrector produced a non-empty delta
dict on the current step; an `EpochScheduledCorrector` on a disabled epoch
returns empty diagnostics and is inactive.

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
class Correction(Protocol):
    # Docstring gains a contract: the key set a correction returns may depend
    # on its config and on which keys are present in its inputs, never on
    # tensor values. Modified-name discovery below relies on this.

    @property
    def keep_gradient_names(self) -> frozenset[str]:  # NEW — names corrected via
        ...                                           # straight-through clamps


class CorrectorABC(abc.ABC):
    @property
    def modified_names(self) -> frozenset[str] | None:  # NEW — the delta keys this
        return None                                     # corrector produces when
                                                        # active; None before discovery
    @property
    def keep_gradient_names(self) -> frozenset[str]:  # NEW
        return frozenset()

    def discover_modified_names(
        self,
        input_names: Collection[str],
        gen_names: Collection[str],
        forcing_names: Collection[str],
        img_shape: tuple[int, int],
    ) -> None:  # NEW — default no-op; modified_names stays None
        ...


class CorrectionSequence(CorrectorABC):
    def discover_modified_names(self, ...) -> None:
        # One __call__ on zero tensors of shape (1, *img_shape) keyed by the
        # given names; records frozenset(result.diagnostics.delta.keys()).
        ...

    @property
    def modified_names(self) -> frozenset[str] | None: ...

    @property
    def keep_gradient_names(self) -> frozenset[str]:  # union over corrections
        ...


class EpochScheduledCorrector(CorrectorABC):
    # discover_modified_names / modified_names / keep_gradient_names forwarded
    # to the wrapped corrector, independent of the epoch's disabled state. NEW
```

### Critical detail — the discovery pass

Fake-data values can go NaN/inf inside the budget corrections (they divide by
global means); that is harmless, because the *key set* every current
correction returns depends only on its config and on which keys exist in its
inputs — verified across `utils.py`, `atmosphere.py`, `ocean.py`, `ice.py` —
and the `Correction` docstring contract above pins that for future
corrections. The pass runs no network and costs one correction sweep over
`(1, H, W)` zeros at build.

Each `Correction` gains `keep_gradient_names` (default empty). `ForcePositive`
(`fme/core/corrector/utils.py`) returns its names when `keep_gradient` is set;
`SeaIceFractionCorrection` (`fme/core/corrector/ocean.py`) returns the fields
it clamps when `keep_gradient` is set. Today those names are unreachable from
a corrector instance (`CorrectionSequence._corrections` is private). No
corrector's numerical behavior changes.

## `fme/core/step/step.py` (modified)

```python
class StepABC(abc.ABC):
    @property
    @abc.abstractmethod
    def corrector(self) -> CorrectorABC | None:  # NEW — introspection surface for
        ...                                      # build-time corrector_loss validation

    def set_detach_corrector_deltas(self, detach: bool) -> None:  # NEW — concrete;
        self._detach_corrector_deltas = detach  # default True, set in __init__
```

One read property instead of per-fact forwards (`corrector_enabled`,
`corrector_keep_gradient_names`, `corrector_modified_names`): validation reads
`corrector.modified_names` / `keep_gradient_names` directly, and future
corrector surfaces don't touch `StepABC`.

### Critical detail — detach threading

The detach flag is step-level *state*, not a `step()` parameter: `StepABC`
stores `_detach_corrector_deltas = True` in its existing `__init__`, and each
concrete step passes it into `step_with_adjustments` — the minimal threading;
the `StepABC.step` / `StepArgs` signatures do not change. Detach stays at the
step seam rather than inside `CorrectionSequence.__call__` because the seam
already post-processes the diagnostics (the ocean-overlap guard and the
prescribed-prognostic filter) and covers every corrector type uniformly.
Wrapper steps (`MultiCallStep`) forward the setter and the `corrector`
property to the wrapped step, mirroring the existing `train()` / `set_epoch()`
forwarding pattern. Default is detached everywhere; only the train stepper
flips it, once at build, exactly when `StepLoss.needs_corrector_deltas`.
Inference paths run under `no_grad`, so an attached-mode stepper builds no
graphs there.

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


class SingleModuleStepConfig(StepConfigABC):
    def get_step(self, ...) -> "SingleModuleStep":
        # CHANGED — after corrector = self.corrector.get_corrector(dataset_info):
        #   corrector.discover_modified_names(
        #       input_names=self.input_names, gen_names=self.output_names,
        #       forcing_names=self.next_step_input_names,
        #       img_shape=dataset_info.img_shape,
        #   )


class SingleModuleStep(StepABC):
    @property
    def corrector(self) -> CorrectorABC | None:  # NEW — self._corrector
        ...

    def step(self, args, wrapper=...) -> StepOutput:
        # CHANGED — passes detach_corrector_deltas=self._detach_corrector_deltas
        ...
```

The loss-visible delta set is `modified_names` minus
`prescribed_prognostic_names` — `step_with_adjustments` filters prescribed
prognostics out of the delta after the corrector call — and the validation
below checks entries against that filtered set.

## `fme/core/step/secondary_module.py`, `fme/core/step/radiation.py`, `fme/ace/step/fcn3.py`, `fme/core/step/multi_call.py` (modified)

The remaining `step_with_adjustments` callers — `secondary_module.py`,
`radiation.py`, `fcn3.py` — get the same three-line change as
`SingleModuleStep`: run discovery in `get_step`, expose `corrector`, and pass
`detach_corrector_deltas=self._detach_corrector_deltas`. `MultiCallStep` does
not call `step_with_adjustments` (no corrector of its own; it already passes
the wrapped step's `corrector_diagnostics` through) — it forwards the
`corrector` property and `set_detach_corrector_deltas` to the wrapped step.

## `fme/core/loss.py` (modified)

```python
class LossOutput:
    def __init__(
        self,
        losses: list[LossComponent],
        channel_names: list[str],
        mask: torch.Tensor | None = None,
        corrector_regularization: torch.Tensor | None = None,  # NEW — unweighted
        corrector_regularization_weight: float = 1.0,          # NEW
    ): ...

    def total(self) -> torch.Tensor:
        # CHANGED — adds weight * corrector_regularization when present.

    def scale(self, weight: float) -> "LossOutput":
        # CHANGED — carries the two new fields through unchanged: the per-step
        # sqrt decay applies to the main loss only, never the penalty.


@dataclasses.dataclass
class StepLossCorrectorArgs:  # NEW — lives here so corrector/loss.py imports
    precorrector_names: list[str] | None  # loss.py, never the reverse
    regularizer: WeightedMappingLoss | None
    regularization_weight: float


class StepLoss(torch.nn.Module):
    def __init__(
        self,
        loss: WeightedMappingLoss,
        sqrt_loss_decay_constant: float = 0.0,
        corrector_args: StepLossCorrectorArgs | None = None,  # NEW
    ): ...

    @property
    def needs_corrector_deltas(self) -> bool:  # NEW — either feature configured;
        ...  # both differentiate through delta

    def forward(
        self,
        predict_dict: TensorMapping,
        target_dict: TensorMapping,
        step: int,
        data_mask: TensorMapping | None = None,
        deltas: TensorMapping | None = None,  # NEW — after data_mask: the coupled
    ) -> LossOutput: ...                      # stepper calls positionally with three
                                              # args and stays untouched
```

### Critical detail — `forward` algorithm

- `deltas` `None` or empty, or no `corrector_args` configured: exactly
  today's behavior. Epoch-scheduled-off steps land here — no swap, no
  penalty, no per-step metric.
- Non-empty `deltas`: every name in `precorrector_names` and in the
  regularizer's packed name list must be present, else raise — an active
  corrector must produce deltas for every selected name. Cost, accepted: a
  corrector whose modified set varies step-to-step once active is unsupported
  with `corrector_loss`; no current correction does that (key sets are
  config- and key-presence-dependent only, per the `Correction` contract).
  Build-time validation (below) makes this raise unreachable for current
  correctors; it remains as the drift guard.
- Swap: the prediction fed to the main loss is
  `predict_dict[k] − deltas[k]` for `k in precorrector_names`, plain
  `predict_dict[k]` otherwise; targets untouched;
  `main = self.loss(swapped, target_dict, data_mask).scale(step_weight)`.
- Penalty: `regularizer(selected_deltas, targets)` with
  `targets[k] = torch.where(deltas[k].isnan(), nan, 0.0)` — copying the
  delta's NaN pattern onto the zeros target makes masked (NaN-filled) points
  drop through `WeightedMappingLoss`'s existing NaN-target zeroing. With an
  affine normalizer the means cancel, so this penalizes `delta/std`, mean
  over the selected channels. No per-step decay on the penalty — only the
  weight scales it, unlike the main loss's `sqrt_loss_step_decay` (config
  docstring notes this).
- Returns `main` with the unweighted penalty and its weight attached.

`StepLossConfig` is unchanged; its `build` gains a default-`None`
`corrector_args: StepLossCorrectorArgs | None` parameter passed into
`StepLoss`. `fme/core/loss.py` imports nothing from `fme/core/corrector/`.

## `fme/core/corrector/loss.py` (new)

```python
@dataclasses.dataclass
class PreCorrectorOptimizationConfig:
    names_and_prefixes: list[str] | None = None

    def __post_init__(self):
        # None ⇒ error: configuring the feature while selecting nothing is a
        # contradiction, not a no-op.


@dataclasses.dataclass
class CorrectorRegularizationConfig:
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    weight: float = 1.0
    names_and_prefixes: list[str] | None = None

    def __post_init__(self):
        # names_and_prefixes None ⇒ error (as above);
        # reject EnsembleLoss / NaN loss types and any global_mean_type
        # (per #1273); reject weight <= 0.
        # Config docstrings document: no per-step decay on the penalty, and
        # that the two features may be enabled together (a supported, tested
        # configuration, per #1273's acceptance criteria).


@dataclasses.dataclass
class CorrectorLossConfig:
    precorrector_optimization: PreCorrectorOptimizationConfig | None = None
    regularization: CorrectorRegularizationConfig | None = None

    def __post_init__(self):
        # error when both are None: configuring corrector_loss while selecting
        # no feature is a contradiction, not a no-op.

    def build(
        self,
        corrector: CorrectorABC | None,
        prescribed_prognostic_names: Collection[str],
        normalizer: StandardNormalizer,
        gridded_operations: GriddedOperations | None,
        channel_dim: int = -3,
    ) -> StepLossCorrectorArgs:
        # Validates (below), then builds the regularizer as
        # WeightedMappingLoss(loss=self.regularization.loss.build(...),
        # weights={}, out_names=selection.matched(loss_visible_names),
        # normalizer=normalizer, channel_dim=channel_dim) — fully constructed
        # at build, no factory.
```

### Critical detail — build-time validation

All name validation runs here, when the run starts; no runtime name check
remains. With `loss_visible_names = corrector.modified_names −
prescribed_prognostic_names`, `build` raises when:

- `corrector` is `None` or `modified_names` is empty — `corrector_loss` is
  configured with nothing to consume (subsumes #1273's corrector-enabled
  build guard and its empty-modified-set warn-once);
- `modified_names` is `None` — the step type never ran discovery (a
  programming error, not a user-config error);
- any entry of either feature is unmatched against `loss_visible_names`
  (`NameAndPrefixSelection.unmatched_entries`) — this subsumes a separate
  check against network output names, since every delta key is a network
  output name;
- any `precorrector_optimization` entry matches a name in
  `corrector.keep_gradient_names` (divergence 6). Proposed extension, review
  settles it: the same error for `regularization` entries — a
  straight-through clamp's delta is detached, so its penalty carries no
  gradient and the feature is silently inert.

## `fme/ace/stepper/single_module.py` (modified)

```python
class Stepper:
    def build_loss(
        self,
        loss_config: StepLossConfig,
        corrector_loss: CorrectorLossConfig | None = None,  # NEW — folded in;
    ) -> StepLoss:                                          # no second method
        # CHANGED — when corrector_loss is given:
        #   corrector_args = corrector_loss.build(
        #       self._step_obj.corrector,
        #       prescribed_prognostic_names=...,
        #       normalizer=loss_normalizer,
        #       gridded_operations=self._dataset_info.gridded_operations,
        #       channel_dim=self.CHANNEL_DIM)
        # then loss_config.build(..., corrector_args=corrector_args), and
        #   self._step_obj.set_detach_corrector_deltas(False)
        # when the built loss needs_corrector_deltas — detaching would
        # silently zero part of the correction gradient.


@dataclasses.dataclass
class TrainStepperConfig:
    corrector_loss: CorrectorLossConfig | None = None  # NEW — default preserves
                                                       # current behavior


class TrainStepper(TrainStepperABC[...]):
    def __init__(self, stepper: Stepper, config: TrainStepperConfig):
        # CHANGED — one line:
        #   self._loss_obj = stepper.build_loss(config.loss, config.corrector_loss)
        ...

    def _accumulate_loss(self, ...):
        # CHANGED — keeps the yielded StepOutput; when
        # self._loss_obj.needs_corrector_deltas, unfolds the delta dict the
        # same way as .output (unfold_ensemble_dim) and passes it down.
        ...

    def _accumulate_step_loss(
        self,
        gen_step: EnsembleTensorDict,
        target_step: TensorMapping,
        step: int,
        ...,
        deltas: TensorMapping | None = None,  # NEW
    ) -> torch.Tensor:
        # CHANGED — self._loss_obj(gen_step, target_step, step=step,
        # data_mask=data_mask, deltas=deltas); returns result.total() so the
        # penalty is folded into the one per-step accumulate_loss call;
        # records metrics["corrector_regularization_step_{step}"] and a
        # per-batch metrics["corrector_regularization"] from the result's
        # unweighted penalty (mean over steps; the trainer's existing metric
        # averaging yields the epoch aggregate).
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
  attached deltas the NaN fill now sits on a live graph. The regularizer
  drops NaN points via the target trick above; the masked-output test below
  covers the whole path.
- The coupled stepper's `_build_loss` calls `build_loss` per component with
  no `corrector_loss` and stays unchanged (`fme.coupled` support is out of
  scope).

---

## Divergences from #1273

1. **Inclusion, not exclusion**: field selection is opt-in
   `names_and_prefixes` per feature (default selects nothing); #1273 used
   `exclude_names_and_prefixes` defaulting to all corrector-modified
   variables. A present config with `names_and_prefixes=None` is an error.
2. **Extends `StepLoss`/`LossOutput`** instead of adding
   `StepOutputLoss`/`StepOutputLossResult`: `StepLoss.forward` grows an
   optional `deltas` argument and `LossOutput` carries the unweighted penalty
   with the weight applied in `total()`.
3. **All name validation at build, none at runtime**: the corrector discovers
   its delta keys at construction by a dummy pass on fake data; #1273
   warn-onced unmatched entries at runtime. Consequence: an active corrector
   must produce deltas for every selected name (a partial set raises), so a
   corrector whose modified set varies step-to-step once active is
   unsupported with `corrector_loss` — and the `corrector_regularization`
   metric is comparable across epochs whenever the corrector is active.
4. **`CorrectorDiagnostics.detach()`**: #1273 threads a bare flag; this plan
   adds the helper and keeps the detach decision at the step seam.
5. **No `StepOutput.uncorrected`**: #1273 assumed #1271 had delivered it; it
   does not exist on `main`, and with the swap happening inside
   `StepLoss.forward` on plain dicts the convenience would be dead code.
6. **keep-gradient guard made enforceable**: a 2026-07-10 maintainer decision
   held that selecting a `replace_value_keep_gradient` variable (e.g.
   `SeaIceFractionCorrection` under `keep_gradient_through_clamps`) for
   pre-corrector optimization is an error — two gradient mechanisms layered
   on one signal. Its premise has changed: under exclusion-default such a
   selection was always a deliberate per-name entry, whereas under opt-in a
   prefix entry can sweep one in unintentionally — which argues *for* keeping
   the error, since the layering can now happen silently. This plan upholds
   the error and adds the missing discovery surface (`keep_gradient_names`
   through `Correction` → `CorrectorABC`); without it the variables are
   unreachable from a corrector instance and the decision is unenforceable.

---

## Tests

## `fme/core/test_loss.py` (modified)

```python
# Deterministic helper: fixed prediction/target/delta dicts, a normalizer
# with known means/stds, MSE step loss — penalties computable by hand.

def test_step_loss_without_corrector_args_unchanged():
    # GOAL: with no corrector_args, forward matches today's behavior exactly,
    # deltas is ignored, and needs_corrector_deltas is False.
    ...

def test_precorrector_swap_selected_only():
    # GOAL: main loss sees prediction − delta for the configured names only;
    # other keys use the plain prediction; targets untouched.
    ...

def test_regularization_analytic_penalty():
    # GOAL: penalty equals the hand-computed mean of (delta/std)^2 — normalizer
    # means cancel against the zeros target.
    ...

def test_regularization_masked_points_drop():
    # GOAL: NaN-filled delta points contribute nothing; penalty and gradients
    # stay finite.
    ...

def test_total_decomposition():
    # GOAL: total() == main total + weight * penalty; the result carries the
    # unweighted penalty; scale() scales the main loss only.
    ...

def test_missing_selected_delta_raises():
    # GOAL: a non-empty delta dict lacking a selected name raises (the
    # complete-delta-set rule).
    ...

def test_empty_deltas_inert():
    # GOAL: empty deltas ⇒ no swap, no penalty, LossOutput matches the
    # unconfigured result.
    ...

def test_needs_corrector_deltas_per_config():
    # GOAL: property truth table over {precorrector, regularization} presence.
    ...
```

## `fme/core/corrector/test_loss.py` (new)

```python
def test_config_post_init_errors():
    # GOAL: both features None; a present feature with names_and_prefixes=None;
    # weight <= 0; EnsembleLoss / NaN / global_mean_type — each raises in
    # __post_init__.
    ...

def test_build_errors_on_entry_matching_no_modified_name():
    # GOAL: an entry matching no corrector-modified name raises at build.
    # PARAMETERIZE: entry ∈ {exact name, trailing-underscore prefix}.
    ...

def test_build_errors_without_corrector_or_discovery():
    # GOAL: corrector None, modified_names empty, and modified_names None
    # (discovery never ran) each raise with distinct messages.
    ...

def test_build_excludes_prescribed_prognostics():
    # GOAL: an entry matching only a prescribed prognostic name raises —
    # validation runs against the loss-visible set.
    ...

def test_keep_gradient_selection_raises_at_build():
    # GOAL: a precorrector entry matching a keep_gradient name errors at
    # build. PARAMETERIZE: entry ∈ {exact name, prefix that sweeps it in}.
    ...

def test_build_regularizer_packs_matched_names():
    # GOAL: the built WeightedMappingLoss packs exactly
    # selection.matched(loss_visible_names); a prefix entry matches all its
    # level names.
    ...
```

## `fme/core/corrector/test_registry.py` (modified)

```python
def test_discovery_records_modified_names():
    # GOAL: discover_modified_names on a CorrectionSequence records exactly
    # the delta keys a real call produces; modified_names is None before.
    ...

def test_discovery_forwards_through_epoch_schedule():
    # GOAL: EpochScheduledCorrector discovery and modified_names are
    # independent of the disabled-epoch state.
    ...

def test_keep_gradient_names_union():
    # GOAL: CorrectionSequence unions per-correction keep_gradient_names;
    # empty when no correction keeps gradients.
    ...
```

## `fme/core/step/test_step.py` (modified)

```python
def test_step_with_adjustments_detach_flag():
    # GOAL: default detaches deltas (grad_fn is None, matching today);
    # detach_corrector_deltas=False leaves deltas on the graph while the
    # corrected output is unaffected either way.
    ...

def test_get_step_runs_discovery():
    # GOAL: a built step's corrector.modified_names is populated.
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

def test_masked_output_with_corrector_loss_finite():
    # GOAL: masked-output config + corrector_loss yields finite losses and
    # parameter gradients through the NaN-filling apply_output_masking path.
    ...

def test_epoch_scheduled_corrector():
    # GOAL: build-time validation passes via discovery even when epoch 0 is
    # disabled; disabled epochs are inert (no swap, no penalty, no metric, no
    # error); the penalty and metrics appear from the first enabled epoch.
    ...

def test_corrector_regularization_metrics():
    # GOAL: corrector_regularization_step_{i} per optimized step plus the
    # per-batch corrector_regularization mean.
    ...

def test_both_features_together():
    # GOAL: both features configured in one train_on_batch: the main loss sees
    # the swapped predictions, the penalty is added, and both metric families
    # appear (per #1273's acceptance criteria).
    ...
```

---

## Open Questions

- The `CorrectorRegularizationConfig` restrictions — rejecting
  `EnsembleLoss` / `NaN` loss types, any `global_mean_type`, and
  `weight <= 0` — await @mcgibbon's review (thread on this PR).
- `StepABC.corrector` as a single introspection property, or the per-fact
  forwards (`corrector_enabled`, `corrector_keep_gradient_names`,
  `corrector_modified_names`) of the previous revision?
- Extend the keep-gradient build error to `regularization` entries (proposed
  above), or keep it precorrector-only as originally decided?
