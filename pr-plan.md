# Config-selected pre-corrector optimization and corrector regularization

Adds a training-only `corrector_loss` config that consumes the correction
deltas already carried on `StepOutput`: (1) *pre-corrector optimization* — for
selected corrector-modified variables the main loss sees the pre-corrector
network output `output − delta`; (2) *corrector regularization* — a penalty
pushing `delta` toward zero in loss-normalized space. Field selection is pure
opt-in via `NameAndPrefixMatcher` entries. All name validation happens when the
run starts: a corrector discovers the delta keys it will produce by running
once on fake data at construction. Implements the design of #1273, with the
divergences listed at the end.

Throughout, "active" means the step's corrector produced a non-empty delta dict
on the current step; an `EpochScheduledCorrector` on a disabled epoch returns
empty diagnostics and is inactive.

Straight-through corrections (`keep_gradient`) are out of scope here: a
straight-through delta is detached, so it carries no gradient and cannot be
regularized. That feature is removed in a follow-up PR (thread on
`fme/core/corrector/loss.py`), and this PR adds nothing to discover or guard
against it.

---

## `fme/core/name_and_prefix_matcher.py` (modified)

```python
class NameAndPrefixMatcher:
    ...  # unchanged

@dataclasses.dataclass(frozen=True)
class NameAndPrefixSelection:  # NEW — matcher plus its entries, for validation
    entries: tuple[str, ...]

    @property
    def matcher(self) -> NameAndPrefixMatcher: ...

    def matched(self, names: Iterable[str]) -> list[str]:
        """Names (sorted) that match any entry."""

    def unmatched_entries(self, names: Iterable[str]) -> list[str]:
        """Entries that match none of ``names`` — the validation primitive."""
```

`NameAndPrefixMatcher` has no per-entry reporting today; build-time validation
needs it, so the entry list is kept alongside the matcher rather than adding
state to the matcher itself.

## `fme/core/corrector/registry.py` (modified)

```python
class Correction(Protocol):
    # Docstring gains a contract: the key set a correction returns may depend
    # on its config and on which keys are present in its inputs, never on
    # tensor values. Modified-name discovery below relies on this.


class CorrectorConfigProtocol(Protocol):
    def get_corrector(
        self,
        dataset_info: DatasetInfo,
        input_names: Collection[str],   # NEW
        gen_names: Collection[str],     # NEW
        forcing_names: Collection[str], # NEW
    ) -> "CorrectorABC":
        # CHANGED — the one wrapper around _get_corrector runs discovery on the
        # constructed corrector before returning it, so modified_names is set
        # for every corrector type and no caller updates it statefully.


class CorrectorABC(abc.ABC):
    @property
    def modified_names(self) -> frozenset[str]:  # NEW — the delta keys this
        return frozenset()                       # corrector produces when active


class CorrectionSequence(CorrectorABC):
    # __init__ runs one __call__ on zero tensors of shape (1, *img_shape) keyed
    # by the given names and records frozenset(result.diagnostics.delta.keys()).


class EpochScheduledCorrector(CorrectorABC):
    # modified_names forwarded to the wrapped corrector, independent of the
    # epoch's disabled state. NEW
```

### Critical detail — the discovery pass

Discovery happens during construction and takes `img_shape` from
`dataset_info`, so there is no post-construction mutation and no `img_shape`
argument: `modified_names` is a plain `frozenset` rather than
`frozenset | None`, and the "discovery never ran" error disappears. Under
spatial parallelism the fake data is scattered with
`Distributed.scatter_spatial`, the same slicing real data gets, because
gridded operations slice their area weights to the local chunk.

Fake-data values can go NaN/inf inside the budget corrections (they divide by
global means); that is harmless, because the *key set* every current correction
returns depends only on its config and on which keys exist in its inputs —
verified across `utils.py`, `atmosphere.py`, `ocean.py`, `ice.py` — and the
`Correction` docstring contract pins that for future corrections. The pass runs
no network and costs one correction sweep over `(1, H, W)` zeros at build.

## `fme/core/step/step.py` (modified)

```python
class StepABC(abc.ABC):
    @property
    def corrector_modified_names(self) -> frozenset[str]:  # NEW — concrete,
        return frozenset()  # overridden by steps that own a corrector
```

The step exposes only the fact the loss build needs, not the corrector object:
build-time validation reads names, and no other corrector surface reaches
`StepABC`.

## `fme/core/step/single_module.py` (modified)

```python
def step_with_adjustments(...) -> StepOutput:
    # CHANGED — the unconditional per-tensor detach of the corrector deltas is
    # removed; deltas stay on the graph. The prescribed-prognostic filter and
    # the ocean-overlap guard are untouched.


class SingleModuleStepConfig(StepConfigABC):
    def get_step(self, ...) -> "SingleModuleStep":
        # CHANGED — self.corrector.get_corrector(dataset_info,
        #   input_names=self.input_names, gen_names=self.output_names,
        #   forcing_names=self.next_step_input_names)


class SingleModuleStep(StepABC):
    @property
    def corrector_modified_names(self) -> frozenset[str]:  # NEW
        ...
```

### Critical detail — why nothing detaches

Attaching the deltas everywhere frees no memory, so no flag, threading, or
`CorrectorDiagnostics.detach()` helper is needed. `delta[name] = corrected[name]
− network_output[name]`; `corrected` is a prognostic step output feeding both
the next step's input state and the main loss, so its subgraph is retained
until backward regardless, and `detach()` returns a storage-sharing view that
never freed values. Non-optimized steps run under `torch.no_grad()`, where the
detach was a no-op anyway. The `StepDiagnostics` carriage
(`fme/core/step/step_diagnostics.py`) stays detached on its own terms.

The loss-visible delta set is `corrector_modified_names` minus
`prescribed_prognostic_names` — `step_with_adjustments` drops prescribed
prognostics from the delta after the prescribed overwrite, because the delta no
longer describes the returned output — and the validation below checks entries
against that filtered set.

## `fme/core/step/secondary_module.py`, `fme/core/step/radiation.py`, `fme/ace/step/fcn3.py`, `fme/core/step/multi_call.py` (modified)

The other `step_with_adjustments` callers pass the name sets into
`get_corrector` and expose `corrector_modified_names`. `MultiCallStep` has no
corrector of its own and forwards the property to the wrapped step.

## `fme/core/loss.py` (unchanged)

`StepLoss`, `StepLossConfig`, `LossOutput`, and `WeightedMappingLoss` are as on
`main`: `LossOutput` stays a per-channel reduction container, and
`StepLoss.forward(predict_dict, target_dict, step, data_mask)` knows only
predictions, targets, and the step decay. Everything corrector-aware lives in
the new module below, which already imports from here.

## `fme/core/corrector/loss.py` (new)

```python
@dataclasses.dataclass
class PreCorrectorOptimizationConfig:
    names_and_prefixes: list[str] | None = None

    def __post_init__(self):
        # None or empty ⇒ error: configuring the feature while selecting
        # nothing is a contradiction, not a no-op.


@dataclasses.dataclass
class CorrectorRegularizationConfig:
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    weight: float = 1.0
    names_and_prefixes: list[str] | None = None

    def __post_init__(self):
        # names_and_prefixes None or empty ⇒ error (as above);
        # reject EnsembleLoss / NaN loss types and any global_mean_type
        # (per #1273); reject weight <= 0.
        # Docstrings document: no per-step decay on the penalty, and that the
        # two features may be enabled together (supported and tested).


@dataclasses.dataclass
class CorrectorLossConfig:
    precorrector_optimization: PreCorrectorOptimizationConfig | None = None
    regularization: CorrectorRegularizationConfig | None = None

    def __post_init__(self):
        # error when both are None: configuring corrector_loss while selecting
        # no feature is a contradiction, not a no-op.

    def build(
        self,
        corrector_modified_names: frozenset[str],
        prescribed_prognostic_names: Collection[str],
        normalizer: StandardNormalizer,
        gridded_operations: GriddedOperations | None,
        channel_dim: int = -3,
    ) -> "CorrectorLoss":  # CHANGED from a field bag to a behavior-bearing object
        ...


class CorrectorLoss(torch.nn.Module):  # NEW
    """The corrector-delta half of the training loss."""

    def __init__(
        self,
        precorrector_names: list[str] | None,
        regularizer: WeightedMappingLoss | None,
        regularization_weight: float,
    ): ...

    def pre_corrector_outputs(
        self, predict_dict: TensorMapping, deltas: TensorMapping
    ) -> TensorDict:
        """``predict_dict[k] − deltas[k]`` for the selected names, else
        ``predict_dict[k]``; raises when a selected name is missing."""

    def regularization(self, deltas: TensorMapping) -> LossOutput | None:
        """Penalty over the selected deltas, per channel."""

    @property
    def regularization_weight(self) -> float: ...


class StepOutputLoss(torch.nn.Module):  # NEW
    """`StepLoss` plus the corrector-delta terms of `StepOutput`."""

    def __init__(self, step_loss: StepLoss, corrector_loss: CorrectorLoss | None): ...

    def forward(
        self,
        predict_dict: TensorMapping,
        target_dict: TensorMapping,
        step: int,
        data_mask: TensorMapping | None = None,
        deltas: TensorMapping | None = None,
    ) -> "StepOutputLossOutput": ...


@dataclasses.dataclass
class StepOutputLossOutput:  # NEW
    main: LossOutput
    corrector_regularization: LossOutput | None = None
    corrector_regularization_weight: float = 1.0

    def total(self) -> torch.Tensor:
        """``main.total() + weight * corrector_regularization.total()``."""

    def get_channel_losses(self) -> dict[str, ChannelLossInfo]:
        """Delegates to ``main``."""

    def get_corrector_channel_losses(self) -> dict[str, ChannelLossInfo]:
        """Per-channel penalties, empty when no penalty."""
```

`StepOutputLoss` lives here rather than in `fme/core/loss.py` so that
`fme/core/loss.py` keeps importing nothing from `fme/core/corrector/`.

### Critical detail — `forward` order and inert paths

- `deltas` `None`/empty, or no `corrector_loss`: `main` is exactly today's
  `StepLoss` result, no penalty, no per-step metric. Epoch-disabled correctors
  land here.
- Otherwise: pre-corrector outputs **first**, so `StepLoss` never sees a delta;
  then the main loss; then the penalty from the *original* deltas.
- Missing-name rule: with a non-empty delta dict, every name selected by either
  feature must be present, else raise — an active corrector must produce deltas
  for every selected name. Cost, accepted: a corrector whose modified set
  varies step-to-step once active is unsupported with `corrector_loss`; no
  current correction does that. Build-time validation makes this unreachable
  for current correctors; it remains as the drift guard.
- Penalty: `regularizer(selected_deltas, targets)` with
  `targets[k] = torch.where(deltas[k].isnan(), nan, 0.0)` — copying the delta's
  NaN pattern onto the zeros target makes masked (NaN-filled) points drop
  through `WeightedMappingLoss`'s existing NaN-target zeroing. With an affine
  normalizer the means cancel, so this penalizes `delta/std`, mean over the
  selected channels. Only the weight scales it: unlike the main loss it takes
  no `sqrt_loss_step_decay`.

### Critical detail — build-time validation

All name validation runs here, when the run starts; no runtime name check
remains. With
`loss_visible_names = corrector_modified_names − prescribed_prognostic_names`,
`build` raises when:

- `corrector_modified_names` is empty — `corrector_loss` is configured with
  nothing to consume (subsumes #1273's corrector-enabled build guard and its
  empty-modified-set warn-once);
- any entry of either feature is unmatched against `loss_visible_names`
  (`NameAndPrefixSelection.unmatched_entries`) — this subsumes a separate check
  against network output names, since every delta key is a network output name.

The unmatched-entry error names the reason rather than restating set
arithmetic: a prescribed-prognostic entry is reported as such, because
`step_with_adjustments` drops that delta after the prescribed overwrite, so it
never reaches the loss.

## `fme/ace/stepper/single_module.py` (modified)

```python
class Stepper:
    def build_loss(self, loss_config: StepLossConfig) -> StepLoss:
        ...  # unchanged; the coupled stepper's per-component calls stay as they are

    def build_corrector_loss(  # NEW
        self, corrector_loss: CorrectorLossConfig | None
    ) -> CorrectorLoss | None:
        # corrector_loss.build(self._step_obj.corrector_modified_names,
        #   prescribed_prognostic_names=self.get_prescribed_prognostic_names(),
        #   normalizer=self.get_loss_normalizer(),
        #   gridded_operations=self._dataset_info.gridded_operations,
        #   channel_dim=self.CHANNEL_DIM)


@dataclasses.dataclass
class TrainStepperConfig:
    corrector_loss: CorrectorLossConfig | None = None  # NEW — default preserves
                                                       # current behavior


@dataclasses.dataclass
class _AccumulatedLoss:  # NEW — replaces the tuple returned by _accumulate_loss
    output_list: list[EnsembleTensorDict]
    per_channel_losses: dict[str, ChannelLossInfo] | None
    corrector_regularization: torch.Tensor | None  # mean penalty over steps


class TrainStepper(TrainStepperABC[...]):
    def __init__(self, stepper: Stepper, config: TrainStepperConfig):
        # CHANGED — self._loss_obj = StepOutputLoss(
        #     stepper.build_loss(config.loss),
        #     stepper.build_corrector_loss(config.corrector_loss))

    def _accumulate_loss(self, ...) -> _AccumulatedLoss:
        # CHANGED — keeps the yielded StepOutput and unconditionally unfolds
        # its delta dict alongside .output (unfold_ensemble_dim), carrying the
        # hard-boundary comment below.

    def _accumulate_step_loss(self, ..., deltas: TensorMapping) -> torch.Tensor:
        # CHANGED — calls self._loss_obj(..., deltas=deltas) and returns
        # result.total(), so the penalty rides the one per-step
        # accumulate_loss call; records
        # metrics["corrector_regularization_step_{step}"].
```

### Critical detail — accumulation seam and the hard boundary

- Unconditional unfold: `unfold_ensemble_dim` reshapes a contiguous
  `corrected − network_output` into a view sharing storage, so the cost is
  metadata-only, and empty delta dicts pass through unchanged. Nothing needs a
  `needs_corrector_deltas` property.
- The penalty is folded into each optimized step's `result.total()` and
  bypasses the `get_regularizer_loss` accumulation in `train_on_batch` (which
  stays, unchanged, for the module regularizers): the penalty's graph is not
  disjoint from the main loss's, so a second `optimization.accumulate_loss`
  call would backward through it twice under gradient accumulation.
- `train_on_batch` writes `metrics["corrector_regularization"]` from
  `_AccumulatedLoss`, next to its existing `metrics["loss"]` write, so the
  batch aggregate is set where the other aggregate is set.
- Hard boundary: training consumes deltas at the `StepOutput` level inside
  `_accumulate_loss`, never via the `StepDiagnostics` carriage — that carriage
  is detached, output-masked, and inference-only by design. Stated here and
  seam-commented in code.
- `Stepper.step` applies `apply_output_masking` to the diagnostics, so the NaN
  fill sits on a live graph. The regularizer drops NaN points via the target
  trick above; the masked-output test below covers the whole path.
- `fme.coupled` is out of scope: `CoupledStepperTrainLoss` keeps calling
  `build_loss` and using `StepLoss` directly.

---

## Divergences from #1273

1. **Inclusion, not exclusion**: field selection is opt-in `names_and_prefixes`
   per feature (default selects nothing); #1273 used
   `exclude_names_and_prefixes` defaulting to all corrector-modified variables.
   A present config with `names_and_prefixes=None` is an error.
2. **All name validation at build, none at runtime**: the corrector discovers
   its delta keys at construction by a dummy pass on fake data; #1273
   warn-onced unmatched entries at runtime. Consequence: an active corrector
   must produce deltas for every selected name (a partial set raises), so the
   `corrector_regularization` metric is comparable across epochs whenever the
   corrector is active.
3. **No detach control**: #1273 threads a detach flag from the stepper to the
   step seam; deltas are simply always attached, since detaching frees nothing.
4. **No `StepOutput.uncorrected`**: #1273 assumed #1271 had delivered it; it
   does not exist on `main`, and with the pre-corrector outputs built inside
   `CorrectorLoss` on plain dicts the convenience would be dead code.
5. **No keep-gradient guard**: #1273 predates the finding that straight-through
   deltas are detached and therefore unregularizable. Rather than guard the
   combination, this PR ignores `keep_gradient` and a follow-up PR removes it.
6. **`StepOutputLoss` composes rather than subsumes**: #1273's
   `StepOutputLoss`/`StepOutputLossResult` is adopted, but as a thin wrapper
   over an unchanged `StepLoss` plus a `CorrectorLoss`, and the result keeps the
   penalty's own `LossOutput` instead of a scalar.

---

## Tests

## `fme/core/corrector/test_loss.py` (new)

```python
# Deterministic helper: fixed prediction/target/delta dicts, a normalizer with
# known means/stds, MSE step loss — penalties computable by hand.

def test_config_post_init_errors():
    # GOAL: both features None; a present feature with names_and_prefixes None
    # or empty; weight <= 0; EnsembleLoss / NaN / global_mean_type — each
    # raises in __post_init__.

def test_build_errors_on_entry_matching_no_modified_name():
    # GOAL: an entry matching no corrector-modified name raises at build.
    # PARAMETERIZE: entry in {exact name, trailing-underscore prefix}.

def test_build_errors_without_modified_names():
    # GOAL: empty corrector_modified_names raises.

def test_build_errors_on_prescribed_prognostic_entry():
    # GOAL: an entry matching only a prescribed prognostic name raises, and the
    # message says the delta is dropped after the prescribed overwrite.

def test_build_regularizer_packs_matched_names():
    # GOAL: the built WeightedMappingLoss packs exactly
    # selection.matched(loss_visible_names); a prefix entry matches all its
    # level names.

def test_step_output_loss_without_corrector_loss_unchanged():
    # GOAL: with corrector_loss None, total() and get_channel_losses() match a
    # bare StepLoss, deltas ignored.

def test_pre_corrector_outputs_selected_only():
    # GOAL: the main loss sees output − delta for the configured names only;
    # other keys use the network output as-is; targets untouched.

def test_regularization_analytic_penalty():
    # GOAL: penalty equals the hand-computed mean of (delta/std)^2 — normalizer
    # means cancel against the zeros target.

def test_regularization_masked_points_drop():
    # GOAL: NaN-filled delta points contribute nothing; penalty and gradients
    # stay finite.

def test_total_and_channel_decomposition():
    # GOAL: total() == main.total() + weight * penalty.total();
    # get_channel_losses() is main-only; get_corrector_channel_losses() covers
    # exactly the selected names.

def test_missing_selected_delta_raises():
    # GOAL: a non-empty delta dict lacking a selected name raises.

def test_empty_deltas_inert():
    # GOAL: empty deltas ⇒ no pre-corrector swap, no penalty; result matches
    # the unconfigured case.
```

## `fme/core/corrector/test_registry.py` (modified)

```python
def test_construction_records_modified_names():
    # GOAL: a constructed CorrectionSequence's modified_names is exactly the
    # delta keys a real call produces.

def test_discovery_through_epoch_schedule():
    # GOAL: EpochScheduledCorrector.modified_names is independent of the
    # disabled-epoch state.
```

## `fme/core/step/test_step.py` (modified)

```python
def test_corrector_deltas_stay_attached():
    # GOAL: a built step's corrector deltas carry grad_fn while the corrected
    # output is unchanged.

def test_step_exposes_corrector_modified_names():
    # GOAL: corrector_modified_names is populated for a step with a corrector,
    # empty otherwise, and forwarded through MultiCallStep.
```

## `fme/ace/stepper/test_single_module.py` (modified)

```python
# Deterministic corrector: a correction adding a fixed offset to one variable.

def test_train_on_batch_pre_corrector_equivalence():
    # GOAL: with pre-corrector optimization selecting the corrected variable,
    # metrics["loss"] equals the same stepper's loss with no corrector at all;
    # returned predictions stay fully corrected.

def test_gradient_flows_through_correction_when_configured():
    # GOAL: with corrector_loss configured, parameter grads differ from the
    # no-corrector-loss baseline.

def test_corrector_regularization_gradient_accumulation():
    # GOAL: the penalty folded into per-step totals backpropagates under
    # gradient accumulation (no double-backward); one accumulate_loss per
    # optimized step.

def test_masked_output_with_corrector_loss_finite():
    # GOAL: masked-output config + corrector_loss yields finite losses and
    # parameter gradients through the NaN-filling apply_output_masking path.

def test_epoch_scheduled_corrector():
    # GOAL: build-time validation passes via discovery even when epoch 0 is
    # disabled; disabled epochs are inert (no penalty, no metric, no error).

def test_corrector_regularization_metrics():
    # GOAL: corrector_regularization_step_{i} per optimized step plus the
    # per-batch corrector_regularization mean written in train_on_batch.

def test_both_features_together():
    # GOAL: both features configured in one train_on_batch: the main loss sees
    # the pre-corrector outputs, the penalty is added, and both metric families
    # appear (per #1273's acceptance criteria).
```

---

## Open Questions

- **Per-channel reporting of the penalty.** `get_channel_losses()` is main-loss
  only, so per-channel losses do not sum to `total()` under regularization.
  Keeping the penalty as a `LossOutput` makes the fix available:
  `_accumulate_loss` can fold `get_corrector_channel_losses()` into
  `per_channel_losses` under a `corrector_regularization/` key prefix, which
  `PerChannelLossAggregator` logs unchanged as
  `<label>/mean/loss/corrector_regularization/<var>`. That reports every
  selected channel's penalty without mixing two quantities under one variable
  name — at the cost that no single reported number decomposes `total()`. The
  alternative, adding `weight * penalty` into the matching main channels, makes
  the sum work but hides which term moved. Proposed: the prefixed keys.
- **Where discovery runs.** Moving it into `get_corrector` requires the step
  configs to pass their name sets through; the alternative is keeping the
  post-construction `discover_modified_names(...)` call in `get_step`. Proposed:
  construction, since it removes both the mutation and the
  `modified_names is None` state.
- **Splitting field selection into a follow-on PR.** Proposed: keep it here —
  see the PR thread on `fme/core/name_and_prefix_matcher.py`.
