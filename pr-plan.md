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
regularized. That feature is removed in a follow-up PR, and this PR adds
nothing to discover or guard against it.

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

## `fme/core/loss.py` (modified)

`StepLoss`, `StepLossConfig`, `LossOutput`, and `WeightedMappingLoss` are as on
`main`: `LossOutput` stays a per-channel reduction container, and
`StepLoss.forward(predict_dict, target_dict, step, data_mask)` knows only
predictions, targets, and the step decay. The loss objects that consume
corrector deltas join them here.

```python
class CorrectorLoss(torch.nn.Module):  # NEW
    """The corrector-delta half of the training loss."""

    def __init__(
        self,
        precorrector_names: list[str] | None,
        regularizer: WeightedMappingLoss | None,
        penalty_weight: float,
    ): ...

    def pre_corrector_outputs(
        self, predict_dict: TensorMapping, deltas: TensorMapping
    ) -> TensorDict:
        """``predict_dict[k] − deltas[k]`` for the selected names, else
        ``predict_dict[k]``; raises when a selected name is missing."""

    def penalty(
        self, deltas: TensorMapping, data_mask: TensorMapping | None = None
    ) -> LossOutput | None:
        """Penalty over the selected deltas, per channel."""

    @property
    def penalty_weight(self) -> float: ...


class StepOutputLoss(torch.nn.Module):  # NEW
    """`StepLoss` plus the corrector-delta terms of `StepOutput`.

    Deltas come from the `StepOutput`, never the inference-only
    `StepDiagnostics` carriage. The penalty takes no per-step decay, and the
    two corrector features may be enabled together.
    """

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
    corrector_penalty: LossOutput | None = None
    corrector_penalty_weight: float = 1.0

    def total(self) -> torch.Tensor:
        """``main.total() + weight * corrector_penalty.total()``."""

    def get_channel_losses(self) -> dict[str, ChannelLossInfo]:
        """Delegates to ``main``: the penalty is in ``total()`` only."""
```

These live here, not under `fme/core/corrector/`, so the loss objects sit with
the loss objects they compose; `fme/core/loss.py` still imports nothing from
`fme/core/corrector/`, and the dependency runs the other way — the config
module below imports `CorrectorLoss` from here.

### Critical detail — terminology

"Corrector regularization" names the *feature*; the tensors it produces are
*penalties*. So `CorrectorRegularizationConfig` configures the feature, while
`penalty`, `corrector_penalty`, and `penalty_weight` name the terms.

### Critical detail — the penalty enters `total()` and nothing else

`get_channel_losses` stays the main loss per channel. The penalty is not
reported per channel and gets no metric key of its own; it reaches the logs
only inside the totals that already exist. With `C` the active channels
(`LossOutput.total()`'s `mask.sum(dim=0) > 0`) and `S ⊆ C` the selected ones:

```
total()                           = mean_{c∈C} main_c  +  w · mean_{c∈S} penalty_c
mean_{c∈C}(main_c + w · penalty_c) = mean_{c∈C} main_c  +  w · (|S|/|C|) · mean_{c∈S} penalty_c
```

Folding `w · penalty_c` into channel `c` therefore does not decompose
`total()`; the decomposition needs `w · (|C|/|S|) · penalty_c`, whose per-channel
value moves with `|C|` — and `|C|` is a property of the run's mask, not of the
variable. Per-channel penalty reporting is deferred to a follow-up PR, which
settles that choice on its own.

`CorrectorLoss.penalty` forwards `data_mask` to the regularizer so that both
halves of `total()` average over the same samples per channel. Without it the
penalty's channel means have denominator `batch_size` while the main loss's
have the masked count.

## `fme/core/corrector/loss_config.py` (new)

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
    ) -> CorrectorLoss:  # CHANGED from a field bag to a behavior-bearing object
        ...
```

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
        #   normalizer=self._step_obj.get_loss_normalizer(),
        #   gridded_operations=self._dataset_info.gridded_operations,
        #   channel_dim=self.CHANNEL_DIM)


@dataclasses.dataclass
class TrainStepperConfig:
    corrector_loss: CorrectorLossConfig | None = None  # NEW — default preserves
                                                       # current behavior


class TrainStepper(TrainStepperABC[...]):
    def __init__(self, stepper: Stepper, config: TrainStepperConfig):
        # CHANGED — self._loss_obj = StepOutputLoss(
        #     stepper.build_loss(config.loss),
        #     stepper.build_corrector_loss(config.corrector_loss))

    def _accumulate_loss(self, ...):
        # CHANGED — keeps the yielded StepOutput and unconditionally unfolds
        # its delta dict alongside .output (unfold_ensemble_dim), handing it to
        # the per-step seam. Signature and return type otherwise unchanged.

    def _accumulate_step_loss(
        self,
        ...,  # unchanged: gen_step, target_step, step, data_mask, optimize,
              # metrics, weighted_sums, total_counts — still out-params
        deltas: TensorMapping,  # NEW — the only change to this signature
    ) -> torch.Tensor:
        # CHANGED — self._loss_obj(..., deltas=deltas). The metric writes and
        # the per-channel accumulation stay exactly where `main` has them.
```

### Critical detail — the per-step seam keeps `main`'s shape

`_accumulate_step_loss` gains one parameter and nothing else. The out-params it
mutates on `main` — `metrics`, `weighted_sums`, `total_counts` — stay
out-params, `_finalize_per_channel_losses` stays where it is, and
`_accumulate_loss` keeps returning its tuple. Retiring that mutation-based
shape is a real cleanup but an independent one, and it belongs in a follow-up
PR that also covers `CoupledTrainStepper._accumulate_step_loss`
(`fme/coupled/stepper.py`), which has the same shape and which this PR does not
touch.

This PR adds no metric key: `loss_step_{i}` and the per-channel entries keep
their existing meanings, with the penalty riding `loss_step_{i}` through the
per-step total.

### Critical detail — comments on the private surface

Private classes and methods carry no docstrings; a short inline comment on a
field or at a seam is the documentation, and a comment that restates the name
above it is deleted. So the `_accumulate_*` methods are commented as shown
above and nothing more, while the reasoning a reader needs — the `StepDiagnostics` hard boundary, the no-decay penalty, the
two features combining — lives in the public `StepOutputLoss` docstring and in
this plan. Inline comments stay at ~1 line, docstring components at ≤2.

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
- No new metric key, and no aggregator change. The penalty is inside every
  optimized step's `total()`, so `<label>/mean/loss` and
  `<label>/mean/loss_step_{i}` include it, while `<label>/mean/loss/<var>` stays
  the main loss for that channel alone. Reporting the penalty on its own is a
  follow-up PR.
- Hard boundary: training consumes deltas at the `StepOutput` level inside
  `_accumulate_loss`, never via the `StepDiagnostics` carriage — that carriage
  is detached, output-masked, and inference-only by design. The reasoning lives
  in the `StepOutputLoss` docstring; the seam itself gets one line.
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
   penalty metric is comparable across epochs whenever the corrector is active.
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
7. **No penalty metric of its own**: #1273 logs
   `corrector_regularization` and `corrector_regularization_step_{i}`. This PR
   adds no metric key and touches no aggregator — the penalty is visible in the
   `loss` and `loss_step_{i}` totals it is part of. Separate reporting, and the
   per-channel decomposition question above, are a follow-up PR's.

---

## Tests

Each test file follows the module it covers: config and build validation with
the configs, loss behavior with the loss objects.

## `fme/core/corrector/test_loss_config.py` (new)

```python
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
```

## `fme/core/test_loss.py` (modified)

```python
# Deterministic helper: fixed prediction/target/delta dicts, a normalizer with
# known means/stds, MSE step loss — penalties computable by hand.

def test_step_output_loss_without_corrector_loss_unchanged():
    # GOAL: with corrector_loss None, total() and get_channel_losses() match a
    # bare StepLoss, deltas ignored.

def test_pre_corrector_outputs_selected_only():
    # GOAL: the main loss sees output − delta for the configured names only;
    # other keys use the network output as-is; targets untouched.

def test_penalty_analytic_value():
    # GOAL: penalty equals the hand-computed mean of (delta/std)^2 — normalizer
    # means cancel against the zeros target.

def test_penalty_masked_points_drop():
    # GOAL: NaN-filled delta points contribute nothing; penalty and gradients
    # stay finite.

def test_total_and_channel_decomposition():
    # GOAL: total() == main.total() + weight * penalty.total(), and
    # get_channel_losses() equals the bare StepLoss channels — no penalty
    # anywhere per channel.

def test_penalty_uses_the_data_mask():
    # GOAL: with a data_mask hiding one sample of a selected variable, that
    # sample contributes to neither half of total().

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

def test_corrector_penalty_gradient_accumulation():
    # GOAL: the penalty folded into per-step totals backpropagates under
    # gradient accumulation (no double-backward); one accumulate_loss per
    # optimized step.

def test_masked_output_with_corrector_loss_finite():
    # GOAL: masked-output config + corrector_loss yields finite losses and
    # parameter gradients through the NaN-filling apply_output_masking path.

def test_epoch_scheduled_corrector():
    # GOAL: build-time validation passes via discovery even when epoch 0 is
    # disabled; disabled epochs are inert (no penalty, no metric, no error).

def test_penalty_rides_the_existing_metrics():
    # GOAL: metrics["loss"] and metrics["loss_step_{i}"] each exceed the
    # penalty-free baseline by the weighted penalty, the per-channel entries
    # match that baseline, and no key mentions the penalty.

def test_both_features_together():
    # GOAL: both features configured in one train_on_batch: the main loss sees
    # the pre-corrector outputs and the penalty is in the totals (per #1273's
    # acceptance criteria).
```
