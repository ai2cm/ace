# Carry corrector step diagnostics through coupled inference

## Why

Single-component inference can already answer "what is the corrector actually doing?" — it
writes the per-step correction `delta` to `step_diagnostics/correction_deltas.nc` and logs
normalized correction magnitudes. Coupled inference cannot: `CoupledStepper` drops each
component stepper's `StepOutput.corrector_diagnostics` at the component-step seam, so a
coupled run with a corrector on either realm is silent.

That matters when diagnosing a coupled run that drifts: the corrector is one of the few
places where the emulator's output is overwritten by something other than the network, and
today there is no way to tell whether it is nudging gently or holding the state up. This PR
makes the coupled path report what the single-component path already reports, per realm.

## What changes

Thread each component stepper's per-step correction `delta` through `CoupledStepper` onto
that realm's prediction data, then surface the existing `fme.ace` step-diagnostics writer and
correction-metrics aggregator per realm in the coupled inference configs.

## Design choices

- **Reuse the `fme.ace` components once per realm** rather than adding a coupled-specific
  diagnostics container. `StepDiagnostics`, `StepDiagnosticsWriter`, and
  `StepDiagnosticsAggregator` are realm-agnostic; a coupled run is two realms each with
  their own normalizer, time axis, and output subdirectory, which is exactly what building
  the ace components twice gives. The coupled layer adds carriage and config plumbing only.
- **One `step_diagnostics` config field applied to both realms**, not
  `ocean_`/`atmosphere_`-prefixed pairs. This matches how `InferenceEvaluatorAggregatorConfig`
  already applies `log_histograms`, `log_video`, and `log_zonal_mean_images` to both realms.
  Per-realm granularity is deliberately deferred: the only case that would want it is
  enabling `correction_maps` for one realm only, and `correction_maps` is not usable
  per-realm as things stand — `CorrectionDeltaTimeMeanAggregator.get_dataset` in
  `fme/ace/aggregator/inference/step_diagnostics.py` hard-codes `dims = ("lat", "lon")`, so a
  realm whose horizontal dims differ would break. Splitting the field later is a
  config-breaking change; accepted, because the split has no use today.
- **The two normalizer arguments mirror the `fme.ace` signature they wrap.**
  `fme/ace/aggregator/inference/main.py`'s `InferenceAggregatorConfig.build` takes
  `normalize: NormalizeFn | None = None`, so the coupled version takes one optional
  normalizer per realm rather than making them required. The availability rule that makes
  optional safe already lives in `StepDiagnosticsMetricConfig.build`, not in the caller.

---

## `fme/coupled/stepper.py` (modified)

```python
class ComponentStepPrediction:
    def __init__(
        self,
        realm: Literal["ocean", "atmosphere"],
        data: TensorDict,
        step: int,
        stepper_state: StepperState | None,
        corrector_diagnostics: CorrectorDiagnostics,  # NEW
    ):
        ...

    # Required, not defaulted: this class is the seam where the component
    # StepOutput's diagnostics are currently dropped, so a new yield site must be
    # forced to say what it carries rather than silently defaulting to empty.

    @property
    def corrector_diagnostics(self) -> CorrectorDiagnostics:  # NEW
        ...
```

```python
class CoupledStepper:
    def get_prediction_generator(
        self,
        initial_condition: CoupledPrognosticState,
        forcing_data: CoupledBatchData,
        optimizer: OptimizationABC,
    ) -> Generator[ComponentStepPrediction, None, None]:
        # CHANGED — both yield sites (the inner atmosphere loop and the outer
        # ocean step) pass the component StepOutput's corrector_diagnostics
        ...

    def _process_prediction_generator_list(
        self,
        output_list: list[ComponentStepPrediction],
        forcing_data: CoupledBatchData,
    ) -> CoupledBatchData:
        # CHANGED — the per-realm StepOutput reconstruction carries
        # corrector_diagnostics instead of defaulting it to empty, so the stacked
        # series can be built from it later.
        #
        # Still attaches nothing to the returned CoupledBatchData: that batch is
        # about to pass through prepend / compute_derived_variables /
        # remove_initial_condition, all of which raise on a diagnostics-bearing
        # batch. See "Where the diagnostics attach" below.
        ...

    def _predict(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledBatchData, CoupledPrognosticState]:
        # CHANGED — also returns the prognostic state (was: prediction only), so
        # the get_end block that predict and predict_paired each duplicated moves
        # in here, and the stacked per-realm diagnostics can be attached after it
        # via CoupledBatchData.with_step_diagnostics.
        #
        # Rejected: attaching in predict() and predict_paired() separately, which
        # duplicates the attach and its ordering constraint at two call sites.
        ...

    def predict(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledBatchData, CoupledPrognosticState]:
        # CHANGED — returns _predict's tuple directly; its get_end block and its
        # redundant CoupledBatchData rebuild both go away
        ...

    def predict_paired(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledPairedData, CoupledPrognosticState]:
        # CHANGED — consumes _predict's tuple; its get_end block goes away
        ...
```

### Where the diagnostics attach, and why

- Each realm's stacked series comes from `StepOutput.stack_diagnostics` over that realm's own
  `ComponentStepPrediction`s, so it is `None` exactly when that realm's stepper has no
  corrector — matching single-component inference. See "What counts as silence" below for
  the corrector-that-changed-nothing case, which is *not* silent.
- `BatchData` rejects a diagnostics-bearing batch in every time-changing method
  (`prepend`, `compute_derived_variables`, `remove_initial_condition`, `get_end`,
  `select_time_slice`, `subset_names`, `get_start`, `scatter_spatial`) via
  `BatchData._raise_if_step_diagnostics`. Attachment must therefore be the last thing
  `_predict` does: after the prepend / compute-derived / remove-initial-condition dance
  **and** after the prognostic state is taken. This mirrors the attach position in
  `fme/ace/stepper/single_module.py`, where `Stepper.predict` rebuilds its `BatchData` with
  `step_diagnostics` only after `get_end`.
- Per-realm time axes differ (several atmosphere steps per ocean step). Each realm's series
  is stacked from that realm's own steps and is aligned with that realm's prediction time
  axis by construction, so no cross-realm time handling is required.

### What counts as silence

`build_corrector_diagnostics` in `fme/core/corrector/output.py` keys `delta` by the names the
corrector *declares* it writes, not by names whose values actually changed. Consequences the
tests below pin:

- A realm with **no corrector** produces `step_diagnostics is None` — no file, no metrics.
- A realm with a corrector that declares names but leaves them **numerically unchanged**
  produces a non-`None` series of exact zeros: the file *is* written and the scalars *are*
  logged as `0.0`. This is the ace behavior; the coupled path inherits it rather than adding
  a coupled-only "drop all-zero deltas" rule.
- Names in a component's `prescribed_prognostic_names` are dropped from `delta` by
  `fme/core/step/single_module.py`, because a prescribed overwrite replaces the corrected
  value and the delta would no longer be exact. Coupled inference supports per-component
  `ocean_prescribed_prognostic_names` / `atmosphere_prescribed_prognostic_names`, so the
  written variable set is override-dependent. Not new behavior, but worth knowing when
  reading a coupled `correction_deltas.nc`.

### Constraint on a corrector for the atmosphere realm

`fme/core/step/single_module.py` raises

```
post-corrector adjustment names overlap the corrector's modified names: [...]
```

when the `OceanConfig`-prescribed `surface_temperature_name` appears in the corrector's
delta, because the SST prescription runs *after* the corrector and would break
`delta = output - input_snapshot`. A coupled atmosphere stepper is required to have an
`OceanConfig` (`CoupledStepperConfig._validate_component_configs` raises without one), so an
atmosphere corrector in a coupled run can never touch the prescribed surface-temperature
variable. This is pre-existing and unchanged; it constrains the test setup below (the
atmosphere-side corrector must target some other variable) and is not otherwise worked
around.

## `fme/coupled/data_loading/batch_data.py` (modified)

```python
@dataclasses.dataclass
class CoupledBatchData:
    def with_step_diagnostics(  # NEW
        self,
        ocean: StepDiagnostics | None,
        atmosphere: StepDiagnostics | None,
    ) -> "CoupledBatchData":
        # Returns a copy with each realm's BatchData carrying that realm's series,
        # via dataclasses.replace on each BatchData (so BatchData.__post_init__
        # re-validates the diagnostics' sample dim against the batch).
        #
        # Why a method here rather than open-coding it in _predict, as ace does in
        # Stepper.predict: coupled would need the rebuild twice in one expression,
        # and this keeps the ordering constraint documented next to the field it
        # constrains. It is the only new API surface in this PR.
        ...


@dataclasses.dataclass
class CoupledPairedData:
    @classmethod
    def from_coupled_batch_data(
        cls,
        prediction: CoupledBatchData,
        reference: CoupledBatchData,
    ) -> "CoupledPairedData":
        # CHANGED — delegate each realm to PairedData.from_batch_data instead of
        # hand-copying fields. That helper already does the time-equality check and
        # forwards step_diagnostics, so the carriage comes for free.
        #
        # Drive-by fix this delegation includes: the hand-rolled version dropped
        # prediction.n_ensemble (silently defaulting to 1) where the ace helper
        # forwards it. Called out because it is a behavior change beyond diagnostics
        # carriage — see the test for it below.
        ...
```

## `fme/coupled/aggregator.py` (modified)

Both new fields need a `Parameters:` entry in their config's class docstring — these two
docstrings document every field today, and the generated config documentation is built from
them.

```python
@dataclasses.dataclass
class InferenceEvaluatorAggregatorConfig:
    step_diagnostics: StepDiagnosticsMetricConfig = dataclasses.field(  # NEW
        default_factory=StepDiagnosticsMetricConfig
    )

    def build(
        self,
        dataset_info: CoupledDatasetInfo,
        n_timesteps_ocean: int,
        n_timesteps_atmosphere: int,
        initial_time: xr.DataArray,
        ocean_normalize: NormalizeFn,
        atmosphere_normalize: NormalizeFn,
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        ocean_channel_mean_names: Sequence[str] | None = None,
        atmosphere_channel_mean_names: Sequence[str] | None = None,
    ) -> "InferenceEvaluatorAggregator":
        # CHANGED — pass step_diagnostics into both per-realm
        # build_inference_evaluator_aggregator calls, each with that realm's
        # normalizer and output subdirectory. Both normalizers are already
        # required here, so correction-metric availability cannot differ between
        # realms on this path.
        ...


@dataclasses.dataclass
class InferenceAggregatorConfig:
    step_diagnostics: StepDiagnosticsMetricConfig = dataclasses.field(  # NEW
        default_factory=StepDiagnosticsMetricConfig
    )

    def build(
        self,
        dataset_info: CoupledDatasetInfo,
        n_timesteps_ocean: int,
        n_timesteps_atmosphere: int,
        output_dir: str,
        ocean_normalize: NormalizeFn | None = None,  # NEW
        atmosphere_normalize: NormalizeFn | None = None,  # NEW
    ) -> "InferenceAggregator":
        # CHANGED — set step_diagnostics on each realm's ace config and pass that
        # realm's normalizer through.
        #
        # Optional-with-None mirrors fme/ace/aggregator/inference/main.py's
        # InferenceAggregatorConfig.build(normalize=None), which is the config this
        # one wraps twice. The availability rule lives in
        # StepDiagnosticsMetricConfig.build and needs no help here: default config
        # with no normalizer is silently skipped, and an explicit non-default
        # configuration with no normalizer raises. This is the only path where a
        # normalizer can be absent, so it is the only path where the two realms can
        # resolve availability differently.
        ...
```

## `fme/coupled/inference/inference.py` (modified)

```python
def run_inference_from_config(config: InferenceConfig):
    aggregator = aggregator_config.build(
        ...,
        ocean_normalize=stepper.ocean.normalizer.normalize,  # NEW
        atmosphere_normalize=stepper.atmosphere.normalizer.normalize,  # NEW
    )  # CHANGED
```

This is the only production call site of the coupled `InferenceAggregatorConfig.build`, and
this PR updates it, so the optional defaults above exist for ace-signature parity, not to
avoid touching callers.

---

## Defaults, and what existing runs start doing

`StepDiagnosticsMetricConfig.correction_scalars` defaults to `True` (`correction_maps`
defaults to `False`), and coupled training already supplies both realms' normalizers to
`InferenceEvaluatorAggregatorConfig.build` in `fme/coupled/train/train_config.py`. So:

| Run shape | What changes |
| --- | --- |
| No corrector on either realm | Nothing. No new files, no new log keys, unchanged rollout values. |
| Corrector on a realm | That realm gains `time_mean_norm/correction_magnitude/*` and, where per-step time series are enabled, `mean_norm/weighted_correction_*` log keys. Rollout values unchanged. |
| Corrector on a realm, `save_step_diagnostics: true` on that realm's writer config | Additionally writes `<realm>/step_diagnostics/correction_deltas.nc`. Off by default. |

This applies to coupled inference, the coupled evaluator, and coupled inline validation
during training, since all three build through the same two configs. No writer config
surface is added. The corrector-free row is pinned by
`test_evaluator_logs_no_correction_metrics_without_corrector` below.

## Not changed (verified, no edits needed)

- **`fme/coupled/inference/data_writer.py`.** `CoupledDataWriterConfig` already holds one
  `DataWriterConfig` per realm and builds into the `ocean/` and `atmosphere/`
  subdirectories, and `CoupledPairedDataWriter.append_batch` forwards each realm's
  `PairedData` to that realm's `PairedDataWriter`, which writes
  `step_diagnostics/correction_deltas.nc` when the batch carries diagnostics. Once the
  carriage lands, `save_step_diagnostics: true` works end-to-end with no new writer surface;
  the end-to-end test pins that.
- **The coupled training path.** `CoupledTrainStepper._accumulate_step_loss` rebuilds each
  `ComponentStepPrediction` into `ComponentEnsembleStepPrediction`, which has no diagnostics
  field, so `train_on_batch` still drops them. Deliberate, and consistent with
  single-component training, where `TrainOutput` carries no diagnostics either. Making
  `corrector_diagnostics` a required constructor argument means a future ensemble-side
  carriage has to make that choice explicitly rather than inherit an empty default.
- **`InferenceAggregator.log_time_series`** in `fme/coupled/aggregator.py` consults only the
  ocean aggregator. Wiring normalizers in flips it to unconditionally `True`. Nothing in
  `fme/` reads the property, so this is inert — noted because it is a behavior change the
  diff does not otherwise show.

---

## Tests

### `fme/coupled/test_stepper.py` (modified)

Build on `get_stepper_and_batch` (`fme/coupled/test_stepper.py:1289`) and the existing
`test_predict_paired` (`fme/coupled/test_stepper.py:1739`) setup; inject
`CorrectionSequence([ConstantOffsetCorrection(...)])` onto a component stepper's step object,
as `fme/ace/stepper/test_single_module.py` does. The injected corrector must not target the
atmosphere's prescribed surface-temperature variable (see the constraint above).

```python
def test_predict_paired_attaches_step_diagnostics_for_corrected_realm():
    # GOAL: with a constant-offset corrector on exactly one realm, that realm's
    # PairedData carries step_diagnostics whose delta equals the offset at every
    # forward step and is shaped like that realm's prediction series; the other
    # realm's step_diagnostics is None.
    # PARAMETERIZE: corrected realm in {ocean, atmosphere} — this also covers the
    # differing per-realm step counts (outer ocean steps vs inner atmosphere steps).
    ...

def test_predict_paired_step_diagnostics_both_realms():
    # GOAL: correctors on both components yield independent per-realm delta series,
    # each with that realm's own step count and offset value.
    ...

def test_predict_paired_regression_against_baseline():
    # GOAL: regression guard on the values, since carriage is unconditional and
    # there is no runtime toggle to A/B against. Pin prediction values and the
    # returned prognostic state with validate_tensor_dict
    # (fme/core/testing/regression.py) against a committed .pt baseline, with a
    # corrector installed and compute_derived_variables=True, so the prepend /
    # compute-derived / remove-initial-condition path is exercised.
    # The baseline is generated on main, before the carriage lands.
    ...

def test_predict_attaches_step_diagnostics():
    # GOAL: the non-paired predict path attaches the same per-realm series and
    # still returns a usable prognostic state (the attach happens after get_end).
    # NOTE: CoupledStepper.predict has no production call sites today — only
    # predict_paired is used, by coupled inference and the coupled evaluator. This
    # is coverage for an otherwise test-only path, kept because _predict now serves
    # both and the attach ordering is easy to break.
    ...

def test_predict_paired_without_corrector_has_no_step_diagnostics():
    # GOAL: both realms' step_diagnostics are None when no component has a
    # corrector.
    ...

def test_predict_paired_step_diagnostics_zero_for_unchanged_variable():
    # GOAL: a corrector that declares a name but leaves it unchanged still produces
    # a non-None series of exact zeros — pins the declared-names semantics that the
    # writer and aggregator tests below depend on.
    ...

def test_prediction_generator_yields_corrector_diagnostics():
    # GOAL: each ComponentStepPrediction carries its component step's
    # corrector_diagnostics, for both the inner atmosphere and outer ocean yields.
    ...
```

### `fme/coupled/data_loading/test_batch_data.py` (new)

```python
def test_from_coupled_batch_data_forwards_step_diagnostics():
    # GOAL: each realm's prediction step_diagnostics lands on that realm's
    # PairedData, and None passes through as None.
    # PARAMETERIZE: (ocean, atmosphere) diagnostics presence in
    # {(set, None), (None, set), (set, set)}.
    ...

def test_from_coupled_batch_data_forwards_n_ensemble():
    # GOAL: pins the drive-by fix — n_ensemble reaches each realm's PairedData
    # instead of defaulting to 1.
    ...

def test_from_coupled_batch_data_rejects_mismatched_time():
    # GOAL: the time-equality check survives delegating to
    # PairedData.from_batch_data, for each realm independently.
    ...

def test_with_step_diagnostics_attaches_per_realm():
    # GOAL: attaches each realm's series independently, passes None through, and
    # leaves the input CoupledBatchData unmutated.
    ...
```

### `fme/coupled/test_aggregator.py` (modified)

```python
def test_inference_evaluator_aggregator_logs_correction_metrics_per_realm():
    # GOAL: recording a CoupledPairedData carrying a known constant delta on both
    # realms produces the correction scalars for each realm under that realm's
    # label prefix, with the exact expected normalized value (delta / std of that
    # realm's normalizer).
    ...

def test_inference_evaluator_aggregator_silent_without_step_diagnostics():
    # GOAL: no correction keys are logged for a realm whose step_diagnostics is
    # None; the other realm's are unaffected.
    ...

def test_inference_aggregator_correction_metrics_require_normalizer():
    # GOAL: the no-target coupled config builds silently (no correction metrics)
    # with default step_diagnostics and no normalizer, and raises when
    # step_diagnostics is explicitly non-default and no normalizer is supplied.
    # PARAMETERIZE: normalizer supplied for neither / only one realm.
    ...

def test_inference_aggregator_builds_correction_metrics_with_normalizers():
    # GOAL: with both normalizers supplied, each realm's aggregator records
    # correction metrics into its own output subdirectory.
    ...
```

### Test-setup work the end-to-end tests need first

The coupled inference and evaluator tests build their checkpoint through
`save_coupled_stepper` (`fme/coupled/inference/test_evaluator.py:106`), which delegates to
`get_stepper_config` (`fme/coupled/test_stepper.py:1197`). Neither exposes a corrector today,
so both need a new optional per-component corrector-config argument threaded through — this
is real setup work, not an "extend the existing test" one-liner. `test_inference.py` imports
`save_coupled_stepper` from `test_evaluator.py`, so one threading change serves both files.

The ocean-side corrector should be `OceanCorrectorConfig(force_positive_names=[...])`: it is
the only ocean corrector option that works on the synthetic single-level test data (the
sea-ice, surface-energy-flux, and ocean-heat-content corrections all need extra fields).
Because a `force_positive` clamp only bites where the value is already negative, the test
data must be seeded to make the clamp fire, or the assertion has to accept the all-zero
series from the semantics above — pick one explicitly when writing the test.

### `fme/coupled/inference/test_inference.py` (modified)

```python
def test_inference_writes_step_diagnostics_per_realm(tmp_path):
    # GOAL: a run whose ocean stepper has a config-declared corrector and
    # save_step_diagnostics=True on the ocean writer writes
    # ocean/step_diagnostics/correction_deltas.nc with the corrected variables on
    # the ocean time axis, and writes no such file under atmosphere/.
    ...

def test_inference_no_step_diagnostics_by_default(tmp_path):
    # GOAL: default configuration writes no step_diagnostics directory under
    # either realm.
    ...
```

### `fme/coupled/inference/test_evaluator.py` (modified)

```python
def test_evaluator_logs_correction_metrics_per_realm(tmp_path):
    # GOAL: an evaluator run with a corrector-equipped component logs that realm's
    # correction scalars under that realm's prefix.
    ...

def test_evaluator_logs_no_correction_metrics_without_corrector(tmp_path):
    # GOAL: the defaults-preserved check — a corrector-free evaluator run's logged
    # key set contains no correction keys at all.
    ...
```

---

## Open questions

- **Should the writer flag be surfaced any differently?** This PR adds no writer config
  surface, relying on the existing per-realm `DataWriterConfig.save_step_diagnostics`. That
  means turning the files on for both realms is two YAML edits in two places. Acceptable, or
  worth a coupled-level convenience flag? (Not the same granularity question as the
  aggregator config above — this one asks for *less* per-realm surface, not more.)
