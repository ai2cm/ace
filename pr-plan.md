# Carry corrector step diagnostics through coupled inference

Threads each component stepper's per-step correction `delta` through `CoupledStepper` onto
that realm's prediction data, and surfaces the existing step-diagnostics writer and
correction-metrics aggregator per realm in the coupled inference configs. No new
coupled-specific diagnostics container: the `fme.ace` components are reused once per realm.

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
        # corrector_diagnostics instead of defaulting it to empty. Attaches
        # nothing: the returned batch still passes through time-changing ops.
        ...

    def _predict(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledBatchData, CoupledPrognosticState]:
        # CHANGED — also returns the end state (was: prediction only), so the
        # stacked per-realm diagnostics can be attached after get_end.
        # Rejected: attaching in predict() and predict_paired() separately, which
        # duplicates the attach and its ordering constraint at two call sites.
        ...

    def predict(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledBatchData, CoupledPrognosticState]:
        # CHANGED — returns _predict's tuple; the duplicated get_end block moves
        # into _predict
        ...

    def predict_paired(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledPairedData, CoupledPrognosticState]:
        # CHANGED — consumes _predict's tuple
        ...
```

### Critical detail — where the diagnostics attach, and why

- Each realm's stacked series comes from `StepOutput.stack_diagnostics` over that realm's
  own `ComponentStepPrediction`s, so it is `None` when that realm's stepper has no
  corrector or its corrector modified nothing — matching single-component inference.
- `BatchData` rejects a diagnostics-bearing batch in every time-changing method
  (`prepend`, `compute_derived_variables`, `remove_initial_condition`, `get_end`, …).
  Attachment must therefore be the last thing `_predict` does: after the
  prepend / compute-derived / remove-initial-condition dance **and** after the end states
  are taken. This mirrors the attach position in single-component `Stepper.predict`.
- Per-realm time axes differ (several atmosphere steps per ocean step). Each realm's series
  is stacked from that realm's own steps and is aligned with that realm's prediction time
  axis by construction, so no cross-realm time handling is required.

## `fme/coupled/data_loading/batch_data.py` (modified)

```python
@dataclasses.dataclass
class CoupledPairedData:
    @classmethod
    def from_coupled_batch_data(
        cls,
        prediction: CoupledBatchData,
        reference: CoupledBatchData,
    ) -> "CoupledPairedData":
        # CHANGED — forward each realm's prediction step_diagnostics onto that
        # realm's PairedData, so it reaches the per-realm writers and aggregators
        ...
```

## `fme/coupled/aggregator.py` (modified)

```python
@dataclasses.dataclass
class InferenceEvaluatorAggregatorConfig:
    # NEW — one field applied to both realms, matching how this config already
    # applies log_histograms / log_video / log_zonal_mean_images to both.
    # Rejected: ocean_/atmosphere_-prefixed fields, which double the surface for a
    # granularity choice that is not realm-specific; per-realm *availability*
    # still resolves separately, from each realm's own normalizer.
    step_diagnostics: StepDiagnosticsMetricConfig = dataclasses.field(
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
        # normalizer and output subdirectory
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
        # realm's normalizer through. Optional-with-None default mirrors the ace
        # InferenceAggregatorConfig.build signature: with the default config and
        # no normalizer the correction metrics are silently skipped; an explicit
        # non-default configuration without a normalizer raises.
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

## `fme/coupled/inference/data_writer.py` (unchanged — verified, not modified)

`CoupledDataWriterConfig` already holds one `DataWriterConfig` per realm and builds into the
`ocean/` and `atmosphere/` subdirectories, and `CoupledPairedDataWriter.append_batch`
forwards each realm's `PairedData` to that realm's `PairedDataWriter`, which writes
`step_diagnostics/correction_deltas.nc` when the batch carries diagnostics. Once the
carriage above lands, `save_step_diagnostics: true` on a realm's writer config works
end-to-end with no new writer surface; the end-to-end test below pins that.

The coupled evaluator's inline-validation path needs no change either: coupled training
already builds the evaluator aggregator with both realms' normalizers, so the new
`step_diagnostics` field reaches it through the same config.

---

## Tests

## `fme/coupled/test_stepper.py` (modified)

```python
# Build on get_stepper_and_batch and the existing test_predict_paired setup;
# inject CorrectionSequence([ConstantOffsetCorrection(...)]) onto a component
# stepper's step object, as the single-component stepper tests do.

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

def test_predict_paired_prediction_values_unchanged_by_carriage():
    # GOAL: regression guard — with a corrector installed, prediction values and
    # the returned prognostic state are identical whether or not the diagnostics
    # are carried, including with compute_derived_variables=True (the prepend /
    # compute-derived / remove-initial-condition path).
    ...

def test_predict_attaches_step_diagnostics():
    # GOAL: the non-paired predict path attaches the same per-realm series and
    # still returns a usable end state (the attach happens after get_end).
    ...

def test_predict_paired_without_corrector_has_no_step_diagnostics():
    # GOAL: both realms' step_diagnostics are None when no component has a
    # corrector.
    ...

def test_prediction_generator_yields_corrector_diagnostics():
    # GOAL: each ComponentStepPrediction carries its component step's
    # corrector_diagnostics, for both the inner atmosphere and outer ocean yields.
    ...
```

## `fme/coupled/data_loading/test_batch_data.py` (new)

```python
def test_from_coupled_batch_data_forwards_step_diagnostics():
    # GOAL: each realm's prediction step_diagnostics lands on that realm's
    # PairedData, and None passes through as None.
    # PARAMETERIZE: (ocean, atmosphere) diagnostics presence in
    # {(set, None), (None, set), (set, set)}.
    ...
```

## `fme/coupled/test_aggregator.py` (modified)

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

## `fme/coupled/inference/test_inference.py` (modified)

```python
# Extend the existing end-to-end coupled inference test setup.

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

## `fme/coupled/inference/test_evaluator.py` (modified)

```python
def test_evaluator_logs_correction_metrics_per_realm(tmp_path):
    # GOAL: an evaluator run with a corrector-equipped component logs that realm's
    # correction scalars, and a run with no corrector logs none — the defaults-
    # preserved check for existing coupled evaluator runs.
    ...
```

---

## Open Questions

- The correction-metrics granularity is one `step_diagnostics` field applied to both realms.
  Worth per-realm fields instead if enabling `correction_maps` for only the ocean is a real
  use case?
- `InferenceAggregatorConfig.build` takes the two normalizers as optional keyword arguments
  to match the single-component signature and keep existing call sites working. Making them
  required would force every caller to opt in explicitly — preferable?
