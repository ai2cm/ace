# Carry corrector deltas onto prediction data and add a step-diagnostics netCDF writer

Carries the per-step corrector `delta` series out of `Stepper.predict` in a new
opaque `StepDiagnostics` container on `BatchData`/`PairedData`, NaN-masks the
delta consistently with the prediction at the stepper step seam, and adds an
off-by-default `autoregressive_step_diagnostics.nc` writer as the carriage's
first consumer. No aggregator, no normalizer wiring, no metric flags.

Coupled runs are explicit follow-up work, not silently unsupported: the coupled
generator's `ComponentStepPrediction` drops diagnostics at the component-step
seam, so carrying deltas through `CoupledStepper` requires threading
`ComponentStepPrediction`, per-realm stacking, and `CoupledBatchData` — a
separate PR.

---

## `fme/core/spatial_masking.py` (modified)

```python
SpatialMasking = StaticSpatialMasking | NullSpatialMasking  # NEW — the output-masking type
```

The masking-specific type that everything downstream is annotated with, so a
non-masking callable (e.g. a derived-variables function) can no longer flow in
silently — assigning one is a mypy error, not a latent bug.
`build_output_spatial_masker` (on `SpatialMaskProviderABC`, its
implementations, and the `HasGetSpatialMask` protocol) narrows its return
annotation from the bare `Callable[[TensorMapping], TensorDict]` to
`SpatialMasking`. This rename (together with the two renames below) lands as a
standalone, behavior-preserving first commit, separable from the carriage work.

## `fme/core/coordinates.py` (modified)

`NullPostProcessFn` and `PostProcessFnType` are **deleted**. Their only
consumer is the stepper, and `NullSpatialMasking` already provides the same
no-op with the masking-specific type; the generic "post-process" vocabulary
goes away entirely so nothing suggests an arbitrary func is acceptable here.

## `fme/core/corrector/output.py` (modified)

```python
@dataclasses.dataclass
class CorrectorDiagnostics:
    delta: TensorMapping = dataclasses.field(default_factory=dict)

    def apply_output_masking(  # NEW — map the stepper's output spatial masker over delta
        self, masking: SpatialMasking
    ) -> "CorrectorDiagnostics":
        # Returns a new CorrectorDiagnostics; input unmutated. Named and typed
        # for masking specifically — NOT an arbitrary output-process func:
        # masking a difference is correct ONLY because the output masker is
        # NaN-fill (StaticSpatialMasking with fill_value=NaN, or the
        # NullSpatialMasking no-op): NaN off-mask matches
        # masked_output − masked_snapshot. A finite fill, or any func that
        # derives/transforms values, would break the delta = output − snapshot
        # invariant; this is documented inline.
        ...
```

## `fme/core/step/output.py` (modified)

```python
@dataclasses.dataclass
class StepOutput:
    @classmethod  # NEW — encapsulate how per-step diagnostics stack into a series
    def stack_diagnostics(
        cls, outputs: Sequence["StepOutput"]
    ) -> StepDiagnostics | None:
        # Stacks each output's corrector_diagnostics.delta along a new time dim
        # (dim 1), forward-step-aligned, and returns the stacked container
        # itself; returns None when no output carries a delta. The corrector
        # internals (that the diagnostics are a per-variable delta mapping)
        # stay hidden inside StepOutput/StepDiagnostics — Stepper.predict just
        # attaches the returned object, and the data writer reads it through
        # StepDiagnostics.to_dataset().
        ...
```

## `fme/core/step/step_diagnostics.py` (new)

Opaque per-sample diagnostics container, modeled on `StepperState`. It lives in
`fme/core/step/` (not `fme/ace/data_loading/`) because it is the return type of
`StepOutput.stack_diagnostics` and `fme/core` cannot import from `fme.ace`;
`batch_data.py` imports it in the allowed direction. Holds the stacked per-step
correction series, aligned with the prediction `data` along
`(sample, time, ...)`. An **empty `delta` is valid** — all ops are safe no-ops
on it and nothing asserts non-emptiness — but `stack_diagnostics` returns
`None` when nothing was modified, so `step_diagnostics is None` remains the
single "no correction file" gate for now. It deliberately has **no** time-slice
or prepend-time op — the series is attached only after time-windowing is
finished, and the time-touching `BatchData` methods guard against carrying it
(below).

```python
@dataclasses.dataclass
class StepDiagnostics:
    delta: TensorMapping  # stacked per-step CorrectorDiagnostics.delta, (sample, time, ...)

    def to_device(self) -> "StepDiagnostics": ...
    def to_cpu(self) -> "StepDiagnostics": ...
    def pin_memory(self) -> "StepDiagnostics": ...
    def broadcast_ensemble(self, n_ensemble: int) -> "StepDiagnostics": ...
    def sample_dim_size(self) -> int | None: ...
    def to_dataset(self, time: xr.DataArray) -> xr.Dataset:
        # NEW — the container's only data-export API: the delta variables with
        # the given time coordinate, ready for netCDF writing. Consumers (the
        # step-diagnostics writer) use this instead of reaching into `delta`.
        ...
```

No `scatter_spatial`: diagnostics are produced rank-locally by `predict`, never
read centrally and scattered, so `BatchData.scatter_spatial` raise-guards
instead (below).

## `fme/ace/data_loading/batch_data.py` (modified)

```python
@dataclasses.dataclass
class BatchData:
    step_diagnostics: StepDiagnostics | None = None  # NEW — per-sample corrector delta series

    # CHANGED — forward step_diagnostics via the matching container op:
    def to_device(self) -> "BatchData": ...
    def to_cpu(self) -> "BatchData": ...
    def pin_memory(self: SelfType) -> SelfType: ...
    def broadcast_ensemble(self: SelfType, n_ensemble: int) -> SelfType: ...

    # CHANGED — accept + store step_diagnostics as a constructor param:
    @classmethod
    def new_on_cpu(cls, ..., step_diagnostics: StepDiagnostics | None = None) -> "BatchData": ...
    @classmethod
    def new_on_device(cls, ..., step_diagnostics: StepDiagnostics | None = None) -> "BatchData": ...

    def __post_init__(self):  # CHANGED — also validate step_diagnostics.sample_dim_size()
        ...

    # CHANGED — these RAISE when step_diagnostics is not None. The time-touching
    # methods would silently misalign the time-indexed delta; subset_names is an
    # input-side op with undefined semantics for the sparse delta name set; and
    # scatter_spatial is a central-read→scatter input op that never sees
    # diagnostics. No current caller hits any of these with a diagnostics-bearing
    # batch, so a call here is a bug — fail loudly:
    def select_time_slice(self: SelfType, time_slice: slice) -> SelfType: ...
    def prepend(self: SelfType, initial_condition: PrognosticState) -> SelfType: ...
    def remove_initial_condition(self: SelfType, n_ic_timesteps: int) -> SelfType: ...
    def compute_derived_variables(self: SelfType, ...) -> SelfType: ...
    def get_start(self: SelfType, ...) -> PrognosticState: ...
    def get_end(self: SelfType, ...) -> PrognosticState: ...
    def subset_names(self: SelfType, names: Collection[str]) -> SelfType: ...
    def scatter_spatial(self: SelfType, global_img_shape: tuple[int, int]) -> SelfType: ...


@dataclasses.dataclass
class PairedData:
    step_diagnostics: StepDiagnostics | None = None  # NEW — first per-sample state on PairedData

    def broadcast_ensemble(self, n_ensemble: int) -> "PairedData":  # CHANGED — forward via container op
        ...

    @classmethod
    def from_batch_data(cls, prediction: BatchData, reference: BatchData) -> "PairedData":
        # CHANGED — carry prediction.step_diagnostics (the prediction→paired
        # seam the inference path goes through; currently drops per-sample state)
        ...

    # CHANGED — accept + store step_diagnostics:
    @classmethod
    def new_on_device(cls, ..., step_diagnostics: StepDiagnostics | None = None) -> "PairedData": ...
    @classmethod
    def new_on_cpu(cls, ..., step_diagnostics: StepDiagnostics | None = None) -> "PairedData": ...
```

## `fme/ace/stepper/single_module.py` (modified)

```python
class Stepper:
    def __init__(self, ..., output_masking: SpatialMasking, ...):
        # RENAMED — was output_process_func / _output_process_func, typed as a
        # bare Callable. The masking-specific name and SpatialMasking type
        # propagate everywhere the value flows (constructor param, private
        # attr, the StepperConfig builder local that feeds it), so wiring in a
        # non-masking callable is a visible type error. The builder's
        # MissingDatasetInfo fallback becomes NullSpatialMasking() (was
        # NullPostProcessFn()). No checkpoint/state schema change: the value is
        # rebuilt from config on every load.
        ...

    def step(self, args: StepArgs, wrapper=...) -> StepOutput:
        # CHANGED — also apply the output spatial masker to the carried
        # diagnostics, so delta is NaN-masked off-mask consistently with
        # .output:
        #   corrector_diagnostics=result.corrector_diagnostics
        #       .apply_output_masking(self._output_masking)
        # (_output_masking is always the NaN-fill output spatial masker or the
        # NullSpatialMasking no-op; apply_output_masking's name and contract
        # pin that restriction.)
        ...

    def predict(self, ...) -> tuple[BatchData, PrognosticState]:
        # CHANGED — attach StepOutput.stack_diagnostics(output_list) at the
        # closing BatchData.new_on_device(...) reconstruction — i.e. AFTER the
        # prepend → compute_derived → remove_initial_condition dance and the
        # get_end call, where data and series are both n_forward_steps long.
        # stack_diagnostics returns None when no output carries a delta (no
        # corrector, or nothing modified this rollout), so predict attaches
        # its result unconditionally.
        ...
```

`CoupledStepper` is unchanged: it leaves `step_diagnostics` at its `None`
default (coupled carriage is follow-up work, per the note at the top).

## `fme/ace/inference/data_writer/main.py` (modified)

```python
@dataclasses.dataclass
class DataWriterConfig:
    save_step_diagnostics: bool = False  # NEW — write autoregressive_step_diagnostics.nc

    def build_paired(self, ...) -> "PairedDataWriter":
        # CHANGED — when save_step_diagnostics, build a StepDiagnosticsWriter
        # (label="autoregressive_step_diagnostics", save_names=self.names,
        # time_coarsen=self.time_coarsen) and pass it to PairedDataWriter as
        # step_diagnostics_writer.
        ...

    def build(self, ...) -> "DataWriter":
        # CHANGED — same wiring for the single-source writer, so forcing-only
        # inference runs also get the file.
        ...


class PairedDataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        writers: list[PairedSubwriter],
        step_diagnostics_writer: StepDiagnosticsWriter | None = None,  # NEW
        ...
    ):
        # Held as a separate member, not in the homogeneous paired-writer list:
        # the paired fan-out calls append_batch(target=, prediction=, batch_time=),
        # which the step-diagnostics writer doesn't accept.
        ...

    def append_batch(self, batch: PairedData):
        # CHANGED — additionally dispatch
        #   self._step_diagnostics_writer.append_batch(
        #       batch.step_diagnostics.to_dataset(batch.time))
        # skipped when batch.step_diagnostics is None or no writer configured.
        ...

    def flush(self): ...     # CHANGED — include step-diagnostics writer
    def finalize(self): ...  # CHANGED — include step-diagnostics writer


class DataWriter(WriterABC[PrognosticState, PairedData]):
    # CHANGED — same step_diagnostics_writer member and append_batch dispatch
    # (its append_batch also receives PairedData).
    ...
```

## `fme/ace/inference/data_writer/step_diagnostics.py` (new)

```python
class StepDiagnosticsWriter:
    # Appends the xr.Dataset from StepDiagnostics.to_dataset() to
    # autoregressive_step_diagnostics.nc, modeled on RawDataWriter's file
    # handling. Consumes only the Dataset — no knowledge of StepDiagnostics
    # internals. Honors the save_names subset and applies the configured
    # time-coarsening to the dataset before appending.
    def append_batch(self, dataset: xr.Dataset): ...
    def flush(self): ...
    def finalize(self): ...
```

The delta series is written as-is (already denormalized/physical units),
consistent with `RawDataWriter`, which never denormalizes. Default inference
output is unchanged: the flag is off by default and nothing else consumes the
carriage.

---

## Tests

## `fme/core/step/test_step_diagnostics.py` (new)

```python
def test_to_device_to_cpu_pin_memory_preserve_keys():
    # GOAL: each device op returns a StepDiagnostics with the same keys and the
    # right device/pinned-memory transform applied per tensor.
    ...

def test_broadcast_ensemble_repeat_interleaves_sample_dim():
    # GOAL: leading dim grows sample→sample*n_ensemble with block ordering
    # matching repeat_interleave_batch_dim.
    ...

def test_empty_delta_is_valid():
    # GOAL: a StepDiagnostics with an empty delta constructs, every op is a
    # safe no-op, and sample_dim_size returns None.
    ...

def test_sample_dim_size():
    # GOAL: returns the leading dim of the delta tensors, or None when empty.
    ...

def test_to_dataset():
    # GOAL: returns an xr.Dataset with exactly the delta variables and the
    # given time coordinate; values round-trip.
    ...
```

## `fme/ace/data_loading/test_batch_data.py` (modified)

```python
# PARAMETERIZE each over step_diagnostics present vs. None.

def test_batch_data_forwards_step_diagnostics():
    # GOAL: to_device / to_cpu / pin_memory / broadcast_ensemble forward a
    # populated step_diagnostics consistently with data (and pass None through).
    ...

def test_batch_data_guarded_ops_raise_with_step_diagnostics():
    # GOAL: select_time_slice / prepend / remove_initial_condition /
    # compute_derived_variables / get_start / get_end / subset_names /
    # scatter_spatial raise when step_diagnostics is present, pass when None.
    ...

def test_paired_data_threads_step_diagnostics():
    # GOAL: from_batch_data carries the prediction's step_diagnostics onto the
    # PairedData; broadcast_ensemble / new_on_device / new_on_cpu forward it.
    ...
```

## `fme/core/step/test_output.py` (modified)

```python
def test_stack_diagnostics():
    # GOAL: stacks per-step deltas forward-step-aligned along dim 1 into a
    # StepDiagnostics; returns None when no output carries a delta.
    ...
```

## `fme/core/corrector/test_output.py` (modified)

```python
def test_apply_output_masking():
    # GOAL: maps the masking over delta and returns a NEW CorrectorDiagnostics
    # (input unmutated); NullSpatialMasking preserves values.
    ...
```

## `fme/ace/stepper/test_single_module.py` (modified)

```python
def test_predict_attaches_step_diagnostics():
    # GOAL: a corrector-equipped predict returns a BatchData whose
    # step_diagnostics.delta is the forward-step-aligned, denormalized per-step
    # correction — exact values asserted for a deterministic constant-offset
    # corrector; prediction values unchanged from the current branch.
    ...

def test_predict_without_corrector_has_no_step_diagnostics():
    # GOAL: no-corrector predict returns step_diagnostics is None.
    ...

def test_step_masks_corrector_diagnostics():
    # GOAL: with a NaN-fill output masker, delta is NaN exactly where .output
    # is masked and unchanged on-mask; with NullSpatialMasking (no mask
    # provider) delta is unchanged.
    ...
```

## `fme/ace/inference/data_writer/test_main.py` (modified)

```python
# Reuse the existing data-writer fixtures/helpers.

def test_step_diagnostics_writer_writes_delta_series():
    # GOAL: with save_step_diagnostics=True, autoregressive_step_diagnostics.nc
    # exists and contains exactly the corrector-modified variables with the
    # expected values and no target series — via both build_paired() and build().
    # PARAMETERIZE: save_names subset ∈ {None, subset}; time_coarsen ∈ {None, factor 2}.
    ...

def test_no_step_diagnostics_file_by_default_or_without_corrector():
    # GOAL: flag off ⇒ no file; flag on but step_diagnostics absent (no
    # corrector) ⇒ no file content.
    ...
```
