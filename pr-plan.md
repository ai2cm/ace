# Carry corrector deltas onto prediction data and add a correction netCDF writer

Carries the per-step corrector `delta` series out of `Stepper.predict` in a new
opaque `StepDiagnostics` container on `BatchData`/`PairedData`, NaN-masks the
delta consistently with the prediction at the stepper step seam, and adds an
off-by-default `autoregressive_corrections.nc` writer as the carriage's first
consumer. No aggregator, no normalizer wiring, no metric flags.

---

## `fme/core/corrector/output.py` (modified)

```python
@dataclasses.dataclass
class CorrectorDiagnostics:
    delta: TensorMapping = dataclasses.field(default_factory=dict)

    def apply_output_process_func(  # NEW — map the stepper's output masker over delta
        self, func: Callable[[TensorMapping], TensorDict]
    ) -> "CorrectorDiagnostics":
        # Returns a new CorrectorDiagnostics; input unmutated. Correct ONLY
        # because the output masker is NaN-fill (StaticSpatialMasking with
        # fill_value=NaN, or the NullPostProcessFn no-op): NaN off-mask matches
        # masked_output − masked_snapshot. A finite fill would inject a spurious
        # offset and break the delta = output − snapshot invariant; this is
        # documented inline.
        ...
```

## `fme/ace/data_loading/step_diagnostics.py` (new)

Opaque per-sample diagnostics container, modeled on `StepperState`. Holds the
stacked per-step correction series, aligned with the prediction `data` along
`(sample, time, ...)`. When attached it always carries a non-empty `delta`
(`Stepper.predict` attaches `None` instead of an empty container), so
`step_diagnostics is None` is the single "no correction" gate. It deliberately
has **no** time-slice or prepend-time op — the series is attached only after
time-windowing is finished, and the time-touching `BatchData` methods guard
against carrying it (below).

```python
@dataclasses.dataclass
class StepDiagnostics:
    delta: TensorMapping  # stacked per-step CorrectorDiagnostics.delta, (sample, time, ...)

    def to_device(self) -> "StepDiagnostics": ...
    def to_cpu(self) -> "StepDiagnostics": ...
    def pin_memory(self) -> "StepDiagnostics": ...
    def scatter_spatial(self, global_img_shape: tuple[int, int]) -> "StepDiagnostics": ...
    def broadcast_ensemble(self, n_ensemble: int) -> "StepDiagnostics": ...
    def sample_dim_size(self) -> int | None: ...
```

## `fme/ace/data_loading/batch_data.py` (modified)

```python
@dataclasses.dataclass
class BatchData:
    step_diagnostics: StepDiagnostics | None = None  # NEW — per-sample corrector delta series

    # CHANGED — forward step_diagnostics via the matching container op:
    def to_device(self) -> "BatchData": ...
    def to_cpu(self) -> "BatchData": ...
    def scatter_spatial(self, global_img_shape: tuple[int, int]) -> "BatchData": ...
    def pin_memory(self: SelfType) -> SelfType: ...
    def broadcast_ensemble(self: SelfType, n_ensemble: int) -> SelfType: ...

    # CHANGED — accept + store step_diagnostics as a constructor param:
    @classmethod
    def new_on_cpu(cls, ..., step_diagnostics: StepDiagnostics | None = None) -> "BatchData": ...
    @classmethod
    def new_on_device(cls, ..., step_diagnostics: StepDiagnostics | None = None) -> "BatchData": ...

    def __post_init__(self):  # CHANGED — also validate step_diagnostics.sample_dim_size()
        ...

    def subset_names(self: SelfType, names: Collection[str]) -> SelfType:
        # CHANGED — pass step_diagnostics through UNCHANGED (delta is its own
        # sparse name set; not filtered to `names`)
        ...

    # CHANGED — time-touching methods RAISE when step_diagnostics is not None.
    # delta is time-indexed and these would silently misalign it; no current
    # caller windows a diagnostics-bearing batch, so a call here is a bug:
    def select_time_slice(self: SelfType, time_slice: slice) -> SelfType: ...
    def prepend(self: SelfType, initial_condition: PrognosticState) -> SelfType: ...
    def remove_initial_condition(self: SelfType, n_ic_timesteps: int) -> SelfType: ...
    def compute_derived_variables(self: SelfType, ...) -> SelfType: ...
    def get_start(self: SelfType, ...) -> PrognosticState: ...
    def get_end(self: SelfType, ...) -> PrognosticState: ...


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
    def step(self, args: StepArgs, wrapper=...) -> StepOutput:
        # CHANGED — also apply _output_process_func to the carried diagnostics,
        # so delta is NaN-masked off-mask consistently with .output:
        #   corrector_diagnostics=result.corrector_diagnostics
        #       .apply_output_process_func(self._output_process_func)
        ...

    def predict(self, ...) -> tuple[BatchData, PrognosticState]:
        # CHANGED — collect each StepOutput.corrector_diagnostics.delta from the
        # prediction generator, stack forward-step-aligned via
        # stack_list_of_tensor_dicts(..., time_dim=1), and attach at the closing
        # BatchData.new_on_device(...) reconstruction — i.e. AFTER the
        # prepend → compute_derived → remove_initial_condition dance, where data
        # and series are both n_forward_steps long. Attach None when the stacked
        # series has no keys (no corrector, or nothing modified this rollout).
        ...
```

`CoupledStepper` is unchanged: it leaves `step_diagnostics` at its `None`
default (no correction file for coupled runs).

## `fme/ace/inference/data_writer/main.py` (modified)

```python
@dataclasses.dataclass
class DataWriterConfig:
    save_correction_files: bool = False  # NEW — write autoregressive_corrections.nc

    def build_paired(self, ...) -> "PairedDataWriter":
        # CHANGED — when save_correction_files, build a single-source
        # RawDataWriter(label="autoregressive_corrections", save_names=self.names, ...),
        # wrapped with self.time_coarsen.build(...) (single-source, NOT
        # build_paired — there is no target series), and pass it to
        # PairedDataWriter as correction_writer.
        ...


class PairedDataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        writers: list[PairedSubwriter],
        correction_writer: RawDataWriter | TimeCoarsen | None = None,  # NEW
        ...
    ):
        # Held as a separate member, not in the homogeneous paired-writer list:
        # the paired fan-out calls append_batch(target=, prediction=, batch_time=),
        # which a single-source writer can't accept.
        ...

    def append_batch(self, batch: PairedData):
        # CHANGED — additionally dispatch
        #   self._correction_writer.append_batch(
        #       data=dict(batch.step_diagnostics.delta), batch_time=batch.time)
        # skipped when batch.step_diagnostics is None or no correction writer.
        ...

    def flush(self): ...     # CHANGED — include correction writer
    def finalize(self): ...  # CHANGED — include correction writer
```

The delta series is written as-is (already denormalized/physical units),
consistent with `RawDataWriter`, which never denormalizes. Default inference
output is unchanged: the flag is off by default and nothing else consumes the
carriage.

---

## Tests

## `fme/ace/data_loading/test_step_diagnostics.py` (new)

```python
def test_to_device_to_cpu_pin_memory_preserve_keys():
    # GOAL: each device op returns a StepDiagnostics with the same keys and the
    # right device/pinned-memory transform applied per tensor.
    ...

def test_scatter_spatial_slices_local_chunk():
    # GOAL: scatter_spatial applies the same spatial slicing as BatchData.data.
    ...

def test_broadcast_ensemble_repeat_interleaves_sample_dim():
    # GOAL: leading dim grows sample→sample*n_ensemble with block ordering
    # matching repeat_interleave_batch_dim.
    ...

def test_sample_dim_size():
    # GOAL: returns the leading dim of the delta tensors, or None when empty.
    ...
```

## `fme/ace/data_loading/test_batch_data.py` (modified)

```python
# PARAMETERIZE each over step_diagnostics present vs. None.

def test_batch_data_forwards_step_diagnostics():
    # GOAL: to_device / to_cpu / scatter_spatial / pin_memory /
    # broadcast_ensemble / subset_names forward a populated step_diagnostics
    # consistently with data (and pass None through). subset_names does not
    # drop or filter it.
    ...

def test_batch_data_time_ops_raise_with_step_diagnostics():
    # GOAL: select_time_slice / prepend / remove_initial_condition /
    # compute_derived_variables / get_start / get_end raise when
    # step_diagnostics is present, pass when None.
    ...

def test_paired_data_threads_step_diagnostics():
    # GOAL: from_batch_data carries the prediction's step_diagnostics onto the
    # PairedData; broadcast_ensemble / new_on_device / new_on_cpu forward it.
    ...
```

## `fme/core/corrector/test_output.py` (modified)

```python
def test_apply_output_process_func():
    # GOAL: maps the func over delta and returns a NEW CorrectorDiagnostics
    # (input unmutated); identity func preserves values.
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
    # is masked and unchanged on-mask; with NullPostProcessFn (no mask
    # provider) delta is unchanged.
    ...
```

## `fme/ace/inference/data_writer/test_main.py` (modified)

```python
# Reuse the existing data-writer fixtures/helpers.

def test_correction_writer_writes_delta_series():
    # GOAL: with save_correction_files=True, autoregressive_corrections.nc
    # exists and contains exactly the corrector-modified variables with the
    # expected values and no target series.
    # PARAMETERIZE: save_names subset ∈ {None, subset}; time_coarsen ∈ {None, factor 2}.
    ...

def test_no_correction_file_by_default_or_without_corrector():
    # GOAL: flag off ⇒ no correction file; flag on but step_diagnostics absent
    # (no corrector) ⇒ no correction file content.
    ...
```
