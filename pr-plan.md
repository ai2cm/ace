# Carry corrector correction deltas on a new `StepOutput`

Correctors return a `CorrectorOutput` (corrected data + a `CorrectorDiagnostics`
carrying the per-variable correction `delta` + corrector state) instead of a
`(corrected, corrector_state)` tuple, and every step / stepper-generator hop
returns a new `StepOutput` value object carrying that diagnostics payload up to
`Stepper.predict`, which discards it. Behavior-preserving: corrected output,
rollout state, loss, logged metrics, and saved files are identical to `main`;
existing checkpoints load and run unchanged. No `BatchData`/`PairedData`,
aggregator, writer, loss, or config changes.

The `CorrectorDiagnostics`, `CorrectorOutput`, and `build_corrector_diagnostics`
value objects/helper already exist in `fme/core/corrector/output.py` (no consumer
yet); this PR wires them in.

---

## `fme/core/corrector/registry.py` (modified)

```python
class Correction(Protocol):
    def touched_names(self, gen_data: TensorMapping) -> set[str]:  # NEW
        """Names in ``gen_data`` this correction writes when applied to it.

        Resolved against ``gen_data`` because the written keys depend on the
        data's variable naming (e.g. surface pressure is ``PRESsfc`` or ``PS``;
        a per-level field expands to its present level names; ``sst``/``hfds``
        may or may not be present). The result is static across a run's steps —
        the variable naming does not change — but is not derivable from config
        fields alone for corrections that write through ``AtmosphereData`` /
        ``OceanData`` stackers. Config-only corrections ignore ``gen_data``.
        """
        ...

    def __call__(self, ...) -> tuple[TensorDict, CorrectorState | None]: ...
        # unchanged — Correction sub-operations keep their tuple return


class CorrectorABC(abc.ABC):
    @abc.abstractmethod
    def touched_names(self, gen_data: TensorMapping) -> set[str]:  # NEW
        ...

    @abc.abstractmethod
    def __call__(self, ...) -> CorrectorOutput:  # CHANGED — was tuple[TensorDict, CorrectorState | None]
        ...


class CorrectionSequence(CorrectorABC):
    def touched_names(self, gen_data: TensorMapping) -> set[str]:  # NEW — union over corrections
        return set().union(*(c.touched_names(gen_data) for c in self._corrections))

    def __call__(self, input_data, gen_data, forcing_data, corrector_state) -> CorrectorOutput:  # CHANGED
        # Snapshot ONLY the declared touched names from incoming gen_data (hold
        # references; corrections apply out-of-place so entry tensors are not
        # mutated). Apply corrections unchanged, then build populated diagnostics.
        touched = self.touched_names(gen_data)
        snapshot = {name: gen_data[name] for name in touched}
        gen_data = dict(gen_data)
        for correction in self._corrections:
            gen_data, corrector_state = correction(input_data, gen_data, forcing_data, corrector_state)
        corrected = dict(gen_data)
        return CorrectorOutput(
            corrected=corrected,
            diagnostics=build_corrector_diagnostics(snapshot, corrected, touched),  # not detached here
            corrector_state=corrector_state,
        )


class EpochScheduledCorrector(CorrectorABC):
    def touched_names(self, gen_data: TensorMapping) -> set[str]:  # NEW — delegate to wrapped
        return self._wrapped.touched_names(gen_data)

    def __call__(self, input_data, gen_data, forcing_data, corrector_state) -> CorrectorOutput:  # CHANGED
        if self._corrector_disabled and self._training:
            # applied nothing -> corrected passthrough, empty diagnostics
            return CorrectorOutput(corrected=dict(gen_data), corrector_state=corrector_state)
        return self._wrapped(input_data, gen_data, forcing_data, corrector_state)
```

## `fme/core/corrector/utils.py` (modified)

```python
class ForcePositive:
    def touched_names(self, gen_data: TensorMapping) -> set[str]:  # NEW — config-only
        return set(self.names)
```

## `fme/core/corrector/atmosphere.py` (modified)

Each correction resolves its written keys against `gen_data` via the same
`AtmosphereData` stacker its `__call__` writes through, so the declaration and
the write cannot diverge (the drift test is the backstop).

```python
class ConserveDryAir:
    def touched_names(self, gen_data) -> set[str]:  # NEW — the surface-pressure key present in gen_data
        ...  # AtmosphereData(gen_data) resolved "surface_pressure" key (PRESsfc | PS)

class ZeroGlobalMeanMoistureAdvection:
    def touched_names(self, gen_data) -> set[str]:  # NEW
        return {"tendency_of_total_water_path_due_to_advection"}

class MoistureBudgetCorrection:
    def touched_names(self, gen_data) -> set[str]:  # NEW — by terms_to_modify
        # precipitation -> {precip key};  evaporation -> {evap key}
        # advection_and_* -> + {"tendency_of_total_water_path_due_to_advection"}
        ...

class TotalEnergyBudgetCorrection:
    def touched_names(self, gen_data) -> set[str]:  # NEW — every air_temperature level key
        ...  # AtmosphereData(gen_data).get_all_vertical_level_names("air_temperature")
```

## `fme/core/corrector/ocean.py` (modified)

```python
class SeaIceFractionCorrection:
    def touched_names(self, gen_data) -> set[str]:  # NEW — config-only
        return {self.config.sea_ice_fraction_name, *self.config.zero_where_ice_free_names}

class SurfaceEnergyFluxCorrection:
    def touched_names(self, gen_data) -> set[str]:  # NEW — the hfds key present in gen_data
        return {"hfds"} if "hfds" in gen_data else {"hfds_total_area"}

class OceanHeatContentCorrection:
    def touched_names(self, gen_data) -> set[str]:  # NEW — thetao levels (+ sst if present)
        ...  # {f"thetao_{k}" for k in levels} | ({"sst"} if "sst" in gen_data else set())
```

## `fme/core/corrector/ice.py` (modified)

`IceCorrector` is a bare `CorrectionSequence` subclass, so it gets
`touched_names` / `CorrectorOutput` for free; only the leaf correction declares.

```python
class IceBudgetCorrection:
    def touched_names(self, gen_data) -> set[str]:  # NEW — config-only, from corrected_variables
        # union of each processed prognostic key and its three budget-term names
        ...
```

## `fme/core/step/output.py` (new)

Placed in `fme/core/step/` (not `fme/core/corrector/`) so the corrector package
does not import `StepperState`. All imports stay within `fme.core`.

```python
@dataclasses.dataclass
class StepOutput:
    """One step's denormalized output plus its corrector diagnostics + state."""
    output: TensorDict
    stepper_state: StepperState | None = None
    corrector_diagnostics: CorrectorDiagnostics = dataclasses.field(
        default_factory=CorrectorDiagnostics
    )
```

## `fme/core/step/__init__.py` (modified)

```python
from .output import StepOutput  # NEW
```

## `fme/core/step/step.py` (modified)

```python
class StepABC(abc.ABC):
    @abc.abstractmethod
    def step(self, args: StepArgs, wrapper=...) -> StepOutput:  # CHANGED — was tuple[TensorDict, StepperState | None]
        ...
```

## `fme/core/step/single_module.py` (modified)

```python
def step_with_adjustments(...) -> StepOutput:  # CHANGED — was tuple[TensorDict, StepperState | None]
    ...
    diagnostics = CorrectorDiagnostics()  # NEW — empty when no corrector
    if corrector is not None:
        corrector_state = stepper_state.corrector_state if stepper_state is not None else None
        result = corrector(input, output, next_step_input_data, corrector_state)  # CHANGED — CorrectorOutput
        output = result.corrected
        # Detach diagnostic tensors here, once, unconditionally (step-boundary
        # detach keeps the corrector autograd-agnostic; PR5's attach/detach
        # control is a one-line change at this same seam).
        diagnostics = CorrectorDiagnostics(
            delta={k: v.detach() for k, v in result.diagnostics.delta.items()}
        )
        if result.corrector_state is not None:
            stepper_state = StepperState(corrector_state=result.corrector_state)

    # Post-corrector adjustments still run AFTER and are excluded from diagnostics.
    # Case-2 disjointness guard: their written names must not overlap the
    # corrector's touched names, or delta = output - snapshot stops being exact.
    post_corrector_names: set[str] = set(prescribed_prognostic_names)  # NEW
    if ocean is not None:
        post_corrector_names.add(ocean.surface_temperature_name)
    overlap = post_corrector_names & set(diagnostics.delta)  # delta keys == corrector touched names
    if overlap:
        raise ValueError(
            f"post-corrector adjustment names overlap corrector touched names: {overlap}"
        )

    if ocean is not None:
        output = ocean(input, output, next_step_input_data)
    for name in prescribed_prognostic_names:
        ...  # unchanged
    return StepOutput(output=output, stepper_state=stepper_state, corrector_diagnostics=diagnostics)


class SingleModuleStep(StepABC):
    def step(self, args, wrapper=...) -> StepOutput:  # CHANGED — delegates to step_with_adjustments
        ...
```

### Critical detail — the disjointness guard uses the corrector's *resolved* touched names

The guard compares post-corrector adjustment names against the keys of the
populated `delta` (which are exactly the corrector's resolved `touched_names` for
this step) — no second `touched_names(gen_data)` call. A disabled
`EpochScheduledCorrector` yields an empty `delta`, so the guard is trivially
satisfied (it wrote nothing). This holds today (OHC corrects predicted `sst`;
`Ocean` prescribes `sst` only in atmosphere-only configs that don't predict it —
mutually exclusive) and fails loudly on any future overlapping config.

## `fme/core/step/secondary_module.py` (modified)

```python
class SecondaryModuleStep(StepABC):
    def step(self, args, wrapper=...) -> StepOutput:  # CHANGED — returns step_with_adjustments(...) result
        ...
```

## `fme/core/step/radiation.py` (modified)

```python
class SeparateRadiationStep(StepABC):
    def step(self, args, wrapper=...) -> StepOutput:  # CHANGED
        ...
```

## `fme/core/step/multi_call.py` (modified)

Wraps another step; not in the issue's "four step implementations" list but
must change for the contract to hold. The secondary `_multi_call.step` output is
merged into the wrapped step's output as today; its diagnostics are discarded —
only the wrapped step's diagnostics/state are carried.

```python
class MultiCallStep(StepABC):
    def step(self, args, wrapper=...) -> StepOutput:  # CHANGED — was tuple[TensorDict, StepperState | None]
        wrapped = self._wrapped_step.step(args=args, wrapper=wrapper)  # StepOutput
        output = wrapped.output
        if self._multi_call is not None:
            multi_called = self._multi_call.step(args=args, wrapper=wrapper)  # StepOutput; diagnostics discarded
            output = {**multi_called.output, **output}
        return StepOutput(
            output=output,
            stepper_state=wrapped.stepper_state,
            corrector_diagnostics=wrapped.corrector_diagnostics,
        )
```

## `fme/ace/step/fcn3.py` (modified)

```python
class FCN3Step(StepABC):
    def step(self, args, wrapper=...) -> StepOutput:  # CHANGED
        ...
```

## `fme/ace/stepper/single_module.py` (modified)

```python
def process_prediction_generator_list(  # CHANGED — consumes StepOutput
    output_list: list[StepOutput],  # was list[tuple[TensorDict, StepperState | None]]
    ...,
) -> BatchData:
    output_dicts = [item.output for item in output_list]
    terminal_state = output_list[-1].stepper_state if output_list else None
    # corrector_diagnostics intentionally NOT read here (PR3+)
    ...


class Stepper:
    def step(self, args, wrapper=...) -> StepOutput:  # CHANGED
        args = args.apply_input_process_func(self._input_process_func)
        result = self._step_obj.step(args=args, wrapper=wrapper)  # StepOutput
        return StepOutput(
            output=self._output_process_func(result.output),
            stepper_state=result.stepper_state,
            corrector_diagnostics=result.corrector_diagnostics,
        )

    def get_prediction_generator(self, ...) -> Generator[StepOutput, None, None]:  # CHANGED
        ...

    def predict_generator(self, ...) -> Generator[StepOutput, None, None]:  # CHANGED
        ...
        result = self.step(StepArgs(...), wrapper=checkpoint)  # StepOutput
        state = result.output
        stepper_state = result.stepper_state
        yield result
        state = optimizer.detach_if_using_gradient_accumulation(state)

    # def predict(...) -> tuple[BatchData, PrognosticState]:  # UNCHANGED externally —
    #   diagnostics are discarded at this boundary; no BatchData/PrognosticState change.


class TrainStepper:
    def _accumulate_loss(self, ...):  # CHANGED — unwrap .output only; do NOT retain diagnostics
        ...
        gen_step = next(output_iterator).output  # was: gen_step, _ = next(output_iterator)
        ...
```

## `fme/coupled/stepper.py` (modified)

Unwrap `.output` at the two generator-consumption sites and wrap into a
`StepOutput` when calling the (now `StepOutput`-consuming)
`process_prediction_generator_list`; keep forcing `stepper_state=None` as today.
`ComponentStepPrediction` stays carrying only `.data`.

```python
class CoupledStepper:
    def get_prediction_generator(self, ...):  # CHANGED — unwrap .output at the two next(...) sites
        atmos_step = next(atmos_generator).output  # was: atmos_step, _ = next(atmos_generator)
        ...
        ocean_step = next(iter(self.ocean.get_prediction_generator(...))).output  # was: ocean_step, _ = ...
        ...

    def _process_prediction_generator_list(self, output_list, forcing_data):  # CHANGED — wrap into StepOutput
        atmos_data = process_prediction_generator_list(
            [StepOutput(output=x.data, stepper_state=None) for x in output_list if x.realm == "atmosphere"],
            ...,
        )
        ocean_data = process_prediction_generator_list(
            [StepOutput(output=x.data, stepper_state=None) for x in output_list if x.realm == "ocean"],
            ...,
        )
        ...
```

---

## Tests

## `fme/core/corrector/test_atmosphere.py` (modified) · `test_ocean.py` (modified)

Build on the existing per-option corrector fixtures. For each corrector
(`AtmosphereCorrector`, `OceanCorrector`) and each enabled correction option:

```python
def test_corrector_output_behavior_preserving(...):
    # GOAL: CorrectorOutput.corrected equals current main output for identical inputs.
    # PARAMETERIZE: over each enabled correction option.

def test_corrector_delta_matches_declared_touched_names(...):
    # GOAL: diagnostics.delta keys == declared touched_names(gen_data);
    #       delta[name] == corrected[name] - input[name] exactly.
    # Prefer a deterministic constant-offset/scale scenario so delta is hand-computable.

def test_corrector_delta_empty_when_nothing_modified(...):
    # GOAL: no enabled option that modifies a field -> empty delta.

def test_touched_names_drift(...):  # THE BACKSTOP
    # GOAL: declared touched_names(gen_data) == {name : not torch.equal(corrected[name], input[name])}
    #       under a representative input, for both correctors across enabled options.
```

## `fme/core/corrector/test_registry.py` (modified)

```python
def test_correction_sequence_touched_names_is_union(...):
    # GOAL: CorrectionSequence.touched_names == union of its corrections' touched_names.

def test_epoch_scheduled_corrector_disabled_returns_empty_diagnostics(...):
    # GOAL: disabled train-mode step -> CorrectorOutput(corrected passthrough, empty delta);
    #       enabled -> delegates and passes the wrapped CorrectorOutput through.
```

## `fme/core/step/test_step.py` (modified)

```python
def test_step_returns_step_output_with_populated_delta(...):
    # GOAL: each step impl returns StepOutput; corrector_diagnostics.delta populated
    #       exactly when its corrector modifies variables, empty otherwise;
    #       output / stepper_state unchanged vs main.
    # PARAMETERIZE: over the step implementations (single/secondary/radiation/multi_call).

def test_step_diagnostics_detached(...):
    # GOAL: every delta tensor has requires_grad == False (detached at the step boundary).

def test_boundary_disjointness_passes_when_disjoint(...):
    # GOAL: corrector touched names disjoint from post-corrector adjustment names -> ok.

def test_boundary_disjointness_raises_on_overlap(...):
    # GOAL: a config where a prescribed_prognostic / Ocean SST name overlaps a corrector
    #       touched name raises ValueError.
```

## `fme/ace/stepper/test_single_module.py` (modified)

```python
def test_predict_returns_corrected_only_batchdata(...):
    # GOAL: Stepper.predict exposes no diagnostics field; rollout state is the corrected
    #       output; result is byte-for-byte identical to main for a corrector-equipped config.
```

(FCN3's step seam is exercised via its existing step test alongside the others.)

---

## Design note — why `touched_names` takes `gen_data`

Physics corrections write through `AtmosphereData`/`OceanData` stackers, so the
written gen_data keys (`PRESsfc` vs `PS`, present air-temperature/thetao level
names, `sst`/`hfds` presence) are resolved from the data, not from config fields
— and `_build` is not given the variable naming. `touched_names` therefore takes
`gen_data` and resolves at sequence entry, where the data is in hand; the result
is still static across a run's steps. Config-only corrections (`ForcePositive`,
`SeaIceFractionCorrection`, `IceBudgetCorrection`) ignore the argument.
Alternative considered: thread the resolved names into each correction at build
time — rejected because `_build` has no variable list and `DatasetInfo` carries
none.

## Open Questions

- **Disjointness guard input.** The guard reads the populated `delta` keys
  (equal to the corrector's resolved touched names for the step) rather than
  calling `touched_names(gen_data)` a second time. Equivalent and avoids a
  redundant stacker pass; flagging in case a reviewer prefers an explicit
  `corrector.touched_names(...)` call at the seam.
