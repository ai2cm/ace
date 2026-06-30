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

Each `Correction` now returns **only the fields it modified** (today they return
the full `gen_data`); `CorrectionSequence` dict-updates `gen_data` with that
subset and accumulates the modified-name set from its keys. The correction's
returned dict is the single source of truth for what it changed — it is exactly
what gets applied — so the modified-name set is load-bearing and cannot silently
drift from the write. The `delta` is built over that accumulated set.

---

## `fme/core/corrector/output.py` (modified)

Add a convenience property to the existing `CorrectorOutput` value object (the
dataclasses are otherwise unchanged):

```python
@dataclasses.dataclass
class CorrectorOutput:
    corrected: TensorDict
    diagnostics: CorrectorDiagnostics = ...
    corrector_state: CorrectorState | None = None

    @property
    def modified_names(self) -> KeysView[str]:  # NEW — the corrector-modified variables
        return self.diagnostics.delta.keys()
```

## `fme/core/corrector/registry.py` (modified)

```python
class Correction(Protocol):
    """A correction applied to generated data.

    `__call__` implementations must return only the fields this correction
    modified (see the Returns annotation below) — the dict is exactly what the
    caller applies.
    """

    def __call__(self, ...) -> tuple[TensorDict, CorrectorState | None]: ...
        # CONTRACT CHANGE (not a signature change): the returned TensorDict must
        # contain ONLY the fields this correction modified — not the full gen_data.
        # The caller dict-updates gen_data with it and unions its keys into the
        # modified-name set. (Today each correction returns the full gen_data.)
        #
        # This contract is load-bearing, so it is documented in two places that
        # travel with the code: this `Correction` class docstring and an explicit
        # `Returns:` annotation on each concrete `__call__` ("the TensorDict
        # contains only the fields modified by this correction"). Inline docs are
        # the primary guard against drift (the backstop test is deliberately not
        # comprehensive — see Tests).


class CorrectorABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, ...) -> CorrectorOutput:  # CHANGED — was tuple[TensorDict, CorrectorState | None]
        ...
    # the modified-name set is derived from what corrections return (no separate
    # per-correction declaration method)


class CorrectionSequence(CorrectorABC):
    def __call__(self, input_data, gen_data, forcing_data, corrector_state) -> CorrectorOutput:  # CHANGED
        snapshot = dict(gen_data)        # references at entry; corrections apply out-of-place,
                                         # so entry tensors are never mutated
        gen_data = dict(gen_data)
        modified: set[str] = set()
        for correction in self._corrections:
            changed, corrector_state = correction(input_data, gen_data, forcing_data, corrector_state)
            gen_data.update(changed)     # `changed` holds ONLY the fields this correction modified
            modified |= changed.keys()
        corrected = dict(gen_data)
        return CorrectorOutput(
            corrected=corrected,
            diagnostics=build_corrector_diagnostics(snapshot, corrected, modified),  # not detached here
            corrector_state=corrector_state,
        )


class EpochScheduledCorrector(CorrectorABC):
    def __call__(self, input_data, gen_data, forcing_data, corrector_state) -> CorrectorOutput:  # CHANGED
        if self._corrector_disabled and self._training:
            # applied nothing -> corrected passthrough, empty diagnostics
            return CorrectorOutput(corrected=dict(gen_data), corrector_state=corrector_state)
        return self._wrapped(input_data, gen_data, forcing_data, corrector_state)
```

The individual `Correction.__call__`s keep their `tuple[TensorDict,
CorrectorState | None]` return type — only their *content* contract changes
(return the modified subset, not the full `gen_data`). Only `CorrectorABC.__call__`
changes its return type, to `CorrectorOutput`.

## `fme/core/corrector/utils.py` (modified)

`ForcePositive.__call__` returns only the clamped fields it writes. It keeps its
existing `keep_gradient` flag: `_force_positive` applies the clamp through the
straight-through estimator (`replace_value_keep_gradient`) when set, so the
modified-only return must **subset `_force_positive`'s output** (preserving the
STE-applied tensors) rather than recompute a plain `torch.clamp` — otherwise the
gradient-through-clamp behavior would silently regress:

```python
@dataclasses.dataclass
class ForcePositive:
    names: list[str]
    keep_gradient: bool = False  # unchanged

    def __call__(self, input_data, gen_data, forcing_data, corrector_state):  # modified-only return
        clamped = _force_positive(gen_data, self.names, keep_gradient=self.keep_gradient)
        return {name: clamped[name] for name in self.names}, corrector_state
```

## `fme/core/corrector/atmosphere.py` (modified)

Each correction's helper still computes over the full `AtmosphereData` (it needs
the surrounding fields), but `__call__` now returns **only the fields the helper
wrote**, resolved against `gen_data` via the same `AtmosphereData` stacker the
write goes through — so the returned set and the write cannot diverge. The
modified fields per correction:

- `ConserveDryAir` → the surface-pressure key present in `gen_data` (`PRESsfc` | `PS`).
- `ZeroGlobalMeanMoistureAdvection` → `{"tendency_of_total_water_path_due_to_advection"}`.
- `MoistureBudgetCorrection` → the keys implied by `terms_to_modify`
  (precipitation → precip key; evaporation → evap key; `advection_and_*` →
  + `"tendency_of_total_water_path_due_to_advection"`).
- `TotalEnergyBudgetCorrection` → every `air_temperature` level key
  (`AtmosphereData(gen_data).get_all_vertical_level_names("air_temperature")`).

## `fme/core/corrector/ocean.py` (modified)

Same pattern — `__call__` returns only the modified subset. As in `utils.py`,
where a correction applies clamps through the `keep_gradient` straight-through
estimator (`SeaIceFractionCorrection` carries the same flag, and
`SeaIceFractionConfig.__call__` STE-clamps the 0–1 bound and the negative-ocean
rebalance), the subset must come from that helper's STE-applied output, not a
re-clamp, so the gradient path is preserved.

- `SeaIceFractionCorrection` → `{sea_ice_fraction_name, *zero_where_ice_free_names}` (config).
- `SurfaceEnergyFluxCorrection` → the `hfds` key present in `gen_data`
  (`"hfds"` or `"hfds_total_area"`).
- `OceanHeatContentCorrection` → the `thetao` level keys it scales (+ `"sst"` if present).

## `fme/core/corrector/ice.py` (modified)

`IceCorrector` is a bare `CorrectionSequence` subclass, so it gets the
`CorrectorOutput` accumulation for free; only the leaf correction changes its
return:

- `IceBudgetCorrection` → returns only the processed prognostic keys and their
  three budget-term names (config, from `corrected_variables`); empty subset when
  `corrected_variables is None`.

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
        # detach keeps the corrector autograd-agnostic). Kept inline for this PR;
        # a `CorrectorDiagnostics.detach()` convenience helper is deferred to the
        # later PR that makes the attach/detach choice conditional on the
        # optimization (where some deltas must stay attached).
        diagnostics = CorrectorDiagnostics(
            delta={k: v.detach() for k, v in result.diagnostics.delta.items()}
        )
        if result.corrector_state is not None:
            stepper_state = StepperState(corrector_state=result.corrector_state)

    # Post-corrector adjustments still run AFTER and are excluded from diagnostics.
    # Case-2 disjointness guard: their written names must not overlap the
    # corrector's modified names, or delta = output - snapshot stops being exact.
    post_corrector_names: set[str] = set(prescribed_prognostic_names)  # NEW
    if ocean is not None:
        post_corrector_names.add(ocean.surface_temperature_name)
    overlap = post_corrector_names & set(diagnostics.delta)  # delta keys == corrector modified names
    if overlap:
        raise ValueError(
            f"post-corrector adjustment names overlap corrector modified names: {overlap}"
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

### Critical detail — the disjointness guard uses the corrector's modified-name set (the delta keys)

The guard compares post-corrector adjustment names against the keys of the
populated `delta` (which are exactly the corrector's modified names for this step,
i.e. `result.modified_names`). A disabled `EpochScheduledCorrector` yields an
empty `delta`, so the guard is trivially satisfied (it wrote nothing). This holds
today (OHC corrects predicted `sst`;
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
    # corrector_diagnostics intentionally NOT read here (a follow-up PR carries them)
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
    #       (This also guards the modified-set: a correction that fails to return a
    #       field it changed would drop that change from `corrected` and fail here.)
    # PARAMETERIZE: over each enabled correction option.

def test_corrector_delta_matches_modified_returns(...):
    # GOAL: diagnostics.delta keys == union of the corrections' returned keys
    #       (== CorrectorOutput.modified_names); delta[name] == corrected[name] - input[name] exactly.
    # Prefer a deterministic constant-offset/scale scenario so delta is hand-computable.

def test_corrector_delta_empty_when_nothing_modified(...):
    # GOAL: no enabled option that modifies a field -> empty delta.

def test_modified_returns_no_uncorrected_fields(...):  # THE BACKSTOP (deliberately narrow)
    # GOAL: guard the most-likely regression — a correction blindly returning the full
    #       gen_data (or any field outside its responsibility). Assert that under a
    #       representative input, the corrector's modified-name set (returned keys / delta
    #       keys) contains NO field the corrector is not responsible for correcting; i.e.
    #       at least one un-corrected gen_data field is present and is NOT in the set.
    # NOT exhaustive reverse-equality: a correction legitimately returns the fields it is
    #       responsible for even when a given field's delta is zero this step (e.g.
    #       ForcePositive on already-non-negative data), so modified-set == {numerically
    #       changed} is too strong. The contract is held primarily by the inline
    #       Correction docs (above), not by this test.
```

The modified-name set is "the fields each correction is responsible for writing"
(the returned keys), which under a representative input normally coincides with
the numerically-changed fields but may legitimately include a corrected field
whose `delta` is exactly zero this step. `delta[name] == corrected[name] −
input[name]` still holds exactly (zero for such a field).

## `fme/core/corrector/test_ice.py` (new)

`IceCorrector` is a bare `CorrectionSequence` subclass whose only leaf is
`IceBudgetCorrection`, so it needs its own coverage of the modified-subset
contract:

```python
def test_ice_corrector_output_behavior_preserving(...):
    # GOAL: IceCorrector's CorrectorOutput.corrected equals current main output.

def test_ice_budget_correction_returns_only_modified(...):
    # GOAL: IceBudgetCorrection returns only the processed prognostic keys + its three
    #       budget-term names (from corrected_variables); empty subset when
    #       corrected_variables is None. delta keys == those names; no uncorrected field.
```

## `fme/core/corrector/test_registry.py` (modified)

```python
def test_correction_sequence_accumulates_modified_keys(...):
    # GOAL: CorrectionSequence applies each correction's returned subset to gen_data and
    #       the resulting delta keys == the union of the corrections' returned keys.

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
    # GOAL: corrector modified names disjoint from post-corrector adjustment names -> ok.

def test_boundary_disjointness_raises_on_overlap(...):
    # GOAL: a config where a prescribed_prognostic / Ocean SST name overlaps a corrector
    #       modified name raises ValueError.
```

## `fme/ace/stepper/test_single_module.py` (modified)

```python
def test_predict_returns_corrected_only_batchdata(...):
    # GOAL: Stepper.predict exposes no diagnostics field; rollout state is the corrected
    #       output; result is byte-for-byte identical to main for a corrector-equipped config.
```

(FCN3's step seam is exercised via its existing step test alongside the others.)

---

## Design note — why corrections return only their modified fields

The diagnostics need the set of variables the corrector changed and the per-name
`delta`. Rather than a separate declaration on each correction (a method living
apart from the code that performs the write, which can silently drift from it),
each `Correction.__call__` returns **only the fields it modified**.
`CorrectionSequence` applies that subset to `gen_data` and unions its keys into
the modified-name set, then builds `delta` over that set. Because the returned
dict is exactly what gets applied, the modified-name set is load-bearing: a
correction that fails to return a field it changed would drop that change from the
output and fail the behavior-preservation test, so the declaration cannot diverge
from the write. The reverse direction — a correction returning a field outside its
responsibility — is held primarily by the inline `Correction` docs and a narrow
backstop test (no un-corrected field is propagated), not by an exhaustive
returned-keys-equal-numerically-changed assertion: a correction legitimately
returns the fields it is responsible for even when one has zero `delta` this step.

The written keys still depend on data naming (`PRESsfc` vs `PS`, present
air-temperature/thetao level names, `sst`/`hfds` presence), but each correction
resolves them against `gen_data` via the same stacker its write goes through, at
the moment it returns — so no resolved-name list needs threading in at build time.

Alternative considered: a `touched_names` property per correction, resolved
against `gen_data`. Rejected — it duplicates knowledge the write already has, sits
apart from the application logic, and needs a drift test as a backstop. Returning
the modified subset folds the declaration into the write itself.
