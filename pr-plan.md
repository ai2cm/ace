# Rename Corrector to ModuleOutputMods and fold ocean SST and prescribed prognostics into the op pipeline

The "Corrector" concept becomes `ModuleOutputMods`: an ordered pipeline of
user-configured post-step operations on the network's outputs. On the
`SingleModuleStep` path, the ocean SST handling (all of `Ocean`) and the
`prescribed_prognostic_names` overwrite become ops in that pipeline, so the
flat per-field `delta = corrected - raw` covers every post-network write.
`raw` is the network output after residual addition and
`global_mean_removal.inverse_transform`; residual prediction and
global-mean removal stay outside the pipeline. No individual op's math
changes. `fme/core/step/secondary_module.py` and `fme/coupled` keep their
current behavior (mechanical renames only).

---

## Rename map (repo-wide, mechanical)

`fme/core/corrector/` → `fme/core/module_output_mods/` (`git mv`, module
basenames unchanged). Symbols:

| Old | New |
| --- | --- |
| `CorrectorABC` | `ModuleOutputModsABC` |
| `CorrectionSequence` | `ModuleOutputMods` |
| `CorrectorConfigABC` | `ModuleOutputModsConfig` (still abstract; `Config` names the built type, matching `OceanConfig` → `Ocean`) |
| `Correction` (protocol) | `OutputMod` |
| `CorrectorOutput` | `OutputModsOutput` |
| `CorrectorDiagnostics` | `OutputModsDiagnostics` |
| `CorrectorState` | `OutputModsState` |
| `EpochScheduledCorrector` | `EpochScheduledMods` |
| `CorrectorSelector` | `ModuleOutputModsSelector` |
| `AtmosphereCorrectorConfig` / `AtmosphereCorrector` | `AtmosphereModsConfig` / `AtmosphereMods` |
| `OceanCorrectorConfig` / `OceanCorrector` | `OceanModsConfig` / `OceanMods` |
| `IceCorrectorConfig` / `IceCorrector` | `IceModsConfig` / `IceMods` |
| `build_corrector_diagnostics` | `build_output_mods_diagnostics` |

Kept as-is (rationale in place):

- Registry type strings `"atmosphere_corrector"`, `"ocean_corrector"`,
  `"ice_corrector"` — stored in checkpoints and user YAML; renaming buys
  nothing functional.
- `CORRECTION_DELTAS = "correction_deltas"` and the
  `step_diagnostics/correction_deltas.nc` output name — on-disk data
  contract for downstream analysis; unchanged.
- `fme/core/ocean.py` (`Ocean`, `OceanConfig`, `SurfaceTemperature`,
  `Prescriber` wiring) — still consumed by `secondary_module.py`,
  `fme/coupled` (`prescribe_sst`), and the inference override path
  (`replace_ocean`); the fold happens at the `SingleModuleStep` call site,
  not by dissolving the class.

User-facing config keys (old spelling accepted with a deprecation warning
via the existing `remove_deprecated_keys` hooks; checkpoint loading for
inference never breaks):

- `corrector:` → `module_output_mods:` on `SingleModuleStepConfig`,
  `SecondaryModuleStepConfig`, and the legacy `SingleModuleStepperConfig`.
- `corrector_disabled_epochs:` → `disabled_epochs:` on
  `ModuleOutputModsConfig`.
- `ocean:` and `prescribed_prognostic_names:` keep their spelling and
  position (see decision below).

---

## `fme/core/module_output_mods/registry.py` (moved + modified)

```python
class OutputMod(Protocol):  # CHANGED — renamed from Correction; signature unchanged
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        mods_state: OutputModsState | None,
    ) -> tuple[TensorDict, OutputModsState | None]: ...

class ModuleOutputMods(ModuleOutputModsABC):  # CHANGED — renamed from CorrectionSequence
    def __init__(self, mods: list[OutputMod]): ...
    # __call__ unchanged: snapshot at entry, apply mods in order,
    # delta = corrected - snapshot over the union of modified names.

class PipelineAsMod:  # NEW — adapts a built ModuleOutputModsABC into one OutputMod
    """Calls the wrapped pipeline and returns only its modified subset
    ({name: corrected[name] for name in modified_names}, state), so a built
    (possibly EpochScheduledMods-wrapped) pipeline composes as an op inside
    an outer ModuleOutputMods. Epoch gating stays scoped to the wrapped
    pipeline: when disabled it returns an empty dict and later ops still run."""
    def __init__(self, wrapped: ModuleOutputModsABC): ...
    def __call__(self, input_data, gen_data, forcing_data, mods_state): ...

class EpochScheduledMods(ModuleOutputModsABC):  # CHANGED — rename only
    ...
```

### Critical detail — snapshot placement and delta exactness

- Today the snapshot lives at `CorrectionSequence.__call__` entry, and the
  ocean/prescribed writes happen after the corrector returns, outside the
  delta — guarded by an overlap check and by dropping prescribed names from
  the reported delta.
- After this PR the outer `ModuleOutputMods` pipeline *is* the whole
  post-network sequence, so its entry snapshot is exactly `raw` and
  `delta[name] = final_output[name] - raw[name]` is exact for every op by
  construction, overlaps included. The overlap guard and the
  prescribed-name delta-dropping in `step_with_adjustments` are deleted.

## `fme/core/module_output_mods/folded.py` (new)

```python
@dataclasses.dataclass
class SurfaceTemperatureMod:  # NEW — Ocean as an OutputMod
    """Wraps a built fme.core.ocean.Ocean; returns only
    {ocean.surface_temperature_name: value}. All three modes (prescribed,
    interpolate, slab) stay inside Ocean/SurfaceTemperature/Prescriber —
    zero math changes."""
    ocean: Ocean
    def __call__(self, input_data, gen_data, forcing_data, mods_state): ...

@dataclasses.dataclass
class PrescribedPrognosticMod:  # NEW — the prescribed_prognostic_names overwrite as an OutputMod
    """Returns {name: forcing_data[name]} for each configured name; raises
    ValueError naming any missing variable (same error text as today)."""
    names: list[str]
    def __call__(self, input_data, gen_data, forcing_data, mods_state): ...
```

## `fme/core/step/single_module.py` (modified)

```python
class SingleModuleStepConfig:
    module_output_mods: AtmosphereModsConfig | ModuleOutputModsSelector  # CHANGED — was `corrector`; old key mapped in _remove_deprecated_keys with a deprecation warning
    ocean: OceanConfig | None = None          # unchanged key — rewired internally into an op
    prescribed_prognostic_names: list[str]    # unchanged key — rewired internally into an op

class SingleModuleStep:
    def __init__(self, ...):  # CHANGED — composes the full pipeline once:
        # mods = [PipelineAsMod(built_user_mods)]
        # + [SurfaceTemperatureMod(ocean)] if config.ocean else []
        # + [PrescribedPrognosticMod(names)] if names else []
        # self._output_mods = ModuleOutputMods(mods)
        # train/set_epoch/get_state/load_state forward to the built user
        # pipeline (the only stateful part); the checkpoint state key
        # "corrector" becomes "module_output_mods", load_state accepts both.
        ...

def step_with_adjustments(
    ...,
    output_mods: ModuleOutputModsABC | None,  # CHANGED — was `corrector`
    ocean: Ocean | None = None,               # kept for the SecondaryModuleStep path only
    prescribed_prognostic_names: list[str] | None = None,  # kept for the SecondaryModuleStep path only
    ...
) -> StepOutput:
    # CHANGED — SingleModuleStep passes the composed pipeline and
    # ocean=None, prescribed_prognostic_names=None. The legacy ocean /
    # prescribed branches, still exercised by SecondaryModuleStep, keep
    # today's semantics (writes outside the delta, overlap guard, prescribed
    # names dropped from the delta) and are documented as legacy.
    ...
```

### Critical detail — op order

`[user-configured mods (epoch-gated as a block), SurfaceTemperatureMod,
PrescribedPrognosticMod]` — exactly main's execution order. The slab-ocean
op keeps reading the corrected surface energy fluxes, the prescribed value
keeps winning last, and epoch gating keeps disabling only the
user-configured mods, so every produced field value is identical to main.
The only behavior change is diagnostic: SST and prescribed names now appear
in the delta.

## `fme/core/ocean.py` (modified)

```python
class Ocean:
    def modified_value(self, input_data, gen_data, target_data) -> torch.Tensor:  # NEW — the prescribed/blended SST tensor alone, used by SurfaceTemperatureMod
        ...
    def __call__(self, ...) -> TensorDict:  # unchanged — full-dict form kept for SecondaryModuleStep
        ...
```

## Consumers of the config surface (modified, mechanical)

- `fme/core/step/step.py`, `fme/core/step/multi_call.py`,
  `fme/core/step/secondary_module.py`, `fme/ace/stepper/single_module.py`:
  rename fallout only (`module_output_mods` key, renamed types/state keys,
  old spellings accepted where states are parsed). `replace_ocean` /
  `get_ocean` / `replace_prescribed_prognostic_names` / `prescribe_sst`
  keep their names and behavior — they operate on the config surface, which
  is unchanged.
- `fme/core/step/step_diagnostics.py`,
  `fme/ace/aggregator/inference/step_diagnostics.py`,
  `fme/ace/inference/data_writer/main.py`: no interface change; docstring
  wording only. SST and prescribed fields flow into
  `correction_deltas.nc` and the step-diagnostics metrics automatically
  because they are now delta keys.

### Decision — `OceanConfig` stays the user-facing YAML surface

`ocean:` and `prescribed_prognostic_names:` keep their keys and are rewired
into ops inside `SingleModuleStep.__init__`. Rationale: every existing
training config and checkpoint keeps building; the inference override path
(`replace_ocean`, `replace_prescribed_prognostic_names`) and
`fme/coupled`'s `prescribe_sst` are untouched. Moving the modes under
`module_output_mods:` would break all of those for no functional gain; it
can happen later behind the same deprecation machinery if wanted.

### Decision — deltas for the folded ops enter the existing streams

SST (under `OceanConfig.surface_temperature_name`) and each prescribed
prognostic name gain per-field entries in the existing correction-delta
streams: `step_diagnostics/correction_deltas.nc`, the inference
step-diagnostics metrics, and anything else keyed off
`StepDiagnostics.delta`. Two stated behavior changes, both intended:

- Runs with an ocean or prescribed prognostics configured report delta
  variables they previously did not.
- A prescribed name the user mods also modified previously had its delta
  dropped; it is now reported as `prescribed_value - raw`.

---

## Tests

## `fme/core/module_output_mods/test_registry.py` (moved + modified)

```python
def test_pipeline_as_mod_returns_modified_subset():
    # GOAL: PipelineAsMod yields exactly the wrapped pipeline's modified
    # names/values and passes state through.
    ...

def test_epoch_gated_block_leaves_later_mods_applied():
    # GOAL: with the user block disabled (train mode, early epoch), a mod
    # appended after PipelineAsMod still applies and still gets a delta.
    ...
```

## `fme/core/module_output_mods/test_folded.py` (new)

```python
def test_surface_temperature_mod_matches_ocean_call():
    # GOAL: SurfaceTemperatureMod output equals Ocean.__call__'s SST field.
    # PARAMETERIZE: mode in {prescribed, interpolate, slab}.
    ...

def test_prescribed_prognostic_mod_overwrites_and_raises_on_missing():
    # GOAL: returns forcing values for configured names; ValueError names
    # the missing variable.
    ...
```

## `fme/core/step/test_step.py` (modified)

```python
def test_output_values_identical_to_pre_fold():
    # GOAL: regression — for a config with corrector + ocean + prescribed
    # names, every output field equals the value produced by applying the
    # three stages sequentially as on main (op math unchanged).
    # PARAMETERIZE: ocean mode in {prescribed, interpolate, slab}.
    ...

def test_delta_covers_folded_ops():
    # GOAL: delta keys include the SST name and prescribed names, and
    # delta[name] == output[name] - raw[name] exactly, raw snapshotted after
    # residual addition + global_mean_removal.inverse_transform.
    ...

def test_sst_overlap_with_user_mods_allowed_and_exact():
    # GOAL: a user mod writing the SST name no longer raises; the single
    # reported SST delta equals final - raw.
    ...

def test_deprecated_config_and_state_keys_load():
    # GOAL: `corrector:` / `corrector_disabled_epochs:` config keys and the
    # "corrector" step-state key load with a deprecation warning and build
    # the same step as the new spellings.
    ...
```

Existing suites (`fme/core/step/test_step.py`, `fme/ace` stepper/inference
tests, `fme/core/distributed/parallel_tests/test_step.py`) must stay green
under the renames with no expectation changes except the new delta keys.

---

## Open Questions

- Register alias type strings (`"atmosphere"`, `"ocean"`, `"ice"`) alongside
  the kept `"*_corrector"` registry names, or leave aliasing for later?
- Should `EpochScheduledMods`'s field rename (`corrector_disabled_epochs` →
  `disabled_epochs`) instead keep the old spelling to shrink the deprecation
  surface?
