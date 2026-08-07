# Carry per-variable correction categories on `CorrectorDiagnostics`

Each `Correction` labels every variable it writes with the facts that hold for that write — what
the delta means, and whether the straight-through estimator was involved — and
`CorrectorDiagnostics` carries those labels per name alongside `delta`. Pure plumbing: no config
surface, no consumer, no change to any corrector's numerical behavior.

---

## `fme/core/corrector/output.py` (modified)

```python
CorrectionCategory = Literal["budget", "overwrite", "residual", "straight_through"]  # NEW
"""A fact about what a correction did to one variable it wrote.

These are independent facts, not mutually exclusive kinds: a variable carries the
set of them that hold, and each has exactly one consequence for a consumer.

The three below are defined by **what the delta is**, because that is what a
consumer reads. Exactly one holds per write:

- ``budget``: the delta is the network's own violation of a constraint computed
  over the network's own fields. It vanishes when the network already satisfies
  the constraint, and the corrected value is the network's value adjusted (offset
  or scaled) to satisfy it.
- ``overwrite``: the delta is the network's disagreement with a target computed
  from other fields, with the network's value for this variable playing no part
  in that target. It vanishes when the network already agrees. A derived
  reference, a reconstruction from other tendencies, a mask, or a bound.
- ``residual``: the delta is a term the network does not control. It does not
  depend on the network's value for this variable and does not vanish for any
  value the network could produce for it.

``straight_through`` is an orthogonal fact about the gradient path, and combines
with one of the above:

- ``straight_through``: this write went through ``replace_value_keep_gradient``,
  so the forward value is the corrected one while the gradient flows as if the
  correction had not happened. It names the gradient path rather than the
  operation: a bound enforced without the estimator is an ``overwrite`` and
  nothing more, because the reason to distinguish it -- that an explicit
  optimization override would collide with a gradient path already there -- does
  not apply.

``residual`` is the third distinct sense of that word within this call stack's
reach, and does not mean either of the other two:

- ``step_with_adjustments(residual_prediction=...)``: the *network* predicts a
  tendency against its own previous state.
- ``SurfaceEnergyFluxCorrectionConfig.method="residual_prediction"``: the
  network's ``hfds`` is a residual on top of an atmosphere-derived flux.
- ``CorrectionCategory`` ``residual``: this *delta* is an exogenous term, so it
  says nothing about the network's value.

The second implies the third and is currently its only instance; the first is
unrelated. Renaming the category (``exogenous``, ``imposed``) was rejected:
``residual`` is the accurate word for the relationship, and ``exogenous`` is
already used for forcing variables in ``fme/coupled``.

A ``Literal`` alias rather than an ``Enum``, because the repo's idiom for a closed
set of string choices is ``Literal`` (e.g. ``SurfaceEnergyFluxCorrectionConfig.method``).
"""


@dataclasses.dataclass
class CorrectionResult:  # NEW
    """What one ``Correction`` did to ``gen_data`` in a single call.

    Replaces the ``(modified, corrector_state)`` tuple so a correction can label
    each field it wrote. This mirrors the corrector level, where ``CorrectorOutput``
    replaced a tuple for the same reason.

    Parameters:
        modified: Only the fields this correction wrote.
        categories: The facts holding for each write, keyed exactly by ``modified``.
        corrector_state: Passed through unchanged by corrections without state.
    """

    modified: TensorDict
    categories: Mapping[str, frozenset[CorrectionCategory]]
    corrector_state: CorrectorState | None

    def __post_init__(self) -> None:
        """Raise ``ValueError`` for keys that differ from ``modified``'s, and for
        any write not carrying exactly one of ``budget``/``overwrite``/``residual``.

        A single write's delta cannot both be the network's own constraint
        violation and be independent of the network's value, so a set holding two
        of the three is an authoring error. Requiring one rather than allowing
        none also means a write can never be labeled ``{"straight_through"}``
        alone, which would say nothing about how a consumer should treat it.
        """
        ...

    @classmethod
    def uniform(
        cls,
        modified: TensorDict,
        categories: frozenset[CorrectionCategory],
        corrector_state: CorrectorState | None,
    ) -> "CorrectionResult":
        """Label every written field with the same set of facts.

        Most corrections write one kind of field, so this keeps the common case a
        single call while leaving the general constructor for corrections whose
        writes are not homogeneous.
        """
        ...


@dataclasses.dataclass
class CorrectorDiagnostics:
    delta: TensorMapping = dataclasses.field(default_factory=dict)
    categories: Mapping[str, frozenset[CorrectionCategory]] = dataclasses.field(  # NEW
        default_factory=dict
    )

    def __post_init__(self) -> None:  # NEW — ValueError unless categories.keys() == delta.keys()
        # Deliberately does NOT apply CorrectionResult's exactly-one-of-three rule.
        # These sets are unions across corrections, and a variable written by a
        # budget correction and then an overwrite legitimately carries both -- the
        # cumulative delta really is part constraint violation, part disagreement
        # with a replacement.
        ...

    def apply_output_masking(self, masking: SpatialMasking) -> "CorrectorDiagnostics":
        # CHANGED — copy `categories` through unchanged; masking preserves keys, so
        # the key-equality invariant holds.
        ...


def build_corrector_diagnostics(
    input_snapshot: TensorMapping,
    corrected: TensorMapping,
    # CHANGED — was `touched_names: Iterable[str]`. The category mapping's keys ARE the
    # touched names, so passing one mapping makes `categories.keys() == delta.keys()`
    # structural at the primary construction site instead of a second thing to keep in
    # sync. Rejected: a parallel `categories` argument alongside `touched_names`.
    touched: Mapping[str, frozenset[CorrectionCategory]],
) -> CorrectorDiagnostics:
    ...
```

`Iterable` becomes unused in this module and its import is dropped.

### Critical detail — the decision procedure

Two questions, applied to each write. This is the rule the table below encodes and the rule a
later author applies to a new correction:

```
Q1. Does the delta depend on the network's value for this variable?
      no  -> residual
      yes -> Q2

Q2. What makes the delta vanish?
      the network satisfying a constraint over its own fields  -> budget
      the network agreeing with a target built from other fields, in which its
        own value for this variable plays no part                -> overwrite
```

Two consequences worth stating, because both come up:

- **An exogenous constant inside a budget target does not make it `residual`.**
  `TotalEnergyBudgetCorrection`'s `unaccounted_heating` and `ConserveDryAir`'s IC-seeded reference
  mass both shift the target without the network controlling them, but the delta still moves with
  the network's own value, so Q1 is "yes" and Q2 gives `budget`. `residual` needs the delta to be
  *independent* of the network's value, which is a strictly stronger condition than "some term is
  exogenous".
- **Coverage weighting does not change the answer.** `SurfaceEnergyFluxCorrection(prescribed)`
  writes `net_flux·of + gen_hfds·(1 − of)`, so for `0 < of < 1` the network's value survives in
  proportion. Q1/Q2 still give `overwrite`: `delta = of·(net_flux − gen_hfds)` vanishes exactly
  when the network agrees with `net_flux`, and `of` only scales how much of that disagreement is
  charged. The same holds for a partial mask.

### Critical detail — a label describes the operation, not the grid cells

The category is a claim about the operation the code performed on a variable, evaluated once for
the variable. It is not per grid cell, and it is not conditional on the delta being nonzero.

This matters because two corrections write unconditionally:

- `_force_positive` (`fme/core/corrector/utils.py:37-43`) puts **every** name in `names` in the
  output whether the clamp bound or not, so a variable in `force_positive_names` carries
  `overwrite` even in a step where its delta is identically zero.
- `SurfaceEnergyFluxCorrection` writes `hfds` everywhere, including land cells where `of = 0` and
  the delta is exactly zero.

Consequence, and it is deliberate: via the union rule a variable that is both budget-corrected and
in `force_positive_names` permanently carries `{budget, overwrite}`, in every step, whether the
clamp ever binds. The consumer rule below is built to absorb that rather than being surprised by
it.

### Critical detail — where each fact lands, per write

| correction (`file`) | write | fact | Q1/Q2 |
|---|---|---|---|
| `ConserveDryAir` (`atmosphere.py:53`) | surface pressure | `budget` | delta = offset solving for the IC-seeded global dry-air mass; moves with the network's own pressure |
| `ZeroGlobalMeanMoistureAdvection` (`atmosphere.py:101`) | advection tendency | `budget` | delta = `−mean(gen_advection)`; vanishes when the network's own global mean is already zero |
| `MoistureBudgetCorrection` (`atmosphere.py:126`) | precipitation *or* evaporation rate | `budget` | multiplied by the global-mean **ratio** `new_gm/current_gm` (`atmosphere.py:586-598`), so the network's field survives and the delta vanishes when the global budget already closes |
| " | advection tendency, when `terms_to_modify.startswith("advection")` | `overwrite` | `new_advection = twp_total_tendency − (evaporation − precipitation)` (`atmosphere.py:599-607`) — the network's own advection tendency does not appear in the target at all |
| " | frozen precipitation rate, when `clip_frozen_precipitation` | `overwrite` | clipped to `min(frozen, PRATEsfc)`; the target is a bound built from another field |
| `TotalEnergyBudgetCorrection` (`atmosphere.py:182`) | every air-temperature level | `budget` | uniform `temperature_correction` closing the network's own energy budget |
| `ForcePositive` (`utils.py:47`) | each name in `names` | `overwrite` (+ `straight_through` iff `keep_gradient`) | target is the bound `0`; written unconditionally |
| `SeaIceFractionCorrection` (`ocean.py:139`) | the sea-ice fraction | `overwrite` (+ `straight_through` iff `keep_gradient`) | clamped and rebalanced against land fraction |
| " | each `zero_where_ice_free_names` | `overwrite` | written as `gen · (sif > 0)`, never straight-through |
| `SurfaceEnergyFluxCorrection` (`ocean.py:173`) | `hfds` / `hfds_total_area`, `method="prescribed"` | `overwrite` | see below |
| " | `hfds` / `hfds_total_area`, `method="residual_prediction"` | `residual` | see below |
| `OceanHeatContentCorrection` (`ocean.py:200`) | scaled ocean temperature | `budget` | scaled to close the network's own heat-content budget |
| `IceBudgetCorrectionConfig` (`ice.py:18`) | each `terms[0..2]` | `budget` | rebalanced from the network's own terms |
| " | each prognostic `key` | `overwrite` | `x_in[key] + timestep · sum(budgets)`; `gen_data[key]` is discarded |

`MoistureBudgetCorrection` is therefore non-uniform across **three** write sites, not two — the
2026-07 reading that all of `_force_conserve_moisture` is `budget` was wrong for the advection
branch, which is a reconstruction exactly parallel to the ice prognostics.

#### The two `hfds` methods are different kinds of correction

Writing out `delta = corrected − gen_hfds` for each branch of `_correct_hfds` (`ocean.py:373-408`):

```
residual_prediction:  out = net_flux·of + gen_hfds        =>  delta = net_flux·of
prescribed:           out = net_flux·of + gen_hfds·(1-of) =>  delta = of·(net_flux - gen_hfds)
```

- `residual_prediction` → **`residual`**. The delta does not contain `gen_hfds` anywhere, at any
  `of`: no `hfds` the network could produce would change it. Unconditional, so the label is exact.
- `prescribed` → **`overwrite`**. The delta vanishes exactly when the network agrees with
  `net_flux`, which is built from forcing and the input SST.

### Critical detail — why the labels are per name and set-valued

**Per name**, because a single call writes fields that fall in different categories:

- `IceBudgetCorrection` writes `budget` terms and `overwrite` prognostics in one call.
- `MoistureBudgetCorrection` writes a `budget` precipitation rate, an `overwrite` advection
  tendency, and an `overwrite` frozen-precipitation clip in one call.
- `SeaIceFractionCorrection` writes two `overwrite` fields that differ in `straight_through`, so
  even a correction uniform in the exclusive fact is not uniform in the set.

Returning the labels with the write keeps the existing rule — the returned dict's keys are the
modified names — and extends it: the same loop that records a write records what kind it was.
Rejected: a `category` attribute or property on `Correction`, which cannot express any of the three
cases; splitting these corrections in two, which restructures working code for a labelling reason
and fights the read-after-write dependency in the ice loop.

**Set-valued**, because `delta` is already the cumulative union when several corrections touch one
variable, so the label composes the same way as the thing it labels. `CorrectionSequence` unions;
it has no error path today (`registry.py:184-192`) and gains none. A single value would have needed
either an error path over a legal configuration, or a `mixed` sentinel that erases which facts
combined — exactly where a consumer most needs to know.

### Critical detail — how the labels are meant to be read

Not part of this PR, but the reason there are four. Each fact carries one consequence, and the
intended consumer (corrector-aware training) composes them by membership:

| fact | consequence for the consumer |
|---|---|
| `budget` | its presence is the reason to train against this variable |
| `overwrite` | not by itself a reason to train against it — and **not** a veto over `budget` |
| `residual` | a veto: never train against it by default, and an explicit override is incoherent, not merely off — the delta does not depend on the network's value, and the value is a residual against a derived term rather than an estimate of the reference |
| `straight_through` | an explicit override collides with the gradient path already there — guard it |

So the default is **`budget ∈ set and residual ∉ set`**, not "the set is exactly `{budget}`". The
distinction is load-bearing, because mixed sets are the norm rather than the exception. Working
`_build_full_atmosphere_corrector` (`test_atmosphere.py:704-714`) through the table — the
configuration the atmosphere delta test treats as the flagship, and one whose mixed advection set is
not avoidable, since `AtmosphereCorrectorConfig`'s docstring (`atmosphere.py:268-270`) requires
`zero_global_mean_moisture_advection=True` whenever `moisture_budget_correction` is set (documented
only — there is no `__post_init__` enforcing it):

| variable | written by | set |
|---|---|---|
| `PRATEsfc` | `ForcePositive`, then the moisture-budget precipitation ratio | `{budget, overwrite}` |
| advection tendency | `ZeroGlobalMeanMoistureAdvection`, then the moisture-budget reconstruction | `{budget, overwrite}` |
| `total_frozen_precipitation_rate` | the frozen-precipitation clip | `{overwrite}` |
| surface pressure | `ConserveDryAir` | `{budget}` |
| air-temperature levels | `TotalEnergyBudgetCorrection` | `{budget}` |

Under "exactly `{budget}`" the two most interesting variables in the standard atmosphere
configuration would both be excluded from corrector-aware training — which is why that rule is
wrong and the membership rule is stated here rather than being left to the consumer PR to
discover. Overriding an `overwrite` is legitimate and has been observed to help; that is the
distinction `residual` exists to keep separate.

## `fme/core/corrector/registry.py` (modified)

```python
class Correction(Protocol):
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> CorrectionResult:  # CHANGED — was tuple[TensorDict, CorrectorState | None]
        """
        Returns:
            A ``CorrectionResult`` carrying only the fields this correction
            modified, the facts holding for each write, and the corrector state.
        """
        ...


class CorrectionSequence(CorrectorABC):
    def __call__(self, ...) -> CorrectorOutput:  # CHANGED
        # Unions each result's per-name categories into the accumulator alongside the
        # existing modified-name set, then feeds the accumulated mapping to
        # `build_corrector_diagnostics` in place of the bare name set.
        ...
```

`EpochScheduledCorrector`'s disabled path is unchanged: it returns the default
`CorrectorDiagnostics()`, whose empty `delta` and empty `categories` are trivially key-consistent.

## `fme/core/corrector/utils.py` (modified)

```python
@dataclasses.dataclass
class ForcePositive:
    def __call__(self, ...) -> CorrectionResult:  # CHANGED
        # {"overwrite"}, plus "straight_through" when self.keep_gradient
        ...
```

## `fme/core/corrector/atmosphere.py` (modified)

```python
# Each returns CorrectionResult.uniform(corrected, frozenset({"budget"}), state):
class ConserveDryAir:
    def __call__(self, ...) -> CorrectionResult: ...              # CHANGED

class ZeroGlobalMeanMoistureAdvection:
    def __call__(self, ...) -> CorrectionResult: ...              # CHANGED

class TotalEnergyBudgetCorrection:
    def __call__(self, ...) -> CorrectionResult: ...              # CHANGED


class MoistureBudgetCorrection:
    def __call__(self, ...) -> CorrectionResult:  # CHANGED — never uniform
        # precipitation / evaporation rate scaled by the global-mean ratio -> {"budget"}
        # advection tendency recomputed from the other terms                -> {"overwrite"}
        # frozen precipitation clipped to the total rate                    -> {"overwrite"}
        ...
```

`_force_conserve_moisture` returns `gen.modified_data` — a single dict whose keys depend on
`terms_to_modify` — so `MoistureBudgetCorrection` builds its category mapping from
`terms_to_modify` and `clip_frozen_precipitation` rather than from the returned keys, and the
end-to-end test below is what ties the two together.

## `fme/core/corrector/ocean.py` (modified)

```python
@dataclasses.dataclass
class SeaIceFractionCorrection:
    def __call__(self, ...) -> CorrectionResult:  # CHANGED — not uniform
        # sea_ice_fraction (clamped and rebalanced): {"overwrite"}, plus
        #   "straight_through" when self.keep_gradient
        # zero_where_ice_free_names (written as gen * (sif > 0), never
        #   straight-through): {"overwrite"}
        ...


@dataclasses.dataclass
class SurfaceEnergyFluxCorrection:
    def __call__(self, ...) -> CorrectionResult:  # CHANGED — depends on method
        # "prescribed" -> {"overwrite"}; "residual_prediction" -> {"residual"}
        # keyed by the name actually written ("hfds" or "hfds_total_area")
        ...


@dataclasses.dataclass
class OceanHeatContentCorrection:
    def __call__(self, ...) -> CorrectionResult: ...  # CHANGED — uniform {"budget"}
```

## `fme/core/corrector/ice.py` (modified)

```python
@dataclasses.dataclass  # existing decorator, shown because the signature below changes
class IceBudgetCorrectionConfig:
    def __call__(  # CHANGED — also return the per-name categories
        self, gen_data: TensorMapping, input_data: TensorMapping, timestep: float
    ) -> tuple[TensorDict, dict[str, frozenset[CorrectionCategory]]]:
        # The existing write-recording loop (`ice.py:194-196`) labels as it records:
        #   terms[0..2] -> {"budget"}    (rebalanced from the network's own terms)
        #   key         -> {"overwrite"} (x_in[key] + timestep * sum(budgets);
        #                                 gen_data[key] is discarded)
        ...


@dataclasses.dataclass
class IceBudgetCorrection:
    def __call__(self, ...) -> CorrectionResult: ...  # CHANGED — not uniform
```

The labels are built in the loop that already records the writes (`modified[name] = work[name]`),
so they cannot drift from the write they describe. One wrinkle that loop already has: it iterates
`processing_order`, and a name serving as a prognostic `key` for one entry and as a `terms[i]` of
another would have its `modified` entry silently overwritten. The category mapping therefore
**unions** per name within the call rather than assigning, so such a name arrives at
`CorrectionResult.__post_init__` carrying `{budget, overwrite}` and is rejected by name. That
configuration is nonsense (a field cannot be both a state variable and a tendency of another), and
it currently fails silently; turning it into a `ValueError` is a small side benefit, not a behavior
change to any working config.

## `fme/core/step/single_module.py` (modified)

```python
def step_with_adjustments(...) -> StepOutput:  # CHANGED — carry `categories` at both seams
    ...
    # 1. The detach seam (`single_module.py:677`): rebuild with detached delta tensors
    #    AND the category mapping copied through, instead of dropping it by
    #    reconstructing from `delta` alone.
    diagnostics = CorrectorDiagnostics(
        delta={k: v.detach() for k, v in result.diagnostics.delta.items()},
        categories=result.diagnostics.categories,
    )
    ...
    # 2. The prescribed-prognostic drop (`single_module.py:722`): filter `categories`
    #    by the same predicate as `delta`, or `__post_init__` rejects the result.
```

Both are inline reconstructions today; they stay inline here. The `detach()` helper that replaces
the first one arrives with the training feature that needs attach/detach control. The third site,
`single_module.py:669`, is the no-corrector default and needs no change.

Cost: `CorrectorDiagnostics.__post_init__` now runs on each of those constructions, i.e. up to
three key-set comparisons per step. Negligible against a network forward pass, and stated here so
it is not re-derived in review.

### Scope boundary — the labels stop at `StepOutput`

`StepOutput.stack_diagnostics` (`fme/core/step/output.py:31-69`) builds a `StepDiagnostics` with
`delta` only, and is left alone: `StepDiagnostics` is the inference/serialization surface (it feeds
`step_diagnostics/correction_deltas.nc` and the inference metrics), while the categories' consumer
is a training-time loss. Threading them further would add an unused field to a checkpointed-adjacent
container. `fme/coupled/stepper.py:1301,1312` builds `StepOutput` with no corrector diagnostics at
all and is likewise untouched.

So the labels are reachable from `StepOutput.corrector_diagnostics` and nowhere else after this PR.
That is the seam the consumer reads from.

### Docstrings this invalidates

Prose that asserts `delta` is the only thing carried, or that a correction returns a tuple:

- `CorrectorDiagnostics` class docstring (`corrector/output.py:11-18`).
- `CorrectorOutput.modified_names` (`corrector/output.py:58-65`) — "exactly the keys of the
  diagnostics `delta`" is now also exactly the keys of `categories`.
- `build_corrector_diagnostics` (`corrector/output.py:73-89`) — documents `touched_names`.
- `Correction` protocol (`registry.py:70-101`) — describes the return as a `TensorDict` tuple.
- `StepOutput.corrector_diagnostics` (`step/output.py:20-21`).
- The `Returns:` block of every `Correction.__call__` being changed: `atmosphere.py:72-76` and its
  siblings, `ocean.py:160-167`, `ocean.py:185-189`, `utils.py`, `ice.py`.

### Typing note

`frozenset[CorrectionCategory]` with `CorrectionCategory` a `Literal` only infers correctly where
the expected type is in context. The four corrections that build their sets conditionally
(`ForcePositive`, `SeaIceFractionCorrection`, `MoistureBudgetCorrection`, the ice loop) get
`set[str]`/`frozenset[str]` from inference and need explicit annotations —
`categories: dict[str, frozenset[CorrectionCategory]] = {}` and a `frozenset[CorrectionCategory]`
annotation on any accumulator — or mypy fails in pre-commit. Called out because it is otherwise
discovered as churn mid-implementation.

---

## Tests

## `fme/core/corrector/test_output.py` (modified)

```python
# Existing `build_corrector_diagnostics` tests pass `{"a": frozenset({"budget"})}` in
# place of `["a"]`. The existing `test_apply_output_masking_masks_delta_and_returns_new_object`
# (test_output.py:85) constructs `CorrectorDiagnostics(delta={"a": ...})` and gains a
# matching `categories` argument, or __post_init__ rejects it.

def test_corrector_diagnostics_defaults_to_empty_categories():
    # GOAL: the no-argument default is an empty category mapping alongside the empty
    # delta, so the "no corrector ran" path needs no special casing.

def test_value_object_rejects_category_key_mismatch():
    # GOAL: neither CorrectorDiagnostics nor CorrectionResult can be built with
    # categories whose keys differ from the tensors they label; ValueError, and the
    # message names both key sets.
    # PARAMETERIZE: (type, missing key | extra key).

def test_correction_result_rejects_incompatible_categories():
    # GOAL: a single write must carry exactly one of budget/overwrite/residual.
    # PARAMETERIZE: {budget, overwrite}, {budget, residual}, {overwrite, residual},
    # {straight_through} alone, and the empty set — each raises ValueError naming the
    # offending variable and the set it was given.

def test_correction_result_accepts_straight_through_with_one_value_category():
    # GOAL: the orthogonal fact composes — {overwrite, straight_through} is valid, and
    # so is a bare {overwrite}.

def test_correction_result_uniform_labels_every_written_field():
    # GOAL: `uniform` produces categories keyed exactly by `modified`, all the same set,
    # and an empty mapping for an empty write.

def test_corrector_diagnostics_accepts_unioned_categories():
    # GOAL: the union rule is deliberately looser than the per-write rule —
    # {budget, overwrite} on one name is valid at the diagnostics level, because two
    # corrections legitimately produce it.

def test_build_corrector_diagnostics_assigns_categories():
    # GOAL: delta and categories are both keyed exactly by the passed mapping's keys,
    # and each name carries the set it was passed, including a name carrying two facts.

def test_apply_output_masking_preserves_categories():
    # GOAL: masking changes delta values only; the category mapping survives unchanged
    # under both a NaN-fill masker and NullSpatialMasking.
```

## `fme/core/step/test_output.py` (modified)

```python
# `_step_output` (test_output.py:9-18) builds CorrectorDiagnostics(delta={"a": ...}) and
# must pass a matching `categories`, or every test in the module fails at construction.
# No new tests: stack_diagnostics is deliberately out of scope (see the scope boundary
# above), so there is no category behavior here to pin.
```

## `fme/core/corrector/test_registry.py` (modified)

```python
# `ConstantOffsetCorrection` (test_registry.py:136) returns `CorrectionResult.uniform(...)`
# and gains a `categories` constructor argument defaulting to `frozenset({"budget"})`, so
# multi-correction cases are expressible. `fme/ace/stepper/test_single_module.py:68`
# imports it (used at :1525, :2945, :2992) and is unaffected by the default.

def test_correction_sequence_assigns_categories_per_modified_name():
    # GOAL: extend `test_correction_sequence_accumulates_modified_keys` — each modified
    # name in the diagnostics carries the set its correction returned, and untouched
    # names appear in neither mapping.

def test_correction_sequence_unions_categories_for_repeated_key():
    # GOAL: extend `test_correction_sequence_unions_repeated_modified_key` (which already
    # asserts the delta case succeeds and yields the net change). Two corrections with
    # different categories writing one field yield the union on that name; two with the
    # same category yield that one set. Nothing raises — there is no error path here and
    # this test is what keeps one from being added.

def test_epoch_scheduled_corrector_disabled_returns_empty_categories():
    # GOAL: extend the existing disabled-path test — the passthrough diagnostics have an
    # empty category mapping as well as an empty delta.
```

## `fme/core/corrector/test_utils.py` (modified)

```python
def test_force_positive_straight_through_category_tracks_keep_gradient():
    # GOAL: pin the decision that `straight_through` names the gradient path, not the
    # operation — the same correction returns {"overwrite", "straight_through"} and
    # {"overwrite"} with it off, while the corrected values are identical.
    # PARAMETERIZE: keep_gradient ∈ {True, False}.

def test_force_positive_labels_names_it_did_not_clamp():
    # GOAL: pin "the label describes the operation, not the cells" — an all-positive
    # input still yields {"overwrite"} on every name in `names` with an all-zero delta.
```

## `fme/core/corrector/test_ocean.py` (modified)

```python
def test_surface_energy_flux_category_and_delta_dependence():
    # GOAL: tie the label to the behavior that justifies it rather than asserting a
    # string. Run each method twice on inputs differing only in gen_data's hfds: under
    # "residual_prediction" the delta is bit-identical (nothing the network does moves
    # it) and the category is {"residual"}; under "prescribed" the delta differs and the
    # category is {"overwrite"}. Reuses `_make_atmos_forcing_data` and the setups in
    # `test_surface_energy_flux_correction_resid` (:286) and
    # `test_surface_energy_flux_correction_prescribed` (:328).
    # REQUIRES: at least one cell with ocean_fraction > 0. Both deltas are identically
    # zero where ocean_fraction == 0, so an all-land input would pass the
    # "residual_prediction" half vacuously and fail the "prescribed" half. Assert
    # `(1 - land_fraction - sea_ice_fraction).max() > 0` in the setup so this cannot
    # rot when the fixture changes.
    # PARAMETERIZE: method ∈ {"residual_prediction", "prescribed"}.

def test_sea_ice_fraction_categories_split_by_field():
    # GOAL: one call labels the fraction and the zeroed fields differently — the fraction
    # carries "straight_through" when keep_gradient is on, the zeroed names never do,
    # and both carry "overwrite". Builds on the existing zero_where_ice_free_names tests.
    # PARAMETERIZE: keep_gradient ∈ {True, False}.
```

## `fme/core/corrector/test_atmosphere.py` (modified)

```python
def test_moisture_budget_categories_split_by_term():
    # GOAL: the branch the old taxonomy got wrong. For terms_to_modify
    # "advection_and_precipitation" the precipitation rate is {"budget"} while the
    # advection tendency is {"overwrite"}, and with clip_frozen_precipitation the frozen
    # rate is {"overwrite"} — three write sites, two facts, one call.
    # PARAMETERIZE: terms_to_modify over every accepted value, so the precipitation-only
    # and evaporation-only branches are pinned as {"budget"} with no advection write.
```

## `fme/core/corrector/test_atmosphere.py`, `test_ocean.py`, `test_ice.py` (modified)

The end-to-end guard that every config-reachable correction is labeled, one per corrector family.
Extend the existing tests rather than adding parallel ones: with every correction option enabled,
assert the built corrector's category mapping labels each modified name as the table above says.

- `test_atmosphere_corrector_delta_matches_modified_returns` (`test_atmosphere.py:752`) already
  runs `_build_full_atmosphere_corrector`, which enables every field-modifying option and is held
  that way by the `test_atmosphere_corrector_config_fields_are_exercised` staleness guard. Assert
  the full expected mapping, including the two `{budget, overwrite}` unions worked out above —
  which makes this test the thing that would catch the flagship-config consequence changing.
- `test_ocean_corrector_delta_matches_modified_returns` (`test_ocean.py:499`) enables only
  `force_positive_names` and `sea_ice_fraction_correction`, so on its own it gives
  `SurfaceEnergyFluxCorrection` and `OceanHeatContentCorrection` **no** end-to-end coverage. It
  must also enable `surface_energy_flux_correction` and `ocean_heat_content_correction`, with the
  forcing from `_make_atmos_forcing_data` and a depth coordinate, so all four ocean corrections are
  labeled in one call. (The ocean config's own staleness guard covers the fields but not this test's
  coverage of them.)
- `test_ice_budget_correction_returns_only_modified` (`test_ice.py:68`) is the ice equivalent —
  there is no `test_ice_corrector_delta_matches_modified_returns`. Assert the per-name mapping
  there.

```python
def test_ice_budget_categories_split_terms_from_prognostics():
    # GOAL: the multi-prognostic case, beyond the single-prognostic end-to-end test above
    # — each reconstructed prognostic is {"overwrite"} and each of its three budget terms
    # is {"budget"}, for every entry in corrected_variables. Builds on
    # `_build_ice_corrector` / `_ice_test_data`.
```

## `fme/core/step/test_step.py` (modified)

```python
# Extend the existing corrector-diagnostics tests around the step seam (:767, :839, :867).

def test_step_preserves_corrector_categories_through_detach():
    # GOAL: the StepOutput's category mapping matches the corrector's and its keys still
    # equal the (detached) delta's keys.

def test_step_drops_prescribed_name_from_categories():
    # GOAL: reuse the existing prescribed-prognostic-overwrite scenario
    # (`test_step_with_prescribed_prognostic_overwrites_output`, :708, and :843) — a
    # prescribed name dropped from delta is dropped from the categories too
    # (construction would otherwise raise), and a non-prescribed corrected name keeps
    # its labels.
```

---

## Open Questions

- **Should `overwrite` be a veto rather than a soft signal?** The membership rule above
  (`budget ∈ set and residual ∉ set`) trains against `PRATEsfc` and the moisture advection tendency
  in the standard atmosphere configuration, on the reasoning that a clamp binding in a handful of
  cells should not disqualify a variable whose delta is dominated by a real budget violation. The
  stricter "exactly `{budget}`" rule excludes both. This PR encodes neither — it only carries the
  sets — but the consumer PR has to pick one, and the answer decides whether the two most-corrected
  atmosphere variables get a corrector-aware loss term at all. Reviewers with an opinion on the
  physics should say so here rather than in the consumer PR.
- **Is `straight_through` in the right vocabulary?** It is the one fact recorded from a config flag
  (`keep_gradient`) rather than from the operation, so `ForcePositive(keep_gradient=False)` and
  `ForcePositive(keep_gradient=True)` label the same operation differently. That is exactly what
  its consumer consequence needs — with the estimator off there is no gradient path for an override
  to collide with — but it does mean the four values are not the same kind of statement. The
  alternative is to leave the gradient path out of the taxonomy and have the consumer read
  `keep_gradient` off the corrector config directly, which keeps the taxonomy homogeneous at the
  cost of making the consumer reach into config it otherwise does not touch.
