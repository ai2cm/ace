# Carry per-variable correction categories on `CorrectorDiagnostics`

Each `Correction` labels every variable it writes with the facts that hold for that write — what
happened to the network's value, and whether the straight-through estimator was involved — and
`CorrectorDiagnostics` carries those labels per name alongside `delta`. Pure plumbing: no config
surface, no consumer, no change to any corrector's numerical behavior.

---

## `fme/core/corrector/output.py` (modified)

```python
CorrectionCategory = Literal["budget", "overwrite", "residual", "straight_through"]  # NEW
"""A fact about what a correction did to one variable it wrote.

These are independent facts, not mutually exclusive kinds: a variable carries the
set of them that hold, and each has exactly one consequence for a consumer.

Exactly one of these three answers "what happened to the network's value":

- ``budget``: it was adjusted to satisfy a conservation or budget constraint. The
  corrected value still carries the network's value, and the delta is the
  network's violation of the constraint.
- ``overwrite``: it was replaced by a value built from other fields -- a derived
  reference, a reconstruction from tendencies, a mask, or a bound. The delta
  measures disagreement with that replacement, which is a proxy for the truth
  rather than the truth itself.
- ``residual``: a term the network does not control was added to it. The network's
  value is a residual against that term, so the delta does not depend on it.

``straight_through`` is an orthogonal fact about the gradient path, and combines
with one of the above:

- ``straight_through``: this write went through ``replace_value_keep_gradient``,
  so the forward value is the corrected one while the gradient flows as if the
  correction had not happened. It names the gradient path rather than the
  operation: a bound enforced without the estimator is an ``overwrite`` and
  nothing more, because the reason to distinguish it -- that an explicit
  optimization override would collide with a gradient path already there -- does
  not apply.

A ``Literal`` alias rather than an ``Enum``: the repo's idiom for a closed set of
string choices is ``Literal`` (e.g. ``SurfaceEnergyFluxCorrectionConfig.method``),
and correctors are constructed through ``dacite.Config(strict=True)`` with no
``cast`` list, so an ``Enum`` would need extra dacite configuration the moment a
category becomes config-facing.
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
        """Reject keys that differ from ``modified``'s, and any write not carrying
        exactly one of ``budget``/``overwrite``/``residual``.

        A single write cannot both preserve and replace the network's value, nor
        both depend and not depend on it, so a set holding two of the three is an
        authoring error. Requiring one rather than allowing none also means a
        write can never be labeled ``{"straight_through"}`` alone, which would say
        nothing
        about how a consumer should treat it.
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

    def __post_init__(self) -> None:  # NEW — enforce categories.keys() == delta.keys()
        # Deliberately does NOT apply CorrectionResult's exactly-one-of-three rule.
        # These sets are unions across corrections, and a variable written by a
        # budget correction and then an overwrite legitimately carries both -- the
        # cumulative delta really is part constraint violation, part replacement.
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

### Critical detail — why the labels are per name and set-valued

**Per name**, because two corrections write more than one kind of field in a single call:

- `IceBudgetCorrection` rebalances the network's budget terms (`budget`) and then reconstructs each
  prognostic as `x_in[key] + timestep * sum(budgets)` (`overwrite` — the network's own prediction
  for that prognostic never appears in the result).
- `SeaIceFractionCorrection` clamps and rebalances the fraction and zeroes ice-dependent fields
  where the corrected fraction is zero; only the former goes through the straight-through estimator.

Returning the labels with the write keeps the existing rule — the returned dict's keys are the
modified names — and extends it: the same loop that records a write records what kind it was.
Rejected: a `category` attribute or property on `Correction`, which cannot express either case;
splitting these corrections in two, which restructures working code for a labelling reason and
fights the read-after-write dependency in the ice loop.

**Set-valued**, because `delta` is already the cumulative union when several corrections touch one
variable, so the label composes the same way as the thing it labels. `CorrectionSequence` unions.
There is no conflict case and nothing to raise. A single value would have needed either an error
path over a legal configuration, or a `mixed` sentinel that erases which facts combined — exactly
where a consumer most needs to know.

### Critical detail — how the labels are meant to be read

Not part of this PR, but the reason there are four. Each fact carries one consequence, and the
intended consumer (corrector-aware training) composes them by membership:

| fact | consequence for the consumer |
|---|---|
| `budget` | train against this variable by default |
| `overwrite` | do not train against it by default |
| `straight_through` | an explicit override collides with the gradient path already there — guard it |
| `residual` | an explicit override is incoherent, not merely off: the delta does not depend on the network's value, and the value is a residual against a derived term rather than an estimate of the reference |

So the default is "train against it iff the set is exactly `{budget}`", and the two guards fire on
membership. Overriding an `overwrite` is legitimate and has been observed to help; that is the
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
    def __call__(self, ...) -> CorrectionResult:  # CHANGED — not uniform when clipping
        # _force_conserve_moisture adjusts its terms by a global-mean shift, so the
        # network's value survives in them   -> {"budget"}
        # _clip_frozen_precipitation replaces the frozen rate with the total where it
        # exceeds it, with no straight-through -> {"overwrite"} and nothing more
        ...
```

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
        ...


@dataclasses.dataclass
class OceanHeatContentCorrection:
    def __call__(self, ...) -> CorrectionResult: ...  # CHANGED — uniform {"budget"}
```

### Critical detail — the two `hfds` methods are different kinds of correction

Writing out `delta = corrected - gen_hfds` for each branch of `_correct_hfds`:

```
residual_prediction:  out = net_flux·of + gen_hfds        =>  delta = net_flux·of
prescribed:           out = net_flux·of + gen_hfds·(1-of) =>  delta = of·(net_flux - gen_hfds)
```

- `residual_prediction` → **`residual`**. The delta does not contain `gen_hfds`: the network's value
  is a residual on top of a derived term, and no `hfds` it could produce would change the delta.
- `prescribed` → **`overwrite`**. Over ocean (`of = 1`) the corrected value is `net_flux` outright —
  the network's `hfds` is discarded there — and the delta is its disagreement with that replacement.

## `fme/core/corrector/ice.py` (modified)

```python
class IceBudgetCorrectionConfig:
    def __call__(  # CHANGED — also return the per-name categories
        self, gen_data: TensorMapping, input_data: TensorMapping, timestep: float
    ) -> tuple[TensorDict, dict[str, frozenset[CorrectionCategory]]]:
        # The existing write-recording loop labels as it records:
        #   terms[0..2] -> {"budget"}    (rebalanced from the network's own terms)
        #   key         -> {"overwrite"} (x_in[key] + timestep * sum(budgets);
        #                                 gen_data[key] is discarded)
        ...


@dataclasses.dataclass
class IceBudgetCorrection:
    def __call__(self, ...) -> CorrectionResult: ...  # CHANGED — not uniform
```

The labels are built in the loop that already records the writes (`modified[name] = work[name]`), so
they cannot drift from the write they describe.

## `fme/core/step/single_module.py` (modified)

```python
def step_with_adjustments(...) -> StepOutput:  # CHANGED — carry `categories` at both seams
    ...
    # 1. The detach seam: rebuild with detached delta tensors AND the category mapping
    #    copied through, instead of dropping it by reconstructing from `delta` alone.
    diagnostics = CorrectorDiagnostics(
        delta={k: v.detach() for k, v in result.diagnostics.delta.items()},
        categories=result.diagnostics.categories,
    )
    ...
    # 2. The prescribed-prognostic drop: filter `categories` by the same predicate as
    #    `delta`, or `__post_init__` rejects the result.
```

Both are inline reconstructions today; they stay inline here. The `detach()` helper that replaces
the first one arrives with the training feature that needs attach/detach control.

---

## Tests

## `fme/core/corrector/test_output.py` (modified)

```python
# Existing `build_corrector_diagnostics` tests pass `{"a": frozenset({"budget"})}` in
# place of `["a"]`.

def test_corrector_diagnostics_defaults_to_empty_categories():
    # GOAL: the no-argument default is an empty category mapping alongside the empty
    # delta, so the "no corrector ran" path needs no special casing.

def test_value_object_rejects_category_key_mismatch():
    # GOAL: neither CorrectorDiagnostics nor CorrectionResult can be built with
    # categories whose keys differ from the tensors they label; the message names both
    # key sets.
    # PARAMETERIZE: (type, missing key | extra key).

def test_correction_result_rejects_incompatible_categories():
    # GOAL: a single write must carry exactly one of budget/overwrite/residual.
    # PARAMETERIZE: {budget, overwrite}, {budget, residual}, {overwrite, residual},
    # {straight_through} alone, and the empty set — each raises, naming the offending
    # variable and the set it was given.

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

## `fme/core/corrector/test_registry.py` (modified)

```python
# `ConstantOffsetCorrection` returns `CorrectionResult.uniform(...)` and gains a
# `categories` constructor argument defaulting to `frozenset({"budget"})`, so
# multi-correction cases are expressible. `test_single_module.py` imports it and is
# unaffected by the default.

def test_correction_sequence_assigns_categories_per_modified_name():
    # GOAL: extend the existing accumulate-modified-keys test — each modified name in the
    # diagnostics carries the set its correction returned, and untouched names appear in
    # neither mapping.

def test_correction_sequence_unions_categories_for_repeated_key():
    # GOAL: the case that used to raise. Two corrections with different categories
    # writing one field yield the union on that name, alongside today's cumulative
    # delta; two corrections with the same category yield that one set, not a duplicate.

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
```

## `fme/core/corrector/test_ocean.py` (modified)

```python
def test_surface_energy_flux_category_and_delta_dependence():
    # GOAL: tie the label to the behavior that justifies it rather than asserting a
    # string. Run each method twice on inputs differing only in gen_data's hfds: under
    # "residual_prediction" the delta is bit-identical (nothing the network does moves
    # it) and the category is {"residual"}; under "prescribed" the delta differs and the
    # category is {"overwrite"}. Reuses `_make_atmos_forcing_data` and the setups in the
    # existing per-method tests.
    # PARAMETERIZE: method ∈ {"residual_prediction", "prescribed"}.

def test_sea_ice_fraction_categories_split_by_field():
    # GOAL: one call labels the fraction and the zeroed fields differently — the fraction
    # carries "straight_through" when keep_gradient is on, the zeroed names never do,
    # and both carry "overwrite". Builds on the existing zero_where_ice_free_names tests.
    # PARAMETERIZE: keep_gradient ∈ {True, False}.
```

## `fme/core/corrector/test_ice.py` (modified)

```python
def test_ice_budget_categories_split_terms_from_prognostics():
    # GOAL: the other non-uniform case — each reconstructed prognostic is {"overwrite"}
    # and each of its three budget terms is {"budget"}, for every entry in
    # corrected_variables. Builds on `_build_ice_corrector` / `_ice_test_data`.
```

## `fme/core/corrector/test_atmosphere.py`, `test_ocean.py`, `test_ice.py` (modified)

```python
# Extend the existing `test_*_corrector_delta_matches_modified_returns` tests (and the
# ice equivalent) rather than adding parallel ones: with every correction option enabled,
# assert the built corrector's category mapping labels each modified name as the taxonomy
# says. This is the end-to-end guard that every config-reachable correction is labeled,
# and it pins the whole taxonomy across the three corrector families.
```

## `fme/core/step/test_step.py` (modified)

```python
# Extend the existing corrector-diagnostics tests around the step seam.

def test_step_preserves_corrector_categories_through_detach():
    # GOAL: the StepOutput's category mapping matches the corrector's and its keys still
    # equal the (detached) delta's keys.

def test_step_drops_prescribed_name_from_categories():
    # GOAL: reuse the existing prescribed-prognostic-overwrite scenario — a prescribed
    # name dropped from delta is dropped from the categories too (construction would
    # otherwise raise), and a non-prescribed corrected name keeps its labels.
```

---

## Open Questions

- `straight_through` is recorded per write from the correction's `keep_gradient` setting, so
  `ForcePositive(keep_gradient=False)` is labeled `{"overwrite"}` and nothing more. That
  matches the fact's only consequence — there is no straight-through path for an override to
  collide with — but it does mean this one label varies with a config flag while the other three
  are properties of the operation itself.
- `MoistureBudgetCorrection`'s frozen-precipitation clip is labeled `{"overwrite"}` on the same
  reasoning. It is a bound, but an ordinary one, so it sits with the replacements rather than with
  the straight-through writes.
