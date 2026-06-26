# Add `CorrectorOutput` with `CorrectorDiagnostics` containing `delta = corrected - network_output`

Adds two value objects to the corrector package — `CorrectorDiagnostics` (a single
`delta` mapping) and `CorrectorOutput` — plus a `build_corrector_diagnostics`
helper that computes `delta[name] = corrected[name] − input_snapshot[name]` over an
explicit declared name set. Pure addition: nothing returns or consumes these yet,
so behavior is unchanged.

---

## `fme/core/corrector/output.py` (new)

```python
import dataclasses

from fme.core.corrector.state import CorrectorState
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class CorrectorDiagnostics:
    """Per-step diagnostic payload describing a corrector's effect.

    delta[name] = corrected[name] − network_output[name] for each variable the
    corrector declares it touched. Empty when no corrector ran or none modified
    anything, so consumers need no None checks. The network's raw pre-correction
    value is recoverable as output − delta wherever the corrected output is known.

    Shaped as a value object (not a bare mapping) so later work can add a category
    label and metric fields without re-opening the corrector return type.
    """

    delta: TensorMapping = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CorrectorOutput:
    """Result of applying a corrector to a single step's generated data.

    corrected is the adjusted output (semantically identical to today's corrected
    gen_data). corrector_state is threaded through unchanged. diagnostics carries
    the per-step delta.
    """

    corrected: TensorDict
    diagnostics: CorrectorDiagnostics = dataclasses.field(
        default_factory=CorrectorDiagnostics
    )
    corrector_state: CorrectorState | None = None
```

## `fme/core/corrector/utils.py` (modified)

```python
from collections.abc import Iterable

from fme.core.corrector.output import CorrectorDiagnostics
from fme.core.typing_ import TensorMapping

# def force_positive(data, names): ...  # unchanged

def build_corrector_diagnostics(  # NEW
    input_snapshot: TensorMapping,
    corrected: TensorMapping,
    touched_names: Iterable[str],
) -> CorrectorDiagnostics:
    """Build the diagnostics for a corrector from an explicit touched-name set.

    delta[name] = corrected[name] − input_snapshot[name] for name in touched_names.
    This is a declared diff over a known name set — it replaces the role of the
    object-identity ``captured_before`` heuristic (removed when the corrector
    starts returning ``CorrectorOutput``). For a name modified by more than one
    enabled correction, ``delta`` is the cumulative ``corrected − input_snapshot``,
    so ``input_snapshot = corrected − delta`` stays exact with no per-operation
    bookkeeping.
    """
    delta = {
        name: corrected[name] - input_snapshot[name] for name in touched_names
    }
    return CorrectorDiagnostics(delta=delta)
```

---

## Tests

## `fme/core/corrector/test_output.py` (new)

```python
def test_corrector_diagnostics_delta_defaults_empty():
    # GOAL: a default-constructed CorrectorDiagnostics has an empty delta mapping,
    # so non-corrector / no-op steps need no None checks.
    ...

def test_corrector_output_defaults():
    # GOAL: CorrectorOutput(corrected=...) defaults to empty diagnostics and
    # corrector_state=None; corrected is carried through verbatim.
    ...
```

## `fme/core/corrector/test_utils.py` (new)

```python
# Hand-computable tensors so delta values are asserted exactly. Prior-art test
# style: fme/core/corrector/test_ocean.py (DEVICE = get_device(), small tensors).

def test_build_corrector_diagnostics_delta():
    # GOAL: delta contains exactly touched_names and delta[name] equals
    # corrected[name] − input_snapshot[name], asserted exactly. Names present in
    # the inputs but absent from touched_names do NOT appear in delta.
    ...

def test_build_corrector_diagnostics_cumulative():
    # GOAL: when corrected reflects two corrections applied to one field
    # (snapshot x → x + a + b), delta[name] == a + b, and
    # corrected − delta == input_snapshot exactly (recoverability).
    ...

def test_build_corrector_diagnostics_empty():
    # GOAL: empty touched_names → empty delta (the no-op / no-corrector case).
    ...
```
