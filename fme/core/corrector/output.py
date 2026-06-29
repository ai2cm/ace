"""Value objects describing the per-step result of a corrector.

These are pure additions in PR1 of the corrector-diagnostics work: they define
the *shape* of a corrector's diagnostic payload without yet being returned by
any corrector or carried on ``StepOutput`` (that wiring is issue 02). They live
in the corrector package so ``fme.core`` does not import from ``fme.ace``.
"""

import dataclasses
from collections.abc import Iterable

from fme.core.corrector.state import CorrectorState
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class CorrectorDiagnostics:
    """The diagnostic effect of a corrector on a single step.

    Parameters:
        delta: ``corrected[name] - network_output[name]`` for each variable the
            corrector declares it touched. The network's raw pre-correction
            value is recoverable as ``corrected[name] - delta[name]``. Defaults
            to empty, which is the value carried when no corrector ran or none
            modified anything.

    This is a value object rather than a bare mapping so the follow-on PR can add
    a per-field category label and metric fields without re-opening the corrector
    return type or the ``StepOutput`` field type.
    """

    delta: TensorMapping = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CorrectorOutput:
    """The full result of applying a corrector to one step's generated data.

    Replaces the corrector's historical ``(corrected, corrector_state)`` tuple
    return with a named object so new per-step payloads can be added without
    breaking every call site.

    Parameters:
        corrected: The adjusted generated data, semantically unchanged from the
            corrector's historical first tuple element.
        diagnostics: The corrector's declared correction deltas. Defaults to
            empty diagnostics.
        corrector_state: Per-sample state carried across step calls, passed
            through unchanged. Defaults to ``None``.
    """

    corrected: TensorDict
    diagnostics: CorrectorDiagnostics = dataclasses.field(
        default_factory=CorrectorDiagnostics
    )
    corrector_state: CorrectorState | None = None


def build_corrector_diagnostics(
    input_snapshot: TensorMapping,
    corrected: TensorMapping,
    touched_names: Iterable[str],
) -> CorrectorDiagnostics:
    """Build correction-delta diagnostics over an explicit set of touched names.

    Produces ``delta[name] = corrected[name] - input_snapshot[name]`` for each
    ``name`` in ``touched_names`` and nothing else, so the delta is an explicit
    declared diff over a known name set rather than something inferred by tensor
    object identity.

    For a field touched by more than one enabled correction, ``corrected`` is the
    corrector's exit value and ``input_snapshot`` is its entry value, so the
    stored delta is the cumulative net effect of all of them and
    ``input_snapshot == corrected - delta`` stays exact with no per-operation
    bookkeeping.

    The returned tensors are *not* detached from the autograd graph here;
    detaching is a step-boundary concern wired in issue 02.

    Args:
        input_snapshot: The corrector's input generated data, snapshotted at
            entry. Must contain every name in ``touched_names``.
        corrected: The corrector's output generated data. Must contain every
            name in ``touched_names``.
        touched_names: The variable names the corrector declares it writes.

    Returns:
        A ``CorrectorDiagnostics`` whose ``delta`` is keyed exactly by
        ``touched_names``.
    """
    delta = {name: corrected[name] - input_snapshot[name] for name in touched_names}
    return CorrectorDiagnostics(delta=delta)
