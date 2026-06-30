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
            value is recoverable as ``corrected[name] - delta[name]``. Empty when
            no corrector ran or none modified anything.
    """

    delta: TensorMapping = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CorrectorOutput:
    """The full result of applying a corrector to one step's generated data.

    Parameters:
        corrected: The adjusted generated data.
        diagnostics: The corrector's declared correction deltas.
        corrector_state: Per-sample state carried across step calls.
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
    ``name`` in ``touched_names``.

    When more than one correction touches a field, ``corrected`` is the
    corrector's exit value and ``input_snapshot`` its entry value, so the stored
    delta is their cumulative net effect and ``input_snapshot == corrected -
    delta`` stays exact.

    The returned tensors are *not* detached from the autograd graph here.

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
