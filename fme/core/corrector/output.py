import dataclasses
from collections.abc import Iterable, KeysView

from fme.core.corrector.state import CorrectorState
from fme.core.spatial_masking import SpatialMasking
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

    def apply_output_masking(self, masking: SpatialMasking) -> "CorrectorDiagnostics":
        """Return a new ``CorrectorDiagnostics`` with the output spatial masking
        applied to ``delta``; ``self`` is not mutated.

        This accepts a spatial masking specifically, not an arbitrary
        output-processing function. Masking a difference with an
        absolute-field masker is correct only because the output masker fills
        with NaN: off-mask the delta becomes NaN, matching
        ``masked_output - masked_snapshot`` (``NaN - NaN``). A finite fill
        (e.g. 0 or a mean) would inject a spurious offset and break the
        ``delta = output - snapshot`` invariant, and any value-transforming
        function would corrupt the delta outright.

        Args:
            masking: The stepper's output spatial masking (NaN-fill, or the
                no-op ``NullSpatialMasking``).
        """
        return CorrectorDiagnostics(delta=masking(self.delta))


@dataclasses.dataclass
class CorrectorOutput:
    """The full result of applying a corrector to one step's generated data.

    Parameters:
        corrected: The adjusted generated data.
        diagnostics: The corrector's diagnostic outputs.
        corrector_state: Per-sample state carried across step calls.
    """

    corrected: TensorDict
    diagnostics: CorrectorDiagnostics = dataclasses.field(
        default_factory=CorrectorDiagnostics
    )
    corrector_state: CorrectorState | None = None

    @property
    def modified_names(self) -> KeysView[str]:
        """Names of the variables the corrector modified this step.

        These are exactly the keys of the diagnostics ``delta``, i.e. the union
        of the fields returned by each applied correction.
        """
        return self.diagnostics.delta.keys()


def build_corrector_diagnostics(
    input_snapshot: TensorMapping,
    corrected: TensorMapping,
    touched_names: Iterable[str],
) -> CorrectorDiagnostics:
    """Build correction-delta diagnostics over an explicit set of touched names.

    Produces ``delta[name] = corrected[name] - input_snapshot[name]`` for each
    ``name`` in ``touched_names``.

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
