import dataclasses
from collections.abc import Iterable

from fme.core.corrector.state import CorrectorState
from fme.core.spatial_masking import SpatialMasking
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class CorrectorDiagnostics:
    """The diagnostic effect of a corrector on a single step.

    Only one pre-diagnosis value may be recorded per field, but an arbitrary number of
    deltas may be accumulated for any field, including one with a pre-diagnosis value.

    Parameters:
        pre_diagnosis_fields: The values of any fields the corrector diagnosed
            (i.e. replaced with a new value) before the diagnosis was applied.
            Empty when no corrector ran or none diagnosed anything.
        delta: ``corrected[name] - network_output[name]`` for each variable the
            corrector declares it touched. The network's raw pre-correction
            value is recoverable as ``corrected[name] - delta[name]``. Empty when
            no corrector ran or none modified anything.
    """

    pre_diagnosis_fields: TensorMapping = dataclasses.field(default_factory=dict)
    delta: TensorMapping = dataclasses.field(default_factory=dict)

    def apply_correction(
        self,
        pre_correction_state: TensorMapping,
        diagnosed: TensorMapping,
        deltas: TensorMapping,
    ) -> "CorrectorDiagnostics":
        """Return a new ``CorrectorDiagnostics`` with the given correction applied to
        ``pre_diagnosis_fields`` and the given diagnostics added to ``delta``; ``self``
        is not mutated.

        Args:
            pre_correction_state: The state before the correction represented by
                diagnosed and deltas was applied. This is used to record the
                pre-diagnosis fields for cases where that field is used in the course of
                determining the diagnosed field. The pre-diagnosed field in this case is
                not meant to have the same physical meaning as the diagnosed field.
            diagnosed: Fields replaced with diagnosed values.
            deltas: Corrective deltas to apply.
        """
        already_diagnosed = set(diagnosed.keys()).intersection(
            self.pre_diagnosis_fields.keys()
        )
        if already_diagnosed:
            raise ValueError(
                f"diagnosed fields {already_diagnosed} cannot be diagnosed again; "
                "they were already diagnosed in this step"
            )
        new_pre_diagnosis_fields = {**self.pre_diagnosis_fields}
        for name in diagnosed.keys():
            new_pre_diagnosis_fields[name] = pre_correction_state[name]
        new_deltas = {**self.delta}
        for name, value in deltas.items():
            if name in new_deltas:
                new_deltas[name] = new_deltas[name] + value
            else:
                new_deltas[name] = value
        return CorrectorDiagnostics(
            pre_diagnosis_fields=new_pre_diagnosis_fields,
            delta=new_deltas,
        )

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
        return CorrectorDiagnostics(
            pre_diagnosis_fields=self.pre_diagnosis_fields,
            delta=masking(self.delta),
        )


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

    def apply_correction(
        self, diagnosed: TensorMapping, deltas: TensorMapping
    ) -> "CorrectorOutput":
        """Return a new ``CorrectorOutput`` with the given correction applied to
        ``corrected`` and the given diagnostics added to ``diagnostics``; ``self``
        is not mutated.

        Args:
            diagnosed: Fields replaced with diagnosed values.
            deltas: Corrective deltas to apply.
        """
        new_diagnostics = self.diagnostics.apply_correction(diagnosed, deltas)
        new_corrected = {**self.corrected, **diagnosed}
        for name, value in deltas.items():
            if name not in new_corrected:
                raise ValueError(f"delta supplied for unknown variable {name}")
            new_corrected[name] = new_corrected[name] + value

        return CorrectorOutput(
            corrected=new_corrected,
            diagnostics=new_diagnostics,
            corrector_state=self.corrector_state,
        )


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
