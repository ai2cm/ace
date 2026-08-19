import dataclasses

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
            corrector applied an additive correction to. The network's raw
            pre-correction value is recoverable as
            ``corrected[name] - delta[name]``. For diagnosed (wholesale-replaced)
            fields, the pre-correction value is in
            ``pre_diagnosis_fields[name]``, not ``corrected - delta``. Empty
            when no corrector ran or none modified anything.
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

        Composition rules:

        - A field that already carries a delta cannot be diagnosed (the
          additive delta would become meaningless after a wholesale
          replacement).
        - A field *can* be diagnosed more than once; the later diagnosis
          supersedes the earlier one in ``corrected``, while
          ``pre_diagnosis_fields`` retains the seed-state value recorded by
          the first diagnosis.

        Args:
            pre_correction_state: The seed state (raw network output at the
                start of the correction fold).  ``pre_diagnosis_fields``
                records the value from this dict, so the stored
                pre-diagnosis value is always the network output before any
                correction ran.
            diagnosed: Fields replaced with diagnosed values.
            deltas: Corrective deltas to apply.
        """
        # Diagnosis after delta raises.
        delta_then_diagnosed = set(diagnosed.keys()) & set(self.delta.keys())
        if delta_then_diagnosed:
            raise ValueError(
                f"cannot diagnose fields {delta_then_diagnosed} that already "
                "carry a delta; diagnosis after delta is not allowed"
            )

        new_pre_diagnosis_fields = {**self.pre_diagnosis_fields}
        for name in diagnosed.keys():
            # Record the seed-state value only on the *first* diagnosis; a
            # later re-diagnosis supersedes the corrected value but the
            # pre-diagnosis snapshot stays at the seed.
            if name not in new_pre_diagnosis_fields:
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
            pre_diagnosis_fields=masking(self.pre_diagnosis_fields),
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

    # Internal: the seed state from the start of the correction fold,
    # captured on the first ``apply_correction`` call and propagated
    # thereafter. Stored by reference (corrections operate out-of-place, so
    # the original dict is never mutated).
    _seed_state: TensorMapping | None = dataclasses.field(
        default=None, repr=False, compare=False
    )

    @property
    def modified_names(self) -> set[str]:
        """The set of variable names modified by the corrector.

        A variable is considered modified if it has a delta (additive
        correction) or a pre-diagnosis entry (wholesale replacement), i.e.
        ``delta.keys() | pre_diagnosis_fields.keys()``.
        """
        return set(self.diagnostics.delta.keys()) | set(
            self.diagnostics.pre_diagnosis_fields.keys()
        )

    def apply_correction(
        self, diagnosed: TensorMapping, deltas: TensorMapping
    ) -> "CorrectorOutput":
        """Return a new ``CorrectorOutput`` with the given correction applied to
        ``corrected`` and the given diagnostics added to ``diagnostics``; ``self``
        is not mutated.

        Args:
            diagnosed: Fields replaced with diagnosed values.  Every key must
                already exist in ``corrected``.
            deltas: Corrective deltas to apply.  Every key must already exist
                in ``corrected``.
        """
        for name in diagnosed:
            if name not in self.corrected:
                raise ValueError(
                    f"diagnosed field '{name}' is not present in corrected data"
                )
        for name in deltas:
            if name not in self.corrected:
                raise ValueError(
                    f"delta field '{name}' is not present in corrected data"
                )

        seed = self._seed_state if self._seed_state is not None else self.corrected
        new_diagnostics = self.diagnostics.apply_correction(seed, diagnosed, deltas)
        new_corrected = {**self.corrected, **diagnosed}
        for name, value in deltas.items():
            new_corrected[name] = new_corrected[name] + value

        return CorrectorOutput(
            corrected=new_corrected,
            diagnostics=new_diagnostics,
            corrector_state=self.corrector_state,
            _seed_state=seed,
        )

    def with_state(self, corrector_state: CorrectorState | None) -> "CorrectorOutput":
        """Return a copy with a different ``corrector_state``; everything else
        (including ``_seed_state``) is shared, not copied.
        """
        return CorrectorOutput(
            corrected=self.corrected,
            diagnostics=self.diagnostics,
            corrector_state=corrector_state,
            _seed_state=self._seed_state,
        )
