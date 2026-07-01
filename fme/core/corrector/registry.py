import abc
import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol, Self, final

import dacite

from fme.core.corrector.output import CorrectorOutput, build_corrector_diagnostics
from fme.core.corrector.state import CorrectorState
from fme.core.dataset_info import DatasetInfo
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class CorrectorConfigABC(abc.ABC):
    """Base for corrector configs.

    Subclasses implement ``_get_corrector``. The ``corrector_disabled_epochs``
    option is handled here: ``get_corrector`` wraps the built corrector in an
    ``EpochScheduledCorrector`` when it is greater than zero.

    Parameters:
        corrector_disabled_epochs: Number of initial training epochs during
            which the corrector is not applied to train-mode steps. The
            corrector is always applied in eval mode (validation, inline
            inference and standalone inference).
    """

    corrector_disabled_epochs: int = dataclasses.field(default=0, kw_only=True)

    def __post_init__(self):
        if self.corrector_disabled_epochs < 0:
            raise ValueError(
                "corrector_disabled_epochs must be non-negative, got "
                f"{self.corrector_disabled_epochs}"
            )

    @classmethod
    @final
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(cls, state, config=dacite.Config(strict=True))

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        """
        This method is used to remove or transform any deprecated keys from the
        state dict before loading it into a CorrectorConfigABC instance. It is
        optional to implement this method on subclasses.
        """
        return dict(state)

    @final
    def get_corrector(self, dataset_info: DatasetInfo) -> "CorrectorABC":
        corrector = self._get_corrector(dataset_info)
        if self.corrector_disabled_epochs == 0:
            return corrector
        return EpochScheduledCorrector(
            wrapped=corrector,
            disabled_epochs=self.corrector_disabled_epochs,
        )

    @abc.abstractmethod
    def _get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "CorrectorABC": ...


class Correction(Protocol):
    """A single correction applied to ``gen_data`` by a corrector.

    Each correction is a self-contained callable object that bundles its own
    parameters and operators (e.g. an area-weighted-mean operator or a vertical
    coordinate) and applies one conservation/positivity step. A corrector holds
    an ordered sequence of these and simply applies them in turn, so it does not
    need to read any config fields itself. The signature mirrors
    ``CorrectorABC.__call__`` so corrections compose: a correction that does not
    maintain state passes ``corrector_state`` through unchanged.

    Each ``__call__`` returns a ``TensorDict`` containing **only the fields this
    correction modified** -- not the full ``gen_data``. The caller dict-updates
    ``gen_data`` with the returned subset and takes its keys as the set of
    variables the correction is responsible for writing. Because the returned
    dict is exactly what gets applied, the returned keys are the single source of
    truth for what changed and cannot drift from the write. This contract is
    documented here and reiterated in a ``Returns`` annotation on each concrete
    ``__call__``; that inline documentation is the primary guard against drift.
    """

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple ``(modified, corrector_state)`` where ``modified`` contains
            only the fields modified by this correction.
        """
        ...


class CorrectorABC(abc.ABC):
    def train(self, mode: bool = True) -> "CorrectorABC":
        """Set the corrector to training or evaluation mode.

        Default implementation is a no-op: a stateless corrector behaves
        identically in both modes. Override to vary behavior by mode.
        """
        return self

    @final
    def eval(self) -> "CorrectorABC":
        """Set the corrector to evaluation mode."""
        return self.train(False)

    def set_epoch(self, epoch: int) -> None:
        """Called by the stepper at the start of each training epoch.

        Default implementation is a no-op.
        """

    def get_state(self) -> dict[str, Any]:
        """Return corrector checkpoint state.

        Correctors without checkpointed state return an empty dict, which is
        the default implementation.
        """
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        """Load corrector checkpoint state. Default implementation is a no-op."""

    @abc.abstractmethod
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> CorrectorOutput:
        """Apply corrections to ``gen_data``.

        Args:
            input_data: Denormalized data from the previous time step.
            gen_data: Raw model output for the current step, to be corrected.
            forcing_data: Forcing data at the current step.
            corrector_state: Per-sample state carried across step calls,
                or None if no state has been seeded. Implementations that do
                not maintain state should pass this through unchanged.

        Returns:
            A ``CorrectorOutput`` carrying the corrected generated data, the
            per-variable correction ``delta`` diagnostics, and the updated
            corrector state.
        """
        ...


class CorrectionSequence(CorrectorABC):
    """A corrector that applies an ordered sequence of ``Correction`` objects.

    The sequence (and thus the order in which corrections are applied) is built
    by the corrector config's ``_build``; the corrector itself only knows to
    apply each correction in turn.
    """

    def __init__(self, corrections: list[Correction]):
        self._corrections = corrections

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> CorrectorOutput:
        # Snapshot the entry data (holding references); corrections apply
        # out-of-place, so these tensors are never mutated and can be diffed
        # against the corrected output to build the per-variable delta.
        snapshot = dict(gen_data)
        gen_data = dict(gen_data)
        modified: set[str] = set()
        for correction in self._corrections:
            changed, corrector_state = correction(
                input_data, gen_data, forcing_data, corrector_state
            )
            # ``changed`` holds only the fields this correction modified; its
            # keys are the set of variables it is responsible for writing.
            gen_data.update(changed)
            modified |= changed.keys()
        corrected = dict(gen_data)
        return CorrectorOutput(
            corrected=corrected,
            diagnostics=build_corrector_diagnostics(snapshot, corrected, modified),
            corrector_state=corrector_state,
        )


class EpochScheduledCorrector(CorrectorABC):
    """Wrap a corrector so it is skipped for train-mode steps during the first
    ``disabled_epochs`` training epochs, while always being applied in eval mode.
    """

    def __init__(self, wrapped: CorrectorABC, disabled_epochs: int):
        if disabled_epochs < 0:
            raise ValueError(
                f"disabled_epochs must be non-negative, got {disabled_epochs}"
            )
        self._wrapped = wrapped
        self._disabled_epochs = disabled_epochs
        # Assume the first epoch until set_epoch is called, so the wrapped
        # corrector is disabled for train-mode steps taken before the trainer
        # signals an epoch boundary.
        self._corrector_disabled = disabled_epochs > 0
        self._training = True

    def train(self, mode: bool = True) -> "EpochScheduledCorrector":
        self._training = mode
        self._wrapped.train(mode)
        return self

    def set_epoch(self, epoch: int) -> None:
        self._corrector_disabled = epoch <= self._disabled_epochs
        self._wrapped.set_epoch(epoch)

    def get_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self._disabled_epochs > 0:
            # persisted so that mid-epoch resume, which does not signal an
            # epoch boundary via set_epoch, keeps the corrector state of the
            # interrupted epoch
            state["corrector_disabled"] = self._corrector_disabled
        wrapped_state = self._wrapped.get_state()
        if len(wrapped_state) > 0:
            state["wrapped"] = wrapped_state
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        if self._disabled_epochs > 0 and "corrector_disabled" not in state:
            raise ValueError(
                "EpochScheduledCorrector state is missing 'corrector_disabled'"
            )
        if "corrector_disabled" in state:
            self._corrector_disabled = state["corrector_disabled"]
        self._wrapped.load_state(state.get("wrapped", {}))

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> CorrectorOutput:
        if self._corrector_disabled and self._training:
            # Nothing was applied: pass the data through with empty diagnostics.
            return CorrectorOutput(
                corrected=dict(gen_data), corrector_state=corrector_state
            )
        return self._wrapped(input_data, gen_data, forcing_data, corrector_state)
