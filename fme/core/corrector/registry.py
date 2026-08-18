import abc
import dataclasses
from collections.abc import Collection, Mapping
from typing import Any, Protocol, Self, final

import dacite
import torch

from fme.core.corrector.output import CorrectorOutput, build_corrector_diagnostics
from fme.core.corrector.state import CorrectorState
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed.distributed import Distributed
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class CorrectorConfigABC(abc.ABC):
    """Base for corrector configs.

    Subclasses implement ``_get_corrector``. The ``corrector_disabled_epochs``
    option is handled here: ``get_corrector`` wraps the built corrector in an
    ``EpochScheduledCorrector`` when it is greater than zero, and runs
    modified-name discovery on the result.

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
    def get_corrector(
        self,
        dataset_info: DatasetInfo,
        input_names: Collection[str],
        gen_names: Collection[str],
        forcing_names: Collection[str],
    ) -> "CorrectorABC":
        """Build the corrector and discover the delta keys it produces.

        Discovery runs here, as part of construction, so ``modified_names`` is
        populated for every corrector type and no caller has to update it
        statefully. ``img_shape`` comes from ``dataset_info``.

        Args:
            dataset_info: Information about the dataset the corrector runs on.
            input_names: Names present in the corrector's ``input_data``.
            gen_names: Names present in the corrector's ``gen_data``.
            forcing_names: Names present in the corrector's ``forcing_data``.
        """
        corrector = self._build_corrector(dataset_info)
        corrector._discover_modified_names(
            input_names=input_names,
            gen_names=gen_names,
            forcing_names=forcing_names,
            img_shape=dataset_info.img_shape,
        )
        return corrector

    @final
    def _build_corrector(self, dataset_info: DatasetInfo) -> "CorrectorABC":
        """Build the corrector, applying ``corrector_disabled_epochs``, without
        running discovery.

        Exists so that a config which delegates to another config (see
        ``CorrectorSelector``) can compose the build without triggering a
        second discovery pass.
        """
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
    truth for what changed and cannot drift from the write.

    The key set a correction returns may depend on its config and on which keys
    are present in its inputs, never on tensor values; modified-name discovery
    at corrector construction relies on this.
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

    @property
    def modified_names(self) -> frozenset[str]:
        """The delta keys this corrector produces when active.

        Default implementation returns an empty frozenset: a corrector which
        modifies nothing.
        """
        return frozenset()

    def _discover_modified_names(
        self,
        input_names: Collection[str],
        gen_names: Collection[str],
        forcing_names: Collection[str],
        img_shape: tuple[int, int],
    ) -> None:
        """Discover the delta keys this corrector produces when active, making
        them available as ``modified_names``.

        Called by ``CorrectorConfigABC.get_corrector`` during construction, not
        by corrector users. Default implementation is a no-op, leaving
        ``modified_names`` empty.
        """

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
        self._modified_names: frozenset[str] = frozenset()

    @property
    def modified_names(self) -> frozenset[str]:
        return self._modified_names

    def _discover_modified_names(
        self,
        input_names: Collection[str],
        gen_names: Collection[str],
        forcing_names: Collection[str],
        img_shape: tuple[int, int],
    ) -> None:
        """Run one ``__call__`` on zero tensors keyed by the given names and
        record the resulting delta keys as ``modified_names``.

        The fake-data values may go NaN/inf inside budget corrections (they
        divide by global means); that is harmless for key discovery because the
        key set a correction returns depends only on its config and on which
        keys are present in its inputs, never on tensor values (the
        ``Correction`` contract).
        """
        dist = Distributed.get_instance()

        def _zeros(names: Collection[str]) -> TensorDict:
            # corrections operate on this rank's spatial shard, so the fake
            # data must be sharded like real data (img_shape is global)
            return dist.scatter_spatial(
                {
                    name: torch.zeros((1, *img_shape), device=get_device())
                    for name in names
                },
                img_shape,
            )

        with torch.no_grad():
            result = self(
                _zeros(input_names), _zeros(gen_names), _zeros(forcing_names), None
            )
        self._modified_names = frozenset(result.diagnostics.delta.keys())

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

    @property
    def modified_names(self) -> frozenset[str]:
        # Forwarded to the wrapped corrector independent of the epoch-disabled
        # state: these describe what the corrector produces when active.
        return self._wrapped.modified_names

    def _discover_modified_names(
        self,
        input_names: Collection[str],
        gen_names: Collection[str],
        forcing_names: Collection[str],
        img_shape: tuple[int, int],
    ) -> None:
        self._wrapped._discover_modified_names(
            input_names, gen_names, forcing_names, img_shape
        )

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
