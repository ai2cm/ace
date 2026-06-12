import abc
import dataclasses
from collections.abc import Mapping
from typing import Any, Self, final

import dacite

from fme.core.corrector.state import CorrectorState
from fme.core.dataset_info import DatasetInfo
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class CorrectorConfigABC(abc.ABC):
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

    @abc.abstractmethod
    def get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "CorrectorABC": ...


@dataclasses.dataclass
class EpochScheduledCorrectorConfigABC(CorrectorConfigABC):
    corrector_disabled_epochs: int = dataclasses.field(default=0, kw_only=True)

    def __post_init__(self):
        if self.corrector_disabled_epochs < 0:
            raise ValueError(
                "corrector_disabled_epochs must be non-negative, got "
                f"{self.corrector_disabled_epochs}"
            )

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


class CorrectorABC(abc.ABC):
    @abc.abstractmethod
    def train(self, mode: bool = True) -> "CorrectorABC":
        """Set the corrector to training or evaluation mode."""
        ...

    @final
    def eval(self) -> "CorrectorABC":
        """Set the corrector to evaluation mode."""
        return self.train(False)

    @abc.abstractmethod
    def set_epoch(self, epoch: int) -> None:
        """Called by the stepper at the start of each training epoch."""
        ...

    @abc.abstractmethod
    def get_state(self) -> dict[str, Any]:
        """
        Return corrector checkpoint state.

        Correctors without checkpointed state can return an empty dict.
        """
        ...

    @abc.abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """Load corrector checkpoint state."""
        ...

    @abc.abstractmethod
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """Apply corrections to ``gen_data``.

        Args:
            input_data: Denormalized data from the previous time step.
            gen_data: Raw model output for the current step, to be corrected.
            forcing_data: Forcing data at the current step.
            corrector_state: Per-sample state carried across step calls,
                or None if no state has been seeded. Implementations that do
                not maintain state should pass this through unchanged.

        Returns:
            A tuple ``(corrected_gen_data, corrector_state)``.
        """
        ...


class EpochScheduledCorrector(CorrectorABC):
    def __init__(self, wrapped: CorrectorABC, disabled_epochs: int):
        if disabled_epochs < 0:
            raise ValueError(
                f"disabled_epochs must be non-negative, got {disabled_epochs}"
            )
        self._wrapped = wrapped
        self._disabled_epochs = disabled_epochs
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
    ) -> tuple[TensorDict, CorrectorState | None]:
        if self._corrector_disabled and self._training:
            return dict(gen_data), corrector_state
        return self._wrapped(input_data, gen_data, forcing_data, corrector_state)
