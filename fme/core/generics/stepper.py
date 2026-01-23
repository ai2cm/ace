import abc
from typing import Any, Generic, TypeVar

from torch import nn

PS = TypeVar("PS")  # prognostic state
BD = TypeVar("BD")  # batch data
FD = TypeVar("FD")  # forcing data
SD = TypeVar("SD")  # stepped data


class StepperABC(abc.ABC, Generic[PS, BD, FD, SD]):
    """
    Abstract base class for steppers that perform inference (forward stepping).

    This class defines the interface for stepping a model forward in time,
    without any training-specific functionality. Training functionality is
    added by TrainStepperABC which inherits from this class.

    Type Parameters:
        PS: Prognostic state type
        BD: Batch data type (output of predict)
        FD: Forcing data type (input to predict)
        SD: Stepped data type (output of predict_paired, typically paired data)
    """

    SelfType = TypeVar("SelfType", bound="StepperABC")

    @property
    @abc.abstractmethod
    def modules(self) -> nn.ModuleList:
        """Returns the list of modules being used for inference."""
        pass

    @property
    @abc.abstractmethod
    def n_ic_timesteps(self) -> int:
        """Number of initial condition timesteps required."""
        pass

    @abc.abstractmethod
    def predict(
        self,
        initial_condition: PS,
        forcing: FD,
        compute_derived_variables: bool = False,
    ) -> tuple[BD, PS]:
        """
        Predict multiple steps forward given initial condition and forcing data.

        Args:
            initial_condition: The initial condition state.
            forcing: The forcing data for all timesteps.
            compute_derived_variables: Whether to compute derived variables.

        Returns:
            A tuple of (prediction_data, final_prognostic_state).
        """
        pass

    @abc.abstractmethod
    def predict_paired(
        self,
        initial_condition: PS,
        forcing: FD,
        compute_derived_variables: bool = False,
    ) -> tuple[SD, PS]:
        """
        Predict multiple steps forward and return paired prediction/reference data.

        Args:
            initial_condition: The initial condition state.
            forcing: The forcing data for all timesteps.
            compute_derived_variables: Whether to compute derived variables.

        Returns:
            A tuple of (paired_data, final_prognostic_state).
        """
        pass

    @abc.abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Returns the serializable state of the stepper."""
        pass

    @abc.abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """Load state into the stepper."""
        pass

    @classmethod
    @abc.abstractmethod
    def from_state(cls: type[SelfType], state: dict[str, Any]) -> SelfType:
        """Create a stepper from a serialized state."""
        pass

    def set_eval(self) -> None:
        """Set all modules to evaluation mode."""
        for module in self.modules:
            module.eval()

    def set_train(self) -> None:
        """Set all modules to training mode."""
        for module in self.modules:
            module.train()
