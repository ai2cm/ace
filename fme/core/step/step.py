import abc
import datetime
from typing import Any, Dict, List, Tuple, Type, TypeVar

from fme.core.coordinates import VerticalCoordinate
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping


class StepConfigABC(abc.ABC):
    @abc.abstractmethod
    def get_step(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        timestep: datetime.timedelta,
    ) -> "StepABC":
        """
        Returns:
            The state of the stepper.
        """


class StepABC(abc.ABC):
    SelfType = TypeVar("SelfType", bound="StepABC")

    @property
    @abc.abstractmethod
    def input_names(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def output_names(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def next_step_forcing_names(self) -> List[str]:
        pass

    @abc.abstractmethod
    def step(
        self,
        input: TensorMapping,
        next_step_forcing_data: TensorMapping,
        use_activation_checkpointing: bool = False,
    ) -> TensorDict:
        """
        Step the model forward one timestep given input data.

        Args:
            input: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This data is used as input for pytorch
                module(s) and is assumed to contain all input variables
                and be denormalized.
            next_step_forcing_data: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This must contain the necessary forcing
                data at the output timestep, such as might be needed to prescribe
                sea surface temperature or use a corrector.
            use_activation_checkpointing: If True, wrap module calls with
                torch.utils.checkpoint.checkpoint, reducing memory consumption
                in exchange for increased computation. This is only relevant during
                training and otherwise has no effect.

        Returns:
            The denormalized output data at the next time step.
        """
        pass

    @abc.abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns:
            The state of the stepper.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_state(cls: Type[SelfType], state: Dict[str, Any]) -> SelfType:
        """
        Load the state of the stepper.

        Args:
            state: The state to load.

        Returns:
            The stepper.
        """
        pass
