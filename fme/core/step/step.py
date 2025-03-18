import abc
import dataclasses
import datetime
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    cast,
    final,
)

import torch

from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean import OceanConfig
from fme.core.registry.registry import Registry
from fme.core.typing_ import TensorDict, TensorMapping


# Children still need to decorate with @dataclass, otherwise
# they will be a dataclass with no dataclass fields.
@dataclasses.dataclass
class StepConfigABC(abc.ABC):
    @abc.abstractmethod
    def get_step(
        self,
        dataset_info: DatasetInfo,
    ) -> "StepABC":
        """
        Returns:
            The state of the stepper.
        """

    @property
    @abc.abstractmethod
    def n_ic_timesteps(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def prognostic_names(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def forcing_names(self) -> List[str]:
        pass

    @property
    def input_names(self) -> List[str]:
        return list(set(self.prognostic_names).union(self.forcing_names))

    @property
    @abc.abstractmethod
    def output_names(self) -> List[str]:
        """
        Names of variables output by the step.
        """
        pass

    @property
    @abc.abstractmethod
    def loss_names(self) -> List[str]:
        """
        Names of variables to be included in the loss function.
        """
        pass

    @abc.abstractmethod
    def get_loss_normalizer(
        self,
        extra_diagnostic_names: Optional[List[str]] = None,
        extra_prognostic_names: Optional[List[str]] = None,
    ) -> StandardNormalizer:
        """
        Args:
            extra_diagnostic_names: Names of diagnostics to include in the loss
                normalizer. These will generally use full-field scale factors.
            extra_prognostic_names: Names of prognostics to include in the loss
                normalizer. These may use residual scale factors.

        Returns:
            The loss normalizer.
        """

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        pass


T = TypeVar("T", bound=StepConfigABC)


@dataclasses.dataclass
class StepSelector(StepConfigABC):
    type: str
    config: Dict[str, Any]
    registry: ClassVar[Registry] = Registry()

    def __post_init__(self):
        self._step_config_instance: StepConfigABC = cast(
            StepConfigABC, self.registry.get(self.type, self.config)
        )

    @property
    def n_ic_timesteps(self) -> int:
        return self._step_config_instance.n_ic_timesteps

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        return cls.registry.register(name)

    def get_step(
        self,
        dataset_info: DatasetInfo,
    ) -> "StepABC":
        return self._step_config_instance.get_step(dataset_info)

    def get_state(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "config": self.config,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "StepSelector":
        return cls(type=state["type"], config=state["config"])

    @classmethod
    def get_available_types(cls) -> Set[str]:
        """This class method is used to expose all available types of Steps."""
        return set(cls(type="", config={}).registry._types.keys())

    @property
    def prognostic_names(self) -> List[str]:
        return self._step_config_instance.prognostic_names

    @property
    def forcing_names(self) -> List[str]:
        return self._step_config_instance.forcing_names

    @property
    def input_names(self) -> List[str]:
        return self._step_config_instance.input_names

    @property
    def output_names(self) -> List[str]:
        """
        Names of variables output by the step.
        """
        return self._step_config_instance.output_names

    @property
    def loss_names(self) -> List[str]:
        """
        Names of variables to be included in the loss function.
        """
        return self._step_config_instance.loss_names

    def get_loss_normalizer(
        self,
        extra_diagnostic_names: Optional[List[str]] = None,
        extra_prognostic_names: Optional[List[str]] = None,
    ) -> StandardNormalizer:
        return self._step_config_instance.get_loss_normalizer(
            extra_diagnostic_names, extra_prognostic_names
        )

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        self._step_config_instance.replace_ocean(ocean)
        self.config = dataclasses.asdict(self._step_config_instance)


class InferenceDataProtocol(Protocol):
    @property
    def timestep(self) -> datetime.timedelta:
        pass


class StepABC(abc.ABC):
    SelfType = TypeVar("SelfType", bound="StepABC")

    @property
    @abc.abstractmethod
    def config(self) -> StepConfigABC:
        pass

    @property
    @final
    def n_ic_timesteps(self) -> int:
        return self.config.n_ic_timesteps

    @property
    @final
    def input_names(self) -> List[str]:
        return self.config.input_names

    @property
    @final
    def output_names(self) -> List[str]:
        return self.config.output_names

    @property
    @final
    def prognostic_names(self) -> List[str]:
        return self.config.prognostic_names

    @property
    @final
    def forcing_names(self) -> List[str]:
        return self.config.forcing_names

    @property
    @final
    def loss_names(self) -> List[str]:
        return self.config.loss_names

    @property
    @abc.abstractmethod
    def modules(self) -> torch.nn.ModuleList:
        pass

    @property
    @abc.abstractmethod
    def normalizer(self) -> StandardNormalizer:
        pass

    @property
    @abc.abstractmethod
    def next_step_input_names(self) -> List[str]:
        """
        Names of variables required in next_step_input_data for .step.
        """
        pass

    @property
    @abc.abstractmethod
    def next_step_forcing_names(self) -> List[str]:
        """Names of input variables which come from the output timestep."""
        pass

    @property
    @abc.abstractmethod
    def surface_temperature_name(self) -> Optional[str]:
        """
        Name of the surface temperature variable, if one is available.
        """
        pass

    @property
    @abc.abstractmethod
    def ocean_fraction_name(self) -> Optional[str]:
        """
        Name of the ocean fraction variable, if one is available.
        """
        pass

    @abc.abstractmethod
    def validate_inference_data(self, data: InferenceDataProtocol):
        """
        Validate the inference data.
        """
        pass

    @abc.abstractmethod
    def get_regularizer_loss(self) -> torch.Tensor:
        """
        Get the regularizer loss.
        """
        pass

    @abc.abstractmethod
    def step(
        self,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
        use_activation_checkpointing: bool = False,
    ) -> TensorDict:
        """
        Step the model forward one timestep given input data.

        Args:
            input: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This data is used as input for pytorch
                module(s) and is assumed to contain all input variables
                and be denormalized.
            next_step_input_data: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This must contain the necessary input
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
            The state of the step object as expected by load_state,
                may or may not include initialization parameters.
        """
        pass

    @abc.abstractmethod
    def load_state(self, state: Dict[str, Any]):
        """
        Load the state of the step object.
        """
        pass
