import abc
import dataclasses
import warnings
from collections.abc import Callable

# we use Type to distinguish from type attr of StepSelector
from typing import Any, ClassVar, Type, TypeVar, cast, final  # noqa: UP035

import torch
from torch import nn

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
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "StepABC":
        """
        Args:
            dataset_info: Information about the training dataset.
            init_weights: Function to initialize the weights of the step before
                wrapping in DistributedDataParallel. This is particularly useful
                when freezing parameters, as the DistributedDataParallel will
                otherwise expect frozen weights to have gradients, and will
                raise an exception.

        Returns:
            The state of the stepper.
        """

    @property
    @abc.abstractmethod
    def n_ic_timesteps(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def input_names(self) -> list[str]:
        pass

    @property
    @abc.abstractmethod
    def output_names(self) -> list[str]:
        """
        Names of variables output by the step.
        """
        pass

    @property
    @abc.abstractmethod
    def next_step_input_names(self) -> list[str]:
        """
        Names of variables required in next_step_input_data for .step.
        """
        pass

    @property
    @final
    def prognostic_names(self) -> list[str]:
        return list(set(self.input_names).intersection(self.output_names))

    @property
    @abc.abstractmethod
    def loss_names(self) -> list[str]:
        """
        Names of variables to be included in the loss function.
        """
        pass

    @abc.abstractmethod
    def get_next_step_forcing_names(self) -> list[str]:
        pass

    @abc.abstractmethod
    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ) -> StandardNormalizer:
        """
        Args:
            extra_names: Names of additional variables to include in the
                loss normalizer.
            extra_residual_scaled_names: extra_names which use residual scale factors,
                if enabled.

        Returns:
            The loss normalizer.
        """

    @abc.abstractmethod
    def replace_ocean(self, ocean: OceanConfig | None):
        pass

    @abc.abstractmethod
    def get_ocean(self) -> OceanConfig | None:
        pass

    @abc.abstractmethod
    def load(self):
        """
        Update configuration in-place so it does not depend on external files.
        """
        pass


T = TypeVar("T", bound=StepConfigABC)


@dataclasses.dataclass
class StepSelector(StepConfigABC):
    type: str
    config: dict[str, Any]
    registry: ClassVar[Registry] = Registry()

    def __post_init__(self):
        self._step_config_instance: StepConfigABC = cast(
            StepConfigABC, self.registry.get(self.type, self.config)
        )

    @property
    def n_ic_timesteps(self) -> int:
        return self._step_config_instance.n_ic_timesteps

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:  # noqa: UP006
        return cls.registry.register(name)

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None] = lambda x: None,
    ) -> "StepABC":
        """
        Args:
            dataset_info: Information about the training dataset.
            init_weights: Function to initialize the weights of the step before
                wrapping in DistributedDataParallel. This is particularly useful
                when freezing parameters, as the DistributedDataParallel will
                otherwise expect frozen weights to have gradients, and will
                raise an exception.

        Returns:
            The state of the stepper.
        """
        return self._step_config_instance.get_step(dataset_info, init_weights)

    @classmethod
    def get_available_types(cls) -> set[str]:
        """This class method is used to expose all available types of Steps."""
        return set(cls(type="", config={}).registry._types.keys())

    def get_next_step_forcing_names(self) -> list[str]:
        return self._step_config_instance.get_next_step_forcing_names()

    @property
    def input_names(self) -> list[str]:
        return self._step_config_instance.input_names

    @property
    def output_names(self) -> list[str]:
        """
        Names of variables output by the step.
        """
        return self._step_config_instance.output_names

    @property
    def next_step_input_names(self) -> list[str]:
        """
        Names of variables required in next_step_input_data for .step.
        """
        return self._step_config_instance.next_step_input_names

    @property
    def loss_names(self) -> list[str]:
        """
        Names of variables to be included in the loss function.
        """
        return self._step_config_instance.loss_names

    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ) -> StandardNormalizer:
        return self._step_config_instance.get_loss_normalizer(
            extra_names=extra_names,
            extra_residual_scaled_names=extra_residual_scaled_names,
        )

    def replace_ocean(self, ocean: OceanConfig | None):
        self._step_config_instance.replace_ocean(ocean)
        self.config = dataclasses.asdict(self._step_config_instance)

    def get_ocean(self) -> OceanConfig | None:
        return self._step_config_instance.get_ocean()

    def load(self):
        self._step_config_instance.load()
        self.config = dataclasses.asdict(self._step_config_instance)


class StepABC(abc.ABC, nn.Module):
    SelfType = TypeVar("SelfType", bound="StepABC")

    @property
    @abc.abstractmethod
    def config(self) -> StepConfigABC:
        pass

    @final
    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ) -> StandardNormalizer:
        return self.config.get_loss_normalizer(
            extra_names=extra_names,
            extra_residual_scaled_names=extra_residual_scaled_names,
        )

    @property
    @final
    def n_ic_timesteps(self) -> int:
        return self.config.n_ic_timesteps

    @property
    @final
    def input_names(self) -> list[str]:
        return self.config.input_names

    @property
    @final
    def output_names(self) -> list[str]:
        return self.config.output_names

    @property
    @final
    def prognostic_names(self) -> list[str]:
        return self.config.prognostic_names

    @property
    @final
    def loss_names(self) -> list[str]:
        return self.config.loss_names

    @property
    @abc.abstractmethod
    def modules(self) -> nn.ModuleList:
        pass

    @property
    @abc.abstractmethod
    def normalizer(self) -> StandardNormalizer:
        pass

    @property
    @final
    def next_step_input_names(self) -> list[str]:
        """
        Names of variables required in next_step_input_data for .step.
        """
        return self.config.next_step_input_names

    @property
    @final
    def next_step_forcing_names(self) -> list[str]:
        """Names of input variables which come from the output timestep."""
        return self.config.get_next_step_forcing_names()

    @property
    @abc.abstractmethod
    def surface_temperature_name(self) -> str | None:
        """
        Name of the surface temperature variable, if one is available.
        """
        pass

    @property
    @abc.abstractmethod
    def ocean_fraction_name(self) -> str | None:
        """
        Name of the ocean fraction variable, if one is available.
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
        self: SelfType,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
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
            wrapper: Wrapper to apply over each nn.Module before calling.

        Returns:
            The denormalized output data at the next time step.
        """
        pass

    @final
    def forward(
        self, input: TensorMapping, next_step_input_data: TensorMapping
    ) -> TensorDict:
        return self.step(input, next_step_input_data)

    @final
    def export(
        self: SelfType,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
    ) -> torch.export.ExportedProgram:
        """
        Script the step function.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*does not reference an nn.Module.*"
            )
            return torch.export.export(self, (input, next_step_input_data))

    @abc.abstractmethod
    def get_state(self) -> dict[str, Any]:
        """
        Returns:
            The state of the step object as expected by load_state,
                may or may not include initialization parameters.
        """
        pass

    @abc.abstractmethod
    def load_state(self, state: dict[str, Any]):
        """
        Load the state of the step object.
        """
        pass
