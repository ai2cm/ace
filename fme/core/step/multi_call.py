import dataclasses
from typing import Any, Callable, Dict, List, Optional, TypeVar

import torch

from fme.core.dataset_info import DatasetInfo
from fme.core.multi_call import MultiCallConfig
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean import OceanConfig
from fme.core.step.step import (
    InferenceDataProtocol,
    StepABC,
    StepConfigABC,
    StepSelector,
)
from fme.core.typing_ import TensorDict, TensorMapping


@StepSelector.register("multi_call")
@dataclasses.dataclass
class MultiCallStepConfig(StepConfigABC):
    """
    Configuration for a multi-call step.

    Parameters:
        wrapped_step: The step to wrap.
        config: The multi-call configuration.
        include_multi_call_in_loss: Whether to include multi-call diagnostics in the
            loss.
    """

    wrapped_step: StepSelector
    config: Optional[MultiCallConfig] = None
    include_multi_call_in_loss: bool = True

    def __post_init__(self):
        if self.config is not None:
            self.config.validate(
                self.wrapped_step.input_names, self.wrapped_step.output_names
            )
        if self.config is None and self.include_multi_call_in_loss:
            raise ValueError("include_multi_call_in_loss is True, but config is None")

    def get_step(
        self,
        dataset_info: DatasetInfo,
    ) -> "MultiCallStep":
        wrapped = self.wrapped_step.get_step(dataset_info)
        return MultiCallStep(
            wrapped_step=wrapped,
            config=self,
        )

    def build(
        self,
        step_method: Callable[[TensorMapping, TensorMapping, bool], TensorDict],
    ):
        if self.config is None:
            return step_method
        return self.config.build(step_method)

    @property
    def _multi_call_outputs(self) -> List[str]:
        if self.config is None:
            return []
        return self.config.names

    @property
    def forcing_names(self) -> List[str]:
        return self.wrapped_step.forcing_names

    @property
    def diagnostic_names(self) -> List[str]:
        return self.wrapped_step.diagnostic_names + self._multi_call_outputs

    @property
    def prognostic_names(self) -> List[str]:
        return self.wrapped_step.prognostic_names

    @property
    def output_names(self) -> List[str]:
        return self.wrapped_step.output_names + self._multi_call_outputs

    @property
    def loss_names(self) -> List[str]:
        if self.include_multi_call_in_loss:
            return self.wrapped_step.loss_names + self._multi_call_outputs
        else:
            return self.wrapped_step.loss_names

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        self.wrapped_step.replace_ocean(ocean)

    @property
    def n_ic_timesteps(self) -> int:
        return self.wrapped_step.n_ic_timesteps


class MultiCallStep(StepABC):
    """
    Step class for a single pytorch module.
    """

    SelfType = TypeVar("SelfType", bound="MultiCallStep")

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        wrapped_step: StepABC,
        config: MultiCallStepConfig,
    ):
        """
        Args:
            wrapped_step: The step to wrap.
            config: The multi-call configuration.
            include_multi_call_in_loss: Whether to include multi-call diagnostics in the
                loss.
        """
        self._wrapped_step = wrapped_step
        self._config = config
        self._multi_call = config.build(self._wrapped_step.step)
        self._include_multi_call_in_loss = config.include_multi_call_in_loss

    @property
    def config(self) -> MultiCallStepConfig:
        return self._config

    @property
    def modules(self) -> torch.nn.ModuleList:
        return self._wrapped_step.modules

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._wrapped_step.normalizer

    @property
    def next_step_input_names(self) -> List[str]:
        return self._wrapped_step.next_step_input_names

    @property
    def next_step_forcing_names(self) -> List[str]:
        return self._wrapped_step.next_step_forcing_names

    @property
    def surface_temperature_name(self) -> Optional[str]:
        return self._wrapped_step.surface_temperature_name

    @property
    def ocean_fraction_name(self) -> Optional[str]:
        return self._wrapped_step.ocean_fraction_name

    def validate_inference_data(self, data: InferenceDataProtocol):
        self._wrapped_step.validate_inference_data(data)

    def get_regularizer_loss(self) -> torch.Tensor:
        return self._wrapped_step.get_regularizer_loss()

    def step(
        self,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
        use_activation_checkpointing: bool = False,
    ) -> TensorDict:
        state = self._wrapped_step.step(
            input,
            next_step_input_data,
            use_activation_checkpointing,
        )
        if self._multi_call is not None:
            multi_called_outputs = self._multi_call.step(
                input, next_step_input_data, use_activation_checkpointing
            )
            state = {**multi_called_outputs, **state}
        return state

    def get_state(self) -> Dict[str, Any]:
        return {
            "wrapped_step": self._wrapped_step.get_state(),
        }

    def load_state(self, state: Dict[str, Any]):
        self._wrapped_step.load_state(state["wrapped_step"])
