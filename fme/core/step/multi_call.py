import dataclasses
import datetime
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch

from fme.core.coordinates import VerticalCoordinate
from fme.core.gridded_ops import GriddedOperations
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
    config: MultiCallConfig
    include_multi_call_in_loss: bool = True

    def get_step(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        timestep: datetime.timedelta,
    ) -> "MultiCallStep":
        wrapped = self.wrapped_step.get_step(
            img_shape, gridded_operations, vertical_coordinate, timestep
        )
        return MultiCallStep(
            wrapped_step=wrapped,
            config=self.config,
            include_multi_call_in_loss=self.include_multi_call_in_loss,
        )


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
        config: MultiCallConfig,
        include_multi_call_in_loss: bool = True,
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
        self._config.validate(
            self._wrapped_step.input_names, self._wrapped_step.output_names
        )
        self._multi_call = config.build(self._wrapped_step.step)
        residual_scaled_names = []
        for prog_name in set(self._wrapped_step.prognostic_names).intersection(
            config.output_names
        ):
            residual_scaled_names.extend(config.get_multi_called_names(prog_name))
        self._multi_call_residual_scaled_names = residual_scaled_names
        self._include_multi_call_in_loss = include_multi_call_in_loss

    @property
    def modules(self) -> torch.nn.ModuleList:
        return self._wrapped_step.modules

    @property
    def prognostic_names(self) -> List[str]:
        return self._wrapped_step.prognostic_names

    @property
    def residual_scaled_names(self) -> List[str]:
        return (
            self._wrapped_step.prognostic_names + self._multi_call_residual_scaled_names
        )

    @property
    def forcing_names(self) -> List[str]:
        return self._wrapped_step.forcing_names

    @property
    def diagnostic_names(self) -> List[str]:
        return self._wrapped_step.diagnostic_names + self._multi_call.names

    @property
    def output_names(self) -> List[str]:
        return self._wrapped_step.output_names + self._multi_call.names

    @property
    def loss_names(self) -> List[str]:
        if self._include_multi_call_in_loss:
            return self._wrapped_step.loss_names + self._multi_call.names
        else:
            return self._wrapped_step.loss_names

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

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        self._wrapped_step.replace_ocean(ocean)

    def validate_inference_data(self, data: InferenceDataProtocol):
        self._wrapped_step.validate_inference_data(data)

    @property
    def n_ic_timesteps(self) -> int:
        return self._wrapped_step.n_ic_timesteps

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
