from typing import Any, Dict, List, Optional

import torch

from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean import OceanConfig
from fme.core.step.step import InferenceDataProtocol, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping


class SerializableStep:
    def __init__(
        self,
        selector: StepSelector,
        dataset_info: DatasetInfo,
    ):
        self._selector = selector
        self._instance = selector.get_step(dataset_info)
        self._dataset_info = dataset_info

    @property
    def modules(self) -> torch.nn.ModuleList:
        return self._instance.modules

    @property
    def prognostic_names(self) -> List[str]:
        return self._instance.prognostic_names

    @property
    def forcing_names(self) -> List[str]:
        return self._instance.forcing_names

    @property
    def diagnostic_names(self) -> List[str]:
        return self._instance.diagnostic_names

    @property
    def input_names(self) -> List[str]:
        return self._instance.input_names

    @property
    def output_names(self) -> List[str]:
        return self._instance.output_names

    @property
    def loss_names(self) -> List[str]:
        return self._instance.loss_names

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._instance.normalizer

    @property
    def next_step_input_names(self) -> List[str]:
        return self._instance.next_step_input_names

    @property
    def next_step_forcing_names(self) -> List[str]:
        return self._instance.next_step_forcing_names

    @property
    def surface_temperature_name(self) -> Optional[str]:
        return self._instance.surface_temperature_name

    @property
    def ocean_fraction_name(self) -> Optional[str]:
        return self._instance.ocean_fraction_name

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        """
        Replace the ocean configuration.
        """
        self._instance.replace_ocean(ocean)

    def validate_inference_data(self, data: InferenceDataProtocol):
        """
        Validate the inference data.
        """
        return self._instance.validate_inference_data(data)

    @property
    def n_ic_timesteps(self) -> int:
        return self._instance.n_ic_timesteps

    def get_regularizer_loss(self) -> torch.Tensor:
        """
        Get the regularizer loss.
        """
        return self._instance.get_regularizer_loss()

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
        return self._instance.step(
            input, next_step_input_data, use_activation_checkpointing
        )

    def to_state(self) -> Dict[str, Any]:
        """
        Returns:
            The state of the stepper.
        """
        return {
            "selector": self._selector.get_state(),
            "instance": self._instance.get_state(),
            "dataset_info": self._dataset_info.to_state(),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "SerializableStep":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.

        Returns:
            The stepper.
        """
        selector = StepSelector.from_state(state["selector"])
        dataset_info = DatasetInfo.from_state(state["dataset_info"])
        stepper = cls(selector, dataset_info)
        stepper._instance.load_state(state["instance"])
        return stepper
