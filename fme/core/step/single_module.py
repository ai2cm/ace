import dataclasses
import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import dacite
import torch
from torch import nn

from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset.utils import encode_timestep
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.normalizer import NetworkAndLossNormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import ActivationCheckpointingConfig, NullOptimization
from fme.core.packer import Packer
from fme.core.registry import CorrectorSelector, ModuleSelector
from fme.core.step.step import (
    InferenceDataProtocol,
    StepABC,
    StepConfigABC,
    StepSelector,
)
from fme.core.typing_ import TensorDict, TensorMapping

DEFAULT_TIMESTEP = datetime.timedelta(hours=6)
DEFAULT_ENCODED_TIMESTEP = encode_timestep(DEFAULT_TIMESTEP)


@StepSelector.register("single_module")
@dataclasses.dataclass
class SingleModuleStepConfig(StepConfigABC):
    """
    Configuration for a single module stepper.

    Parameters:
        builder: The module builder.
        in_names: Names of input variables.
        out_names: Names of output variables.
        normalization: The normalization configuration.
        parameter_init: The parameter initialization configuration.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Names of forcing variables for the next timestep.
        activation_checkpointing: Configuration for activation checkpointing to trade
            increased computation for lowered memory during training.
        crps_training: Whether to use CRPS training for stochastic models.
    """

    builder: ModuleSelector
    in_names: List[str]
    out_names: List[str]
    normalization: NetworkAndLossNormalizationConfig
    ocean: Optional[OceanConfig] = None
    corrector: Union[AtmosphereCorrectorConfig, CorrectorSelector] = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: List[str] = dataclasses.field(default_factory=list)
    activation_checkpointing: ActivationCheckpointingConfig = dataclasses.field(
        default_factory=lambda: ActivationCheckpointingConfig()
    )
    crps_training: bool = False

    def __post_init__(self):
        for name in self.next_step_forcing_names:
            if name not in self.in_names:
                raise ValueError(
                    f"next_step_forcing_name '{name}' not in in_names: {self.in_names}"
                )
            if name in self.out_names:
                raise ValueError(
                    f"next_step_forcing_name is an output variable: '{name}'"
                )

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    def get_state(self):
        return dataclasses.asdict(self)

    def get_loss_normalizer(
        self,
        extra_diagnostic_names: Optional[List[str]] = None,
        extra_prognostic_names: Optional[List[str]] = None,
    ) -> StandardNormalizer:
        if extra_diagnostic_names is None:
            extra_diagnostic_names = []
        if extra_prognostic_names is None:
            extra_prognostic_names = []
        return self.normalization.get_loss_normalizer(
            names=(
                self._normalize_names + extra_diagnostic_names + extra_prognostic_names
            ),
            residual_scaled_names=self.prognostic_names + extra_prognostic_names,
        )

    @classmethod
    def from_state(cls, state) -> "SingleModuleStepConfig":
        state = cls._remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _normalize_names(self):
        """Names of variables which require normalization. I.e. inputs/outputs."""
        return list(set(self.in_names).union(self.out_names))

    @property
    def forcing_names(self) -> List[str]:
        """Names of variables which are inputs only."""
        return list(set(self.in_names) - set(self.out_names))

    @property
    def prognostic_names(self) -> List[str]:
        """Names of variables which both inputs and outputs."""
        return list(set(self.out_names).intersection(self.in_names))

    @property
    def diagnostic_names(self) -> List[str]:
        """Names of variables which are outputs only."""
        return list(set(self.out_names).difference(self.in_names))

    @property
    def output_names(self) -> List[str]:
        return self.out_names

    @property
    def loss_names(self) -> List[str]:
        return self.output_names

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        """
        Replace the ocean model with a new one.

        Args:
            ocean: The new ocean model configuration or None.
        """
        self.ocean = ocean

    @classmethod
    def _remove_deprecated_keys(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        state_copy = state.copy()
        return state_copy

    def get_step(
        self,
        dataset_info: DatasetInfo,
    ) -> "SingleModuleStep":
        logging.info("Initializing stepper from provided config")
        corrector = dataset_info.vertical_coordinate.build_corrector(
            config=self.corrector,
            gridded_operations=dataset_info.gridded_operations,
            timestep=dataset_info.timestep,
        )
        normalizer = self.normalization.get_network_normalizer(self._normalize_names)
        return SingleModuleStep(
            config=self,
            img_shape=dataset_info.img_shape,
            corrector=corrector,
            normalizer=normalizer,
            timestep=dataset_info.timestep,
        )


class SingleModuleStep(StepABC):
    """
    Step class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SingleModuleStepConfig,
        img_shape: Tuple[int, int],
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        timestep: datetime.timedelta,
    ):
        """
        Args:
            config: The configuration.
            img_shape: Shape of domain as (n_lat, n_lon).
            corrector: The corrector to use at the end of each step.
            normalizer: The normalizer to use.
            timestep: Timestep of the model.
            init_weights: Whether to initialize the weights. Should pass False if
                the weights are about to be overwritten by a checkpoint.
        """
        n_in_channels = len(config.in_names)
        n_out_channels = len(config.out_names)
        self.in_packer = Packer(config.in_names)
        self.out_packer = Packer(config.out_names)
        self._normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Optional[Ocean] = config.ocean.build(
                config.in_names, config.out_names, timestep
            )
        else:
            self.ocean = None
        self.module = config.builder.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=img_shape,
        ).to(get_device())
        self._img_shape = img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        dist = Distributed.get_instance()
        self.module = dist.wrap_module(self.module)

        self._timestep = timestep

        self._corrector = corrector
        self.in_names = config.in_names
        self.out_names = config.out_names

        self._activation_checkpointing = config.activation_checkpointing

    @property
    def config(self) -> SingleModuleStepConfig:
        return self._config

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._normalizer

    @property
    def surface_temperature_name(self) -> Optional[str]:
        if self._config.ocean is not None:
            return self._config.ocean.surface_temperature_name
        return None

    @property
    def ocean_fraction_name(self) -> Optional[str]:
        if self._config.ocean is not None:
            return self._config.ocean.ocean_fraction_name
        return None

    @property
    def next_step_forcing_names(self) -> List[str]:
        return self._config.next_step_forcing_names

    @property
    def next_step_input_names(self) -> List[str]:
        """Names of variables provided in next_step_input_data."""
        if self.ocean is None:
            return list(self.forcing_names)
        return list(set(self.forcing_names).union(self.ocean.forcing_names))

    @property
    def modules(self) -> nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return nn.ModuleList([self.module])

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
                [n_batch, n_lat, n_lon] containing denormalized data from the
                initial timestep. In practice this contains the ML inputs.
            next_step_input_data: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon] containing denormalized data from
                the output timestep. In practice this contains the necessary data
                at the output timestep for the ocean model and corrector.
            use_activation_checkpointing: If True, wrap the module call with
                torch.utils.checkpoint.checkpoint, reducing memory consumption
                in exchange for increased computation. This is only relevant during
                training and otherwise has no effect.

        Returns:
            The denormalized output data at the next time step.
        """
        input_norm = self.normalizer.normalize(input)
        input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
        if use_activation_checkpointing:
            output_tensor = torch.utils.checkpoint.checkpoint(
                self.module,
                input_tensor,
                use_reentrant=False,
                **self._activation_checkpointing.kwargs,
            )
        else:
            output_tensor = self.module(input_tensor)
        output_norm = self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)
        output = self.normalizer.denormalize(output_norm)
        if self._corrector is not None:
            output = self._corrector(input, output, next_step_input_data)
        if self.ocean is not None:
            output = self.ocean(input, output, next_step_input_data)
        return output

    def validate_inference_data(self, data: InferenceDataProtocol):
        if self._timestep != data.timestep:
            raise ValueError(
                f"Timestep of step object, {self._timestep}, does not "
                f"match that of the inference data, {data.timestep}."
            )

    def get_regularizer_loss(self):
        return torch.tensor(0.0)

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "module": self.module.state_dict(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
        """
        module = state["module"]
        if "module.device_buffer" in module:
            # for backwards compatibility with old checkpoints
            del module["module.device_buffer"]
        self.module.load_state_dict(module)
