import dataclasses
import datetime
import logging
from collections.abc import Callable
from typing import Any

import dacite
import torch
from torch import nn

from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset.utils import encode_timestep
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.dicts import add_names
from fme.core.distributed import Distributed
from fme.core.normalizer import NetworkAndLossNormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.packer import Packer
from fme.core.registry import CorrectorSelector, ModuleSelector
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.device import get_device, using_gpu
from fme.ace.models.makani_mpu.mappings import init_gradient_reduction_hooks
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
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Names of forcing variables for the next timestep.
        crps_training: Unused, kept for backwards compatibility.
        residual_prediction: Whether to use residual prediction.
    """

    builder: ModuleSelector
    in_names: list[str]
    out_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    crps_training: bool | None = None
    residual_prediction: bool = False

    def __post_init__(self):
        self.crps_training = None  # unused, kept for backwards compatibility
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
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ) -> StandardNormalizer:
        if extra_names is None:
            extra_names = []
        if extra_residual_scaled_names is None:
            extra_residual_scaled_names = []
        return self.normalization.get_loss_normalizer(
            names=self._normalize_names + extra_names,
            residual_scaled_names=self.prognostic_names + extra_residual_scaled_names,
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
    def input_names(self) -> list[str]:
        """
        Names of variables required as inputs to `step`,
        either in `input` or `next_step_input_data`.
        """
        if self.ocean is None:
            return self.in_names
        else:
            return list(set(self.in_names).union(self.ocean.forcing_names))

    def get_next_step_forcing_names(self) -> list[str]:
        """Names of input-only variables which come from the output timestep."""
        return self.next_step_forcing_names

    @property
    def diagnostic_names(self) -> list[str]:
        """Names of variables which are outputs only."""
        return list(set(self.out_names).difference(self.in_names))

    @property
    def output_names(self) -> list[str]:
        return self.out_names

    @property
    def next_step_input_names(self) -> list[str]:
        """Names of variables provided in next_step_input_data."""
        input_only_names = set(self.input_names).difference(self.output_names)
        if self.ocean is None:
            return list(input_only_names)
        return list(input_only_names.union(self.ocean.forcing_names))

    @property
    def loss_names(self) -> list[str]:
        return self.output_names

    def replace_ocean(self, ocean: OceanConfig | None):
        """
        Replace the ocean model with a new one.

        Args:
            ocean: The new ocean model configuration or None.
        """
        self.ocean = ocean

    def get_ocean(self) -> OceanConfig | None:
        return self.ocean

    @classmethod
    def _remove_deprecated_keys(cls, state: dict[str, Any]) -> dict[str, Any]:
        state_copy = state.copy()
        return state_copy

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
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
            init_weights=init_weights,
        )

    def load(self):
        self.normalization.load()


class SingleModuleStep(StepABC):
    """
    Step class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SingleModuleStepConfig,
        img_shape: tuple[int, int],
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        timestep: datetime.timedelta,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        """
        Args:
            config: The configuration.
            img_shape: Shape of domain as (n_lat, n_lon).
            corrector: The corrector to use at the end of each step.
            normalizer: The normalizer to use.
            timestep: Timestep of the model.
            init_weights: Function to initialize the weights of the module.
        """
        super().__init__()
        n_in_channels = len(config.in_names)
        n_out_channels = len(config.out_names)
        self.in_packer = Packer(config.in_names)
        self.out_packer = Packer(config.out_names)
        self._normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Ocean | None = config.ocean.build(
                config.in_names, config.out_names, timestep
            )
        else:
            self.ocean = None

        self.module = config.builder.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=img_shape,
        ).to(get_device())

        capture_stream = None
        dist=Distributed.get_instance()
        if dist.is_spatial_distributed():
          if using_gpu():
            capture_stream = torch.Stream(device="cuda")
          with torch.cuda.stream(capture_stream):
            self.module = init_gradient_reduction_hooks(
                        self.module,
                        device=get_device(),
                        #FIXME: I am not sure how to set reduction_buffer_count
                        reduction_buffer_count=1,
                        broadcast_buffers=False,
                        find_unused_parameters=False,
                        gradient_as_bucket_view=True,
                        static_graph=False,
                        verbose=True,
                    )
          # capture stream sync
          if capture_stream is not None:
            capture_stream.synchronize()
        init_weights([self.module])
        self._img_shape = img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        dist = Distributed.get_instance()
        self.module = dist.wrap_module(self.module)

        self._timestep = timestep

        self._corrector = corrector
        self.in_names = config.in_names
        self.out_names = config.out_names

    @property
    def config(self) -> SingleModuleStepConfig:
        return self._config

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._normalizer

    @property
    def surface_temperature_name(self) -> str | None:
        if self._config.ocean is not None:
            return self._config.ocean.surface_temperature_name
        return None

    @property
    def ocean_fraction_name(self) -> str | None:
        if self._config.ocean is not None:
            return self._config.ocean.ocean_fraction_name
        return None

    def prescribe_sst(
        self,
        mask_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> TensorDict:
        if self.ocean is None:
            raise RuntimeError(
                "The Ocean interface is missing but required to prescribe "
                "sea surface temperature."
            )
        return self.ocean.prescriber(mask_data, gen_data, target_data)

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
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
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
            wrapper: Wrapper to apply over each nn.Module before calling.

        Returns:
            The denormalized output data at the next time step.
        """

        def network_call(input_norm: TensorDict) -> TensorDict:
            input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
            output_tensor = wrapper(self.module)(input_tensor)
            return self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)

        return step_with_adjustments(
            input=input,
            next_step_input_data=next_step_input_data,
            network_calls=network_call,
            normalizer=self.normalizer,
            corrector=self._corrector,
            ocean=self.ocean,
            residual_prediction=self._config.residual_prediction,
            prognostic_names=self.prognostic_names,
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

    def load_state(self, state: dict[str, Any]) -> None:
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


def step_with_adjustments(
    input: TensorMapping,
    next_step_input_data: TensorMapping,
    network_calls: Callable[[TensorDict], TensorDict],
    normalizer: StandardNormalizer,
    corrector: CorrectorABC,
    ocean: Ocean | None,
    residual_prediction: bool,
    prognostic_names: list[str],
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
        network_calls: Callable[[TensorMapping], TensorDict] that takes a
            normalized input and returns a normalized output.
        normalizer: The normalizer to use.
        corrector: The corrector to use at the end of each step.
        ocean: The ocean model to use.
        residual_prediction: Whether to use residual prediction.
        prognostic_names: Names of prognostic variables.

    Returns:
        The denormalized output data at the next time step.
    """
    input_norm = normalizer.normalize(input)
    output_norm = network_calls(input_norm)
    if residual_prediction:
        output_norm = add_names(input_norm, output_norm, prognostic_names)
    output = normalizer.denormalize(output_norm)
    if corrector is not None:
        output = corrector(input, output, next_step_input_data)
    if ocean is not None:
        output = ocean(input, output, next_step_input_data)
    return output
