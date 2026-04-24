import dataclasses
import logging
from collections.abc import Callable
from typing import Any

import dacite
import torch
from torch import nn

from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.normalizer import NetworkAndLossNormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.packer import Packer
from fme.core.registry import CorrectorSelector, ModuleSelector
from fme.core.registry.module import Module
from fme.core.step.args import StepArgs
from fme.core.step.secondary_decoder import (
    NoSecondaryDecoder,
    SecondaryDecoder,
    SecondaryDecoderConfig,
)
from fme.core.step.single_module import step_with_adjustments
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping


@StepSelector.register("secondary_module")
@dataclasses.dataclass
class SecondaryModuleStepConfig(StepConfigABC):
    """
    Configuration for a stepper with a primary and secondary module.

    The primary module (builder) produces the main output variables. The
    secondary module (secondary_builder) receives the same input and produces
    additional full-field outputs and/or residual corrections.

    Parameters:
        builder: The primary module builder.
        in_names: Names of input variables.
        out_names: Names of output variables from the primary module.
        normalization: The normalization configuration.
        secondary_builder: Builder for the secondary network that receives
            the same input as the primary module.
        secondary_out_names: Names of variables output by the secondary network
            as full fields (used directly as output). Must not overlap with
            out_names.
        secondary_residual_out_names: Names of variables for which the secondary
            network predicts a residual correction. If the name is also in
            out_names, the residual is added to the backbone's output;
            otherwise it is added to the (normalized) input value.
        secondary_decoder: Configuration for the secondary decoder that computes
            additional diagnostic variables from outputs.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Names of forcing variables for the next timestep.
        prescribed_prognostic_names: Prognostic variable names to overwrite from
            forcing data at each step (e.g. for inference with observed values).
        residual_prediction: Whether to use residual prediction.
    """

    builder: ModuleSelector
    in_names: list[str]
    out_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    secondary_builder: ModuleSelector
    secondary_out_names: list[str] = dataclasses.field(default_factory=list)
    secondary_residual_out_names: list[str] = dataclasses.field(default_factory=list)
    secondary_decoder: SecondaryDecoderConfig | None = None
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    prescribed_prognostic_names: list[str] = dataclasses.field(default_factory=list)
    residual_prediction: bool = False

    def __post_init__(self):
        for name in self.prescribed_prognostic_names:
            if name not in self.out_names:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in out_names: "
                    f"{self.out_names}"
                )
        for name in self.next_step_forcing_names:
            if name not in self.in_names:
                raise ValueError(
                    f"next_step_forcing_name '{name}' not in in_names: {self.in_names}"
                )
            if name in self.out_names:
                raise ValueError(
                    f"next_step_forcing_name is an output variable: '{name}'"
                )
        all_secondary_names = set(self.secondary_out_names) | set(
            self.secondary_residual_out_names
        )
        if self.secondary_decoder is not None:
            for name in self.secondary_decoder.secondary_diagnostic_names:
                if name in self.in_names:
                    raise ValueError(
                        f"secondary_diagnostic_name is an input variable: '{name}'"
                    )
                if name in set(self.out_names) | all_secondary_names:
                    raise ValueError(
                        f"secondary_diagnostic_name is an output variable: '{name}'"
                    )
        if not self.secondary_out_names and not self.secondary_residual_out_names:
            raise ValueError(
                "at least one of secondary_out_names or "
                "secondary_residual_out_names must be non-empty"
            )
        overlap = set(self.secondary_out_names) & set(self.out_names)
        if overlap:
            raise ValueError(
                f"secondary_out_names must not overlap with out_names. "
                f"Overlap: {overlap}"
            )
        overlap = set(self.secondary_out_names) & set(self.secondary_residual_out_names)
        if overlap:
            raise ValueError(
                f"secondary_out_names must not overlap with "
                f"secondary_residual_out_names. Overlap: {overlap}"
            )
        for name in self.secondary_residual_out_names:
            if name not in self.out_names and name not in self.in_names:
                raise ValueError(
                    f"secondary_residual_out_name '{name}' must be in "
                    f"out_names or in_names: {self.out_names}, {self.in_names}"
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
    def from_state(cls, state) -> "SecondaryModuleStepConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _normalize_names(self):
        """Names of variables which require normalization. I.e. inputs/outputs."""
        return list(set(self.in_names).union(self.output_names))

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
        return list(set(self.output_names).difference(self.in_names))

    @property
    def output_names(self) -> list[str]:
        secondary_decoder_names = (
            self.secondary_decoder.secondary_diagnostic_names
            if self.secondary_decoder is not None
            else []
        )
        return list(
            set(self.out_names)
            .union(secondary_decoder_names)
            .union(self.secondary_out_names)
            .union(self.secondary_residual_out_names)
        )

    @property
    def next_step_input_names(self) -> list[str]:
        """Names of variables provided in next_step_input_data."""
        input_only_names = set(self.input_names).difference(self.output_names)
        result = set(input_only_names)
        if self.ocean is not None:
            result = result.union(self.ocean.forcing_names)
        result = result.union(self.prescribed_prognostic_names)
        return list(result)

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

    def replace_prescribed_prognostic_names(self, names: list[str]) -> None:
        """Replace prescribed prognostic names (e.g. when loading from checkpoint)."""
        for name in names:
            if name not in self.out_names:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in out_names: "
                    f"{self.out_names}"
                )
        self.prescribed_prognostic_names = names

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "SecondaryModuleStep":
        logging.info("Initializing stepper from provided config")
        corrector = self.corrector.get_corrector(dataset_info)
        normalizer = self.normalization.get_network_normalizer(self._normalize_names)
        return SecondaryModuleStep(
            config=self,
            dataset_info=dataset_info,
            corrector=corrector,
            normalizer=normalizer,
            init_weights=init_weights,
        )

    def load(self):
        self.normalization.load()


class SecondaryModuleStep(StepABC):
    """
    Step class with a primary and secondary pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SecondaryModuleStepConfig,
        dataset_info: DatasetInfo,
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        """
        Args:
            config: The configuration.
            dataset_info: Information about the dataset.
            corrector: The corrector to use at the end of each step.
            normalizer: The normalizer to use.
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
                config.in_names, config.out_names, dataset_info.timestep
            )
        else:
            self.ocean = None
        module = config.builder.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
        )
        self.module = module.to(get_device())

        dist = Distributed.get_instance()

        all_secondary_names = (
            config.secondary_out_names + config.secondary_residual_out_names
        )
        secondary_module = config.secondary_builder.build(
            n_in_channels=n_in_channels,
            n_out_channels=len(all_secondary_names),
            dataset_info=dataset_info,
        )
        self.secondary_module: Module = secondary_module.to(get_device())
        self.secondary_out_packer: Packer = Packer(all_secondary_names)

        if config.secondary_decoder is not None:
            self.secondary_decoder: SecondaryDecoder | NoSecondaryDecoder = (
                config.secondary_decoder.build(
                    n_in_channels=n_out_channels,
                ).to(get_device())
            )
        else:
            self.secondary_decoder = NoSecondaryDecoder()

        init_weights(self.modules)
        self._img_shape = dataset_info.img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        self.module = self.module.wrap_module(dist.wrap_module)
        self.secondary_module = self.secondary_module.wrap_module(dist.wrap_module)
        self.secondary_decoder = self.secondary_decoder.wrap_module(dist.wrap_module)
        self._timestep = dataset_info.timestep

        self._corrector = corrector
        self.in_names = config.in_names
        self.out_names = config.out_names

    @property
    def config(self) -> SecondaryModuleStepConfig:
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
        modules = [self.module.torch_module]
        modules.append(self.secondary_module.torch_module)
        modules.extend(self.secondary_decoder.torch_modules)
        return nn.ModuleList(modules)

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> TensorDict:
        """
        Step the model forward one timestep given input data.

        Args:
            args: The arguments to the step function.
            wrapper: Wrapper to apply over each nn.Module before calling.

        Returns:
            The denormalized output data at the next time step.
        """

        def network_call(input_norm: TensorDict) -> TensorDict:
            input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
            output_tensor = self.module.wrap_module(wrapper)(
                input_tensor,
                labels=args.labels,
            )
            output_dict = self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)
            secondary_tensor = self.secondary_module.wrap_module(wrapper)(
                input_tensor,
                labels=args.labels,
            )
            secondary_dict = self.secondary_out_packer.unpack(
                secondary_tensor, axis=self.CHANNEL_DIM
            )
            for name in self._config.secondary_out_names:
                output_dict[name] = secondary_dict[name]
            for name in self._config.secondary_residual_out_names:
                if name in output_dict:
                    output_dict[name] = output_dict[name] + secondary_dict[name]
                else:
                    output_dict[name] = input_norm[name] + secondary_dict[name]
            secondary_output_dict = self.secondary_decoder.wrap_module(wrapper)(
                output_tensor.detach()  # detach avoids changing base outputs
            )
            output_dict.update(secondary_output_dict)
            return output_dict

        return step_with_adjustments(
            input=args.input,
            next_step_input_data=args.next_step_input_data,
            network_calls=network_call,
            normalizer=self.normalizer,
            corrector=self._corrector,
            ocean=self.ocean,
            residual_prediction=self._config.residual_prediction,
            prognostic_names=self.prognostic_names,
            prescribed_prognostic_names=self._config.prescribed_prognostic_names,
        )

    def get_regularizer_loss(self):
        return torch.tensor(0.0)

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "module": self.module.get_state(),
            "secondary_module": self.secondary_module.get_state(),
            "secondary_decoder": self.secondary_decoder.get_module_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
        """
        self.module.load_state(state["module"])
        self.secondary_module.load_state(state["secondary_module"])
        if "secondary_decoder" in state:
            self.secondary_decoder.load_module_state(state["secondary_decoder"])
