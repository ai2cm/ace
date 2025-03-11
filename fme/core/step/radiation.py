import dataclasses
import datetime
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import dacite
import torch
from torch import nn

from fme.ace.requirements import DataRequirements
from fme.core.coordinates import SerializableVerticalCoordinate, VerticalCoordinate
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset.utils import decode_timestep, encode_timestep
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
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


@StepSelector.register("separate_radiation")
@dataclasses.dataclass
class SeparateRadiationStepConfig(StepConfigABC):
    """
    Configuration for a separate radiation stepper.

    Parameters:
        builder: The module builder.
        radiation_builder: The radiation module builder.
        main_prognostic_names: Names of prognostic variables. These are provided
            as input to both the main and radiation models, and output by
            the main model.
        shared_forcing_names: Names of forcing variables.
        radiation_only_forcing_names: Names of forcing variables for the radiation
            model, in addition to the ones specified in `shared_forcing_names`.
        radiation_diagnostic_names: Names of diagnostic variables for the radiation
            model.
        main_diagnostic_names: Names of diagnostic variables for the main model.
        normalization: The normalization configuration.
        next_step_forcing_names: Names of forcing variables which come from
            the output timestep.
        ocean: The ocean configuration.
        loss: The loss configuration.
        corrector: The corrector configuration.
        residual_normalization: Optional alternative to configure loss normalization.
            If provided, it will be used for all *prognostic* variables in loss scaling.
        activation_checkpointing: Configuration for activation checkpointing to trade
            increased computation for lowered memory during training.
    """

    builder: ModuleSelector
    radiation_builder: ModuleSelector
    main_prognostic_names: List[str]
    shared_forcing_names: List[str]
    radiation_only_forcing_names: List[str]
    radiation_diagnostic_names: List[str]
    main_diagnostic_names: List[str]
    normalization: NormalizationConfig
    next_step_forcing_names: List[str] = dataclasses.field(default_factory=list)
    ocean: Optional[OceanConfig] = None
    loss: WeightedMappingLossConfig = dataclasses.field(
        default_factory=lambda: WeightedMappingLossConfig()
    )
    corrector: Union[AtmosphereCorrectorConfig, CorrectorSelector] = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    residual_normalization: Optional[NormalizationConfig] = None
    activation_checkpointing: ActivationCheckpointingConfig = dataclasses.field(
        default_factory=lambda: ActivationCheckpointingConfig()
    )

    def __post_init__(self):
        seen_names: Dict[str, str] = {}
        for name_list, label in (
            (self.main_prognostic_names, "main_prognostic_names"),
            (self.shared_forcing_names, "shared_forcing_names"),
            (self.radiation_only_forcing_names, "radiation_only_forcing_names"),
            (self.main_diagnostic_names, "main_diagnostic_names"),
            (self.radiation_diagnostic_names, "radiation_diagnostic_names"),
        ):
            for name in name_list:
                if name in seen_names:
                    raise ValueError(
                        f"Name '{name}' appears in multiple name lists: "
                        f"{seen_names[name]} and {label}."
                    )
            seen_names[name] = label
        for name in self.next_step_forcing_names:
            if name not in self.forcing_names:
                raise ValueError(
                    "next_step_forcing_name not in forcing_names: "
                    f"'{name}' not in {self.forcing_names}"
                )

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    def get_evaluation_window_data_requirements(
        self, n_forward_steps: int
    ) -> DataRequirements:
        return DataRequirements(
            names=self._all_names,
            n_timesteps=self._window_steps_required(n_forward_steps),
        )

    def _window_steps_required(self, n_forward_steps: int) -> int:
        return n_forward_steps + self.n_ic_timesteps

    def get_state(self):
        return dataclasses.asdict(self)

    def get_base_weights(self) -> Optional[List[Mapping[str, Any]]]:
        """
        If the model is being initialized from another model's weights for fine-tuning,
        returns those weights. Otherwise, returns None.

        The list mirrors the order of `modules` in the `SeparateRadiationStepper` class.
        """
        return None

    def get_step(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        timestep: datetime.timedelta,
    ) -> "SeparateRadiationStep":
        logging.info("Initializing stepper from provided config")
        corrector = vertical_coordinate.build_corrector(
            config=self.corrector,
            gridded_operations=gridded_operations,
            timestep=timestep,
        )
        normalizer = self.normalization.build(self.normalize_names)
        return SeparateRadiationStep(
            config=self,
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            corrector=corrector,
            normalizer=normalizer,
            timestep=timestep,
        )

    @classmethod
    def from_state(cls, state) -> "SeparateRadiationStepConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _all_names(self) -> List[str]:
        """Names of all variables required, including auxiliary ones."""
        extra_names = []
        if self.ocean is not None:
            extra_names.extend(self.ocean.forcing_names)
        all_names = set()
        for names in (
            self.main_prognostic_names,
            self.shared_forcing_names,
            self.radiation_only_forcing_names,
            self.main_diagnostic_names,
            self.radiation_diagnostic_names,
            extra_names,
        ):
            all_names.update(names)
        return list(all_names)

    @property
    def normalize_names(self) -> List[str]:
        """Names of variables which require normalization. I.e. inputs/outputs."""
        all_names = set()
        for names in (
            self.main_prognostic_names,
            self.shared_forcing_names,
            self.radiation_only_forcing_names,
            self.main_diagnostic_names,
            self.radiation_diagnostic_names,
        ):
            all_names.update(names)
        return list(all_names)

    @property
    def prognostic_names(self) -> List[str]:
        """Names of variables which both inputs and outputs."""
        return self.main_prognostic_names

    @property
    def forcing_names(self) -> List[str]:
        return list(
            set(self.shared_forcing_names).union(self.radiation_only_forcing_names)
        )

    @property
    def diagnostic_names(self) -> List[str]:
        return list(
            set(self.main_diagnostic_names).union(self.radiation_diagnostic_names)
        )

    @property
    def main_in_names(self) -> List[str]:
        return self.main_prognostic_names + self.shared_forcing_names

    @property
    def main_out_names(self) -> List[str]:
        return self.main_prognostic_names + self.main_diagnostic_names

    @property
    def radiation_in_names(self) -> List[str]:
        return self.shared_forcing_names + self.radiation_only_forcing_names

    @property
    def radiation_out_names(self) -> List[str]:
        return self.radiation_diagnostic_names

    @property
    def input_names(self) -> List[str]:
        return (
            self.main_prognostic_names
            + self.shared_forcing_names
            + self.radiation_only_forcing_names
        )

    @property
    def output_names(self) -> List[str]:
        return (
            self.main_prognostic_names
            + self.main_diagnostic_names
            + self.radiation_diagnostic_names
        )


class SeparateRadiationStep(StepABC):
    """
    Step class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SeparateRadiationStepConfig,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        timestep: datetime.timedelta,
    ):
        """
        Args:
            config: The configuration.
            img_shape: Shape of domain as (n_lat, n_lon).
            gridded_operations: The gridded operations, e.g. for area weighting.
            vertical_coordinate: The vertical coordinate.
            corrector: The corrector to use at the end of each step.
            normalizer: The normalizer to use.
            timestep: Timestep of the model.
        """
        self._gridded_operations = gridded_operations  # stored for serializing
        self.in_packer = Packer(config.main_in_names)
        self.out_packer = Packer(config.main_out_names)
        self.radiation_in_packer = Packer(config.radiation_in_names)
        self.radiation_out_packer = Packer(config.radiation_out_names)
        self._normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Optional[Ocean] = config.ocean.build(
                config.input_names,
                config.output_names,
                timestep,
            )
        else:
            self.ocean = None
        self.module: nn.Module = config.builder.build(
            n_in_channels=len(config.main_in_names),
            n_out_channels=len(config.main_out_names),
            img_shape=img_shape,
        ).to(get_device())
        self.radiation_module: nn.Module = config.radiation_builder.build(
            n_in_channels=len(config.radiation_in_names),
            n_out_channels=len(config.radiation_out_names),
            img_shape=img_shape,
        ).to(get_device())
        self._img_shape = img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        dist = Distributed.get_instance()
        self.module = dist.wrap_module(self.module)
        self.radiation_module = dist.wrap_module(self.radiation_module)
        self._vertical_coordinate: VerticalCoordinate = vertical_coordinate.to(
            get_device()
        )
        self._timestep = timestep
        self._corrector = corrector
        self._activation_checkpointing = config.activation_checkpointing

    @property
    def vertical_coordinate(self) -> VerticalCoordinate:
        return self._vertical_coordinate

    @property
    def timestep(self) -> datetime.timedelta:
        return self._timestep

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

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        """
        Replace the ocean model with a new one.

        Args:
            ocean: The new ocean model configuration or None.
        """
        self._config.ocean = ocean
        if ocean is None:
            self.ocean = ocean
        else:
            self.ocean = ocean.build(
                self.input_names, self.output_names, self._timestep
            )

    @property
    def prognostic_names(self) -> List[str]:
        return self._config.prognostic_names

    @property
    def forcing_names(self) -> List[str]:
        return self._config.forcing_names

    @property
    def diagnostic_names(self) -> List[str]:
        return self._config.diagnostic_names

    @property
    def next_step_input_names(self) -> List[str]:
        if self.ocean is None:
            return list(self.forcing_names)
        return list(set(self.forcing_names).union(self.ocean.forcing_names))

    @property
    def next_step_forcing_names(self) -> List[str]:
        return self._config.next_step_forcing_names

    @property
    def n_ic_timesteps(self) -> int:
        return self._config.n_ic_timesteps

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._normalizer

    @property
    def modules(self) -> nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return nn.ModuleList([self.module])

    def validate_inference_data(self, data: InferenceDataProtocol):
        if self._timestep != data.timestep:
            raise ValueError(
                f"Timestep of step object, {self._timestep}, does not "
                f"match that of the inference data, {data.timestep}."
            )

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
                [n_batch, n_lat, n_lon]. This data is used as input for `self.module`
                and is assumed to contain all input variables and be denormalized.
            next_step_forcing_data: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This must contain the necessary forcing
                data at the output timestep for the ocean model and corrector.
            use_activation_checkpointing: If True, wrap the module call with
                torch.utils.checkpoint.checkpoint, reducing memory consumption
                in exchange for increased computation. This is only relevant during
                training and otherwise has no effect.

        Returns:
            The denormalized output data at the next time step.
        """
        input_norm = self.normalizer.normalize(input)
        radiation_input_tensor = self.radiation_in_packer.pack(
            input_norm, axis=self.CHANNEL_DIM
        )
        if use_activation_checkpointing:
            radiation_output_tensor = torch.utils.checkpoint.checkpoint(
                self.radiation_module,
                radiation_input_tensor,
                use_reentrant=False,
                **self._activation_checkpointing.kwargs,
            )
        else:
            radiation_output_tensor = self.radiation_module(radiation_input_tensor)
        radiation_output_norm = self.radiation_out_packer.unpack(
            radiation_output_tensor, axis=self.CHANNEL_DIM
        )
        input_tensor = self.in_packer.pack(
            {**input_norm, **radiation_output_norm}, axis=self.CHANNEL_DIM
        )
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
        output = self.normalizer.denormalize({**radiation_output_norm, **output_norm})
        if self._corrector is not None:
            output = self._corrector(input, output, next_step_forcing_data)
        if self.ocean is not None:
            output = self.ocean(input, output, next_step_forcing_data)
        return output

    def get_regularizer_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "module": self.module.state_dict(),
            "radiation_module": self.radiation_module.state_dict(),
            "normalizer": self.normalizer.get_state(),
            "img_shape": self._img_shape,
            "config": self._config.get_state(),
            "gridded_operations": self._gridded_operations.to_state(),
            "vertical_coordinate": self._vertical_coordinate.as_dict(),
            "encoded_timestep": encode_timestep(self._timestep),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
        """
        if "module" in state:
            module = state["module"]
            if "module.device_buffer" in module:
                # for backwards compatibility with old checkpoints
                del module["module.device_buffer"]
            self.module.load_state_dict(module)
        self.radiation_module.load_state_dict(state["radiation_module"])

    @classmethod
    def from_state(cls, state) -> "SeparateRadiationStep":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.

        Returns:
            The stepper.
        """
        config_data = {**state["config"]}  # make a copy to avoid mutating input
        config_data["normalization"] = state["normalizer"]

        gridded_operations = GriddedOperations.from_state(state["gridded_operations"])

        vertical_coordinate: VerticalCoordinate = dacite.from_dict(
            data_class=SerializableVerticalCoordinate,
            data={"vertical_coordinate": state["vertical_coordinate"]},
            config=dacite.Config(strict=True),
        ).vertical_coordinate

        encoded_timestep = state.get("encoded_timestep", DEFAULT_ENCODED_TIMESTEP)
        timestep = decode_timestep(encoded_timestep)
        img_shape = state["img_shape"]
        config = SeparateRadiationStepConfig.from_state(config_data)
        corrector = vertical_coordinate.build_corrector(
            config=config.corrector,
            gridded_operations=gridded_operations,
            timestep=timestep,
        )
        step = cls(
            config=config,
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            corrector=corrector,
            timestep=timestep,
            normalizer=config.normalization.build(config.normalize_names),
        )
        step.load_state(state)
        return step
